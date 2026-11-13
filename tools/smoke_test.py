#!/usr/bin/env python3
"""Smoke test สำหรับ nix.ipynb — รันได้ทั้งบน Python 3.10/3.11/3.12

เช็ก 3 ชั้น:
1. notebook เป็น JSON/nbformat ที่ถูกต้อง + เซลล์โค้ดคอมไพล์ผ่าน (จับ syntax ที่พังเฉพาะบางเวอร์ชัน)
2. ฟังก์ชันสำคัญยังทำงานถูก (unit test แบบดึงเฉพาะฟังก์ชันออกมาทดสอบ ไม่รัน side effect)
3. ไม่มีรูปแบบโค้ดที่เคยเป็นบั๊กกลับมา (extractall เปล่า ๆ, pkill -f llama-server, f-string ที่มี backslash)

ใช้: python3 tools/smoke_test.py
"""
from __future__ import annotations

import ast
import io
import json
import re
import shutil
import sys
import tarfile
import tokenize
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "nix.ipynb"
FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  ✅ {name}")
    except AssertionError as e:
        FAILURES.append(f"{name}: {e}")
        print(f"  ❌ {name}: {e}")
    except Exception as e:  # noqa: BLE001 - รายงานทุกอย่างให้เห็น
        FAILURES.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  ❌ {name}: {type(e).__name__}: {e}")


def load_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "ไม่พบเซลล์โค้ดใน notebook"
    return "\n".join("".join(c["source"]) for c in code_cells)


def extract_functions(src: str, names: set[str]) -> dict:
    """ดึงเฉพาะฟังก์ชันที่ต้องการออกมา exec ใน namespace ปลอม (ไม่รันโค้ดระดับบนสุด)"""
    tree = ast.parse(src)
    ns: dict = {
        "__name__": "smoke",
        "Path": Path,
        "re": re,
        "sys": sys,
        "tarfile": tarfile,
        "zipfile": zipfile,
        "shlex": __import__("shlex"),
        "rprint": lambda *a, **k: None,
        "check_gpu_and_recommend": lambda verbose=False: {
            "has_gpu": True,
            "gpu_name": "TEST",
            "vram_total_gb": 16.0,
            "vram_free_gb": 16.0,
            "ram_total_gb": 51.0,
            "ram_avail_gb": 40.0,
            "recommended_gpu_layers": 99,
            "recommended_ctx_size": 8192,
            "recommended_threads": 4,
        },
    }
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, f"<{node.name}>", "exec"), ns)  # noqa: S102 - ทดสอบโค้ดของเราเอง
            found.add(node.name)
    missing = names - found
    assert not missing, f"ไม่พบฟังก์ชัน: {sorted(missing)}"
    return ns


def fstring_backslash_expressions(src: str) -> list[str]:
    """หา f-string ที่มี backslash อยู่ใน {expression} (SyntaxError บน Python < 3.12)

    ใช้ tokenize แทน ast เพราะ ast ก่อน 3.12 ไม่ให้ตำแหน่งของ expression ใน f-string
    และนับเฉพาะวงเล็บปีกกา (ข้าม {{ }} ที่เป็นตัวอักษร) เพื่อไม่ให้ `[[ ... ]]` ของ shell
    ทำให้เข้าใจผิด
    """
    pat = re.compile(r"(?i)^([rubf]*)('''|\"\"\"|'|\")")
    bad: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.STRING:
            continue
        m = pat.match(tok.string)
        if not m or "f" not in m.group(1).lower():
            continue
        body = tok.string[m.end():-len(m.group(2))]
        depth, i = 0, 0
        while i < len(body):
            c = body[i]
            if c == "{" and body[i + 1:i + 2] == "{":
                i += 2
                continue
            if c == "}" and body[i + 1:i + 2] == "}":
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth = max(0, depth - 1)
            elif c == "\\":
                if depth > 0:
                    bad.append(tok.string)
                    break
                i += 2
                continue
            i += 1
    return bad


def test_notebook_shape() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, "nbformat ต้องเป็น 4"
    assert any(c["cell_type"] == "markdown" for c in nb["cells"]), "ต้องมีเซลล์ markdown (ปุ่ม Open in Colab)"
    assert any(c["cell_type"] == "code" for c in nb["cells"]), "ต้องมีเซลล์โค้ด"


def test_compiles() -> None:
    src = load_source()
    compile(src, str(NOTEBOOK), "exec")


def test_no_buggy_patterns() -> None:
    src = load_source()
    assert ".extractall(" not in src or "filter=" in src, "extractall ต้องมี filter= หรือผ่าน _safe_extract_*"
    for call in re.findall(r"tf\.extractall\(([^)]*)\)", src):
        assert "members=" in call or "filter=" in call, f"tar extractall ไม่ปลอดภัย: {call}"
    for call in re.findall(r"zf\.extractall\(([^)]*)\)", src):
        assert "root" in call, f"zip extractall ไม่ได้ส่ง path ที่ resolve แล้ว: {call}"
    assert "pkill -f llama-server" not in src, "ห้ามใช้ pkill -f llama-server (ฆ่า process อื่น)"
    # f-string ที่มี backslash ใน expression จะ SyntaxError บน Python < 3.12
    bad = [s for s in fstring_backslash_expressions(src)]
    assert not bad, f"f-string expression มี backslash (พังบน py<3.12): {bad[0][:70] if bad else ''}"


def test_pick_main_gguf() -> None:
    src = load_source()
    ns = extract_functions(src, {"_pick_main_gguf"})
    pick = ns["_pick_main_gguf"]
    tmp = Path(tempfile.mkdtemp())
    try:
        def mk(name: str, size: int) -> Path:
            p = tmp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                f.truncate(size)
            return p

        small = mk("m-Q4_K_M.gguf", 1 * 1024**3)
        big = mk("m-Q8_0.gguf", 8 * 1024**3)
        mk("mmproj-model-f16.gguf", 200_000)

        chosen, mm, size = pick(str(tmp), verbose=False)
        assert chosen == big, f"ควรเลือก quant ใหญ่สุดที่โหลดไหว ได้ {chosen.name}"
        assert mm is not None and "mmproj" in mm.name, "ควรเลือก mmproj แยกออกมา"
        assert 7 < size < 9, f"ขนาดที่คืนมาควรใกล้ 8GB ได้ {size}"

        chosen, _, _ = pick(str(tmp), prefer="q4_k_m", verbose=False)
        assert chosen == small, "prefer ต้องเลือก quant ตามที่ขอ"

        # งบหน่วยความจำเล็กมาก -> ถอยไปใช้ไฟล์ที่โหลดไหว
        tight = {"has_gpu": True, "vram_free_gb": 4.0, "ram_avail_gb": 8.0}
        chosen, _, _ = pick(str(tmp), hw=tight, verbose=False)
        assert chosen == small, f"งบจำกัดควรเลือกไฟล์เล็ก ได้ {chosen.name}"

        # shard ไม่ครบ -> ต้อง error ชัดเจน ไม่เงียบ
        shard_dir = tmp / "sharded"
        mk("sharded/m-00001-of-00003.gguf", 1_000)
        mk("sharded/m-00002-of-00003.gguf", 1_000)
        try:
            pick(str(shard_dir), verbose=False)
        except FileNotFoundError as e:
            assert "shard" in str(e), f"ข้อความ error ควรบอก shard ขาด: {e}"
        else:
            raise AssertionError("shard ไม่ครบต้อง raise FileNotFoundError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_safe_extract() -> None:
    src = load_source()
    ns = extract_functions(src, {"_safe_extract_tar", "_safe_extract_zip"})
    tmp = Path(tempfile.mkdtemp())
    try:
        tar_path = tmp / "evil.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo("../escaped-tar.txt")
            data = b"pwn"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                ns["_safe_extract_tar"](tf, tmp / "out")
        except RuntimeError:
            pass
        else:
            raise AssertionError("tar ที่มี ../ ต้องถูกปฏิเสธ")

        zip_path = tmp / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../escaped-zip.txt", "pwn")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                ns["_safe_extract_zip"](zf, tmp / "out2")
        except RuntimeError:
            pass
        else:
            raise AssertionError("zip ที่มี ../ ต้องถูกปฏิเสธ")

        good = tmp / "good.tar.gz"
        with tarfile.open(good, "w:gz") as tf:
            payload = b"ok"
            info = tarfile.TarInfo("pkg/readme.txt")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        with tarfile.open(good, "r:gz") as tf:
            ns["_safe_extract_tar"](tf, tmp / "out3")
        assert (tmp / "out3/pkg/readme.txt").read_bytes() == b"ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_uv_add_validation() -> None:
    src = load_source()
    ns = extract_functions(src, {"uv_add"})
    ns["run_quiet"] = lambda cmd: None
    ns["importlib"] = __import__("importlib")
    try:
        ns["uv_add"]("numpy; rm -rf /")
    except ValueError:
        pass
    else:
        raise AssertionError("uv_add ต้องปฏิเสธชื่อแพ็กเกจที่มีอักขระอันตราย")


def main() -> int:
    print(f"🧪 smoke test: {NOTEBOOK.name} (python {sys.version.split()[0]})")
    check("notebook structure", test_notebook_shape)
    check("code cell compiles", test_compiles)
    check("no known buggy patterns", test_no_buggy_patterns)
    check("_pick_main_gguf", test_pick_main_gguf)
    check("safe archive extraction", test_safe_extract)
    check("uv_add input validation", test_uv_add_validation)
    if FAILURES:
        print(f"\n❌ ล้มเหลว {len(FAILURES)} รายการ")
        return 1
    print("\n✅ ผ่านทั้งหมด")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
