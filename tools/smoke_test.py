#!/usr/bin/env python3
"""Smoke test สำหรับ nix.ipynb — รันได้ทั้งบน Python 3.10/3.11/3.12

เช็ก 3 ชั้น:
1. notebook เป็น JSON/nbformat ที่ถูกต้อง + เซลล์โค้ดคอมไพล์ผ่าน (จับ syntax ที่พังเฉพาะบางเวอร์ชัน)
2. ฟังก์ชัน/คลาสสำคัญยังทำงานถูก (ดึงเฉพาะส่วนที่ต้องการออกมา exec ไม่รัน side effect)
3. ไม่มีรูปแบบโค้ดที่เคยเป็นบั๊ก/ตัวถ่วงประสิทธิภาพกลับมา

ใช้: python3 tools/smoke_test.py
ต้องมี numpy สำหรับเทสต์ StrataRAG (ถ้าไม่มีจะข้ามพร้อมแจ้งเตือน)
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import tokenize
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "nix.ipynb"
FAILURES: list[str] = []
SKIPS: list[str] = []

try:  # numpy เป็น dependency ของ RAG เท่านั้น
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


class Skip(Exception):
    """ใช้เมื่อสภาพแวดล้อมไม่มี dependency ที่จำเป็น"""


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  ✅ {name}")
    except Skip as e:
        SKIPS.append(f"{name}: {e}")
        print(f"  ⏭️  {name} (ข้าม: {e})")
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


def _base_namespace() -> dict:
    return {
        "__name__": "smoke",
        "Path": Path,
        "re": re,
        "sys": sys,
        "io": io,
        "json": json,
        "hashlib": hashlib,
        "struct": struct,
        "time": __import__("time"),
        "tarfile": tarfile,
        "zipfile": zipfile,
        "shlex": __import__("shlex"),
        "np": np,
        "DEFAULT_RAG_PATH": "/tmp/rag_store_test.npz",
        "_DIR_CACHE": {},
        "_DIR_CACHE_TTL": 30.0,
        "_GGUF_LAYERS": {},
        "rprint": lambda *a, **k: None,
        "PdfReader": object,          # ไม่ได้ใช้ในเทสต์นี้
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


class FakeTextEmbedding:
    """embedder ปลอม: คืนเวกเตอร์ 3 มิติตามหมวดคำ เพื่อทดสอบ cosine search ได้จริง"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def embed(self, texts):
        for t in texts:
            low = str(t).lower()
            cat = [1.0, 0.0, 0.0] if "แมว" in low or "cat" in low else None
            dog = [0.0, 1.0, 0.0] if "หมา" in low or "dog" in low else None
            yield np.asarray(cat or dog or [0.0, 0.0, 1.0], dtype=np.float32)


def extract_defs(src: str, names: set[str]) -> dict:
    """ดึงเฉพาะฟังก์ชัน/คลาสที่ต้องการออกมา exec ใน namespace ปลอม (ไม่รันโค้ดระดับบนสุด)"""
    tree = ast.parse(src)
    ns = _base_namespace()
    ns["TextEmbedding"] = FakeTextEmbedding
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, f"<{node.name}>", "exec"), ns)  # noqa: S102 - ทดสอบโค้ดของเราเอง
            found.add(node.name)
    missing = names - found
    assert not missing, f"ไม่พบฟังก์ชัน: {sorted(missing)}"
    return ns


def function_source(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            if seg:
                return seg
    raise AssertionError(f"ไม่พบฟังก์ชัน {name}")


def fstring_backslash_expressions(src: str) -> list[str]:
    """หา f-string ที่มี backslash อยู่ใน {expression} (SyntaxError บน Python < 3.12)

    ใช้ tokenize แทน ast เพราะ ast ก่อน 3.12 ไม่ให้ตำแหน่งของ expression ใน f-string
    และนับเฉพาะวงเล็บปีกกา (ข้าม {{ }} ที่เป็นตัวอักษร) เพื่อไม่ให้ `[[ ... ]]` ของ shell
    ทำให้เข้าใจผิด
    """
    pat = re.compile(r"(?i)^([rubf]*)(\'\'\'|\"\"\"|'|\")")
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


# ------------------------------------------------------------------ 1. โครงสร้าง notebook
def test_notebook_shape() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, "nbformat ต้องเป็น 4"
    assert any(c["cell_type"] == "markdown" for c in nb["cells"]), "ต้องมีเซลล์ markdown (ปุ่ม Open in Colab)"
    assert any(c["cell_type"] == "code" for c in nb["cells"]), "ต้องมีเซลล์โค้ด"


def test_compiles() -> None:
    compile(load_source(), str(NOTEBOOK), "exec")


# ------------------------------------------------------------------ 2. รูปแบบโค้ดที่เคยพัง
def test_no_buggy_patterns() -> None:
    src = load_source()
    for call in re.findall(r"tf\.extractall\(([^)]*)\)", src):
        assert "members=" in call or "filter=" in call, f"tar extractall ไม่ปลอดภัย: {call}"
    for call in re.findall(r"zf\.extractall\(([^)]*)\)", src):
        assert "root" in call, f"zip extractall ไม่ได้ส่ง path ที่ resolve แล้ว: {call}"
    assert "pkill -f llama-server" not in src, "ห้ามใช้ pkill -f llama-server (ฆ่า process อื่น)"
    bad = fstring_backslash_expressions(src)
    assert not bad, f"f-string expression มี backslash (พังบน py<3.12): {bad[0][:70] if bad else ''}"


def test_performance_patterns() -> None:
    src = load_source()
    refresh = function_source(src, "_refresh")
    assert "_B(" not in refresh, "_refresh ต้องไม่อัปเดตด้วยการสร้าง widget ใหม่ทุกครั้ง (สร้างครั้งเดียวแล้วอัปเดต value)"
    assert 'rglob("*.gguf")' not in refresh, "_refresh ต้องใช้ _dir_stats() แบบ cache ไม่ใช่ rglob เอง"
    assert "_dir_stats(" in refresh, "_refresh ควรใช้ _dir_stats() แบบ cache"
    assert "last_draw" in src, "การสตรีมควร throttle การวาด HTML (~100ms) ไม่ใช่ทุก token"
    search = function_source(src, "search")
    assert "min_score" in search, "search() ควรกรองผลลัพธ์ที่ score ต่ำเกินไป"


# ------------------------------------------------------------------ 3. ฟังก์ชันพื้นฐาน
def test_pick_main_gguf() -> None:
    ns = extract_defs(load_source(), {"_pick_main_gguf"})
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

        tight = {"has_gpu": True, "vram_free_gb": 4.0, "ram_avail_gb": 8.0}
        chosen, _, _ = pick(str(tmp), hw=tight, verbose=False)
        assert chosen == small, f"งบจำกัดควรเลือกไฟล์เล็ก ได้ {chosen.name}"

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
    ns = extract_defs(load_source(), {"_safe_extract_tar", "_safe_extract_zip"})
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
    ns = extract_defs(load_source(), {"uv_add"})
    ns["run_quiet"] = lambda cmd: None
    ns["importlib"] = __import__("importlib")
    try:
        ns["uv_add"]("numpy; rm -rf /")
    except ValueError:
        pass
    else:
        raise AssertionError("uv_add ต้องปฏิเสธชื่อแพ็กเกจที่มีอักขระอันตราย")


# ------------------------------------------------------------------ 4. P2: cache / GGUF / RAG
def test_dir_stats_cache() -> None:
    ns = extract_defs(load_source(), {"_dir_stats", "_invalidate_dir_cache"})
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "m.gguf").write_bytes(b"x" * 100)
        first = ns["_dir_stats"](tmp)
        assert first["gguf"] == 1 and first["bytes"] == 100, f"อ่านค่าแรกผิด: {first}"
        (tmp / "second.gguf").write_bytes(b"y" * 50)
        cached = ns["_dir_stats"](tmp)
        assert cached["cached"] is True and cached["gguf"] == 1, "ควรคืนค่า cache ภายใน TTL"
        fresh = ns["_dir_stats"](tmp, ttl=0)
        assert fresh["cached"] is False and fresh["gguf"] == 2, f"ttl=0 ต้องสแกนใหม่: {fresh}"
        ns["_invalidate_dir_cache"]()
        assert ns["_dir_stats"](tmp)["gguf"] == 2, "หลัง invalidate ต้องสแกนใหม่"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_fake_gguf(path: Path, layers: int = 32) -> None:
    def gstr(text: str) -> bytes:
        b = text.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kvs = [
        gstr("general.architecture") + struct.pack("<I", 8) + gstr("llama"),
        gstr("llama.block_count") + struct.pack("<I", 4) + struct.pack("<I", layers),
        gstr("llama.attention.head_count") + struct.pack("<I", 4) + struct.pack("<I", 32),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs))


def test_gguf_layer_count() -> None:
    ns = extract_defs(load_source(), {"_gguf_layer_count", "_gguf_read_value"})
    tmp = Path(tempfile.mkdtemp())
    try:
        good = tmp / "good.gguf"
        _write_fake_gguf(good, layers=40)
        assert ns["_gguf_layer_count"](good) == 40, "ควรอ่าน block_count ได้ถูกต้อง"
        junk = tmp / "junk.gguf"
        junk.write_bytes(b"NOTGGUF" + b"\x00" * 64)
        assert ns["_gguf_layer_count"](junk) == 0, "ไฟล์ที่ไม่ใช่ GGUF ต้องคืน 0 (ไม่ crash)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rag_chunking() -> None:
    if np is None:
        raise Skip("ต้องมี numpy")
    ns = extract_defs(load_source(), {"StrataRAG"})
    Split = ns["StrataRAG"].split_chunks
    text = ("ประโยคแรกยาวพอสมควรเกี่ยวกับแมวและ собака. " * 3) + "\n\n" + ("Second paragraph about dogs and cats. " * 6)
    chunks = Split(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1, "ข้อความยาวต้องถูกแบ่งหลาย chunk"
    assert all(len(c) <= 120 for c in chunks), f"มี chunk ยาวเกินกำหนด: {[len(c) for c in chunks]}"
    assert all(c.strip() for c in chunks), "ต้องไม่มี chunk ว่าง"
    words_in = re.findall(r"[A-Za-zก-๙]+", text)
    words_out = re.findall(r"[A-Za-zก-๙]+", " ".join(chunks))
    assert set(words_out) <= set(words_in), "chunk ต้องไม่ตัดคำจนเกิดคำแปลกใหม่"
    # overlap=0 -> ต้องได้คำครบตามลำดับ ไม่ซ้ำ ไม่หาย
    plain = Split(text, chunk_size=120, overlap=0)
    assert re.findall(r"[A-Za-zก-๙]+", " ".join(plain)) == words_in, "overlap=0 ต้องได้คำครบตามลำดับเดิม"
    assert all(len(c) <= 120 for c in plain)
    assert Split("   ", chunk_size=100) == [], "ข้อความว่างต้องคืน list ว่าง"


def test_rag_dedup_search_persist() -> None:
    if np is None:
        raise Skip("ต้องมี numpy")
    ns = extract_defs(load_source(), {"StrataRAG"})
    cls = ns["StrataRAG"]
    rag = cls()
    added = rag.add_documents([{"source": "cats", "text": "แมว cat น่ารัก"}], chunk_size=500)
    assert added == 1, f"ควรเพิ่ม 1 chunk ได้ {added}"
    again = rag.add_documents([{"source": "cats", "text": "แมว cat น่ารัก"}], chunk_size=500)
    assert again == 0, "เนื้อหาซ้ำต้องไม่ถูกเพิ่มอีก"
    rag.add_documents([{"source": "dogs", "text": "หมา dog ซื่อสัตย์"}], chunk_size=500)
    assert len(rag.chunks) == 2
    assert rag.embeddings is not None and rag.embeddings.shape == (2, 3), f"embedding ผิดรูป: {rag.embeddings}"

    hits = rag.search("แมว", top_k=5)
    assert hits and hits[0]["source"] == "cats", f"ค้นหาไม่ตรง: {hits}"
    assert hits[0]["rank"] == 1 and hits[0]["score"] > 0.9
    assert len(hits) == 1, f"min_score ควรกรอง chunk ที่ไม่เกี่ยวออก: {hits}"
    assert rag.search("แมว", top_k=5, min_score=0.0), "min_score=0 ยังต้องได้ผลลัพธ์"

    path = Path(tempfile.mkdtemp()) / "store.npz"
    try:
        rag.save(path)
        other = cls()
        n = other.load(path)
        assert n == 2 and len(other.chunks) == 2, "โหลดคลังกลับมาไม่ครบ"
        assert other.search("หมา")[0]["source"] == "dogs", "ค้นหาหลังโหลดควรได้ผลเหมือนเดิม"
        assert other.add_documents([{"source": "cats", "text": "แมว cat น่ารัก"}], chunk_size=500) == 0, "หลังโหลดต้องกันข้อมูลซ้ำได้"
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


def main() -> int:
    print(f"🧪 smoke test: {NOTEBOOK.name} (python {sys.version.split()[0]})")
    check("notebook structure", test_notebook_shape)
    check("code cell compiles", test_compiles)
    check("no known buggy patterns", test_no_buggy_patterns)
    check("performance patterns (P2)", test_performance_patterns)
    check("_pick_main_gguf", test_pick_main_gguf)
    check("safe archive extraction", test_safe_extract)
    check("uv_add input validation", test_uv_add_validation)
    check("_dir_stats cache", test_dir_stats_cache)
    check("_gguf_layer_count", test_gguf_layer_count)
    check("StrataRAG chunking", test_rag_chunking)
    check("StrataRAG dedup/search/persist", test_rag_dedup_search_persist)
    if FAILURES:
        print(f"\n❌ ล้มเหลว {len(FAILURES)} รายการ")
        return 1
    if SKIPS:
        print(f"\n✅ ผ่านทั้งหมด (ข้าม {len(SKIPS)} รายการ)")
        return 0
    print("\n✅ ผ่านทั้งหมด")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
