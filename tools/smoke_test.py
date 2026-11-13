#!/usr/bin/env python3
"""Smoke test — ตรวจแพ็กเกจ strata/ + notebook โดยไม่ต้องใช้ GPU/Colab

เช็ก 4 ชั้น:
1. notebook ยังโครงถูก (ไม่มี output/state ค้าง, เซลล์โค้ดคอมไพล์ได้)
2. โค้ดทุกไฟล์ในแพ็กเกจคอมไพล์ผ่าน และไม่มี f-string ที่พังบน Python < 3.12
3. ฟังก์ชันสำคัญทำงานถูก (quant picker, safe extract, GGUF parser, cache, chunking, RAG persist)
4. ไม่มีรูปแบบโค้ดที่เคยเป็นบั๊ก/ตัวถ่วงประสิทธิภาพกลับมา

ใช้: python3 tools/smoke_test.py
ต้องมี: numpy (ถ้าไม่มีจะข้ามเทสต์ RAG)
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import tokenize
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "nix.ipynb"
PKG = ROOT / "strata"
FAILURES: list[str] = []
SKIPS: list[str] = []

sys.path.insert(0, str(ROOT))

try:
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


def package_sources() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(PKG.glob("*.py"))}


def fstring_backslash_expressions(src: str) -> list[str]:
    """หา f-string ที่มี backslash ใน {expression} (SyntaxError บน Python < 3.12)

    ใช้ tokenize เพราะ ast ก่อน 3.12 ไม่ให้ตำแหน่งของ expression ใน f-string
    และนับเฉพาะวงเล็บปีกกา (ข้าม {{ }} ที่เป็นตัวอักษร) เพื่อไม่ให้ `[[ ... ]]` ของ shell หลอก
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


# ------------------------------------------------------------------ 1. notebook
def test_notebook_shape() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, "nbformat ต้องเป็น 4"
    cells = nb["cells"]
    assert any(c["cell_type"] == "markdown" for c in cells), "ต้องมีเซลล์ markdown (ปุ่ม Open in Colab)"
    code = [c for c in cells if c["cell_type"] == "code"]
    assert code, "ต้องมีเซลล์โค้ด"
    for c in code:
        assert not c.get("outputs"), "notebook ไม่ควรมี output ค้าง (ไฟล์จะบวม) — clear output ก่อน commit"
        assert c.get("execution_count") is None, "ไม่ควรบันทึก execution_count"
        assert "widgets" not in c.get("metadata", {}), "ไม่ควรมี widget state ค้างใน notebook"
    size_kb = NOTEBOOK.stat().st_size / 1024
    assert size_kb < 200, f"notebook ใหญ่เกินไป ({size_kb:.0f} KB) — น่าจะมี output/state ค้าง"


def test_notebook_cells_compile() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        compile(src, f"nix.ipynb:cell{i}", "exec")


# ------------------------------------------------------------------ 2. package
def test_package_compiles() -> None:
    sources = package_sources()
    assert "config.py" in sources and "ui.py" in sources, f"โครงแพ็กเกจผิดปกติ: {sorted(sources)}"
    for name, src in sources.items():
        compile(src, name, "exec")
        bad = fstring_backslash_expressions(src)
        assert not bad, f"{name}: f-string มี backslash ใน expression (พังบน py<3.12): {bad[0][:70] if bad else ''}"


def test_public_api() -> None:
    import strata
    from strata.config import ALL_PRESET_KEYS, MODEL_CATALOG, SPEC_STATE

    assert strata.__version__
    assert len(ALL_PRESET_KEYS) == 10, f"preset ต้องมี 10 ตัว ได้ {len(ALL_PRESET_KEYS)}"
    assert len(MODEL_CATALOG) == 11, "MODEL_CATALOG ต้องมี 10 preset + custom"
    assert set(ALL_PRESET_KEYS) <= set(MODEL_CATALOG), "preset ทุกตัวต้องอยู่ใน catalog"
    assert "ctx_size" not in SPEC_STATE or isinstance(SPEC_STATE["ctx_size"], int)
    for key, cfg in MODEL_CATALOG.items():
        assert re.fullmatch(r"[\w.\-]+/[\w.\-]+", cfg["repo_id"]), f"{key}: repo_id ผิดรูป"
        assert isinstance(cfg["desc"], str) and cfg["desc"], f"{key}: ต้องมี desc"


def test_lazy_imports() -> None:
    """โมดูลต้อง import ได้โดยไม่ต้องมี fastembed/pypdf/openai ติดตั้ง"""
    code = (
        "import sys\n"
        "for m in ('fastembed', 'pypdf', 'openai'):\n"
        "    sys.modules.pop(m, None)\n"
        "import strata.config, strata.utils, strata.server, strata.download, strata.rag\n"
        "leaked = [m for m in ('fastembed', 'pypdf') if m in sys.modules]\n"
        "assert not leaked, f'import แล้วดึง dependency หนักทันที: {leaked}'\n"
    )
    ns: dict = {}
    exec(compile(code, "<lazy>", "exec"), ns)  # noqa: S102


def test_no_buggy_patterns() -> None:
    sources = package_sources()
    joined = "\n".join(sources.values())
    for call in re.findall(r"tf\.extractall\(([^)]*)\)", joined):
        assert "members=" in call or "filter=" in call, f"tar extractall ไม่ปลอดภัย: {call}"
    for call in re.findall(r"zf\.extractall\(([^)]*)\)", joined):
        assert "root" in call, f"zip extractall ไม่ได้ส่ง path ที่ resolve แล้ว: {call}"
    assert "pkill -f" not in joined, "ห้ามฆ่า process ด้วย pkill -f (กิน process อื่น)"
    assert "curl -sL https://nixos.org/nix/install | sh" not in joined, "ห้าม curl|sh ติดตั้ง Nix"
    assert '"/bin/bash", "-c"' in sources["utils.py"], "คำสั่ง shell ควรผ่าน run_quiet/run_stream เท่านั้น"
    # ต้องมี lock กันกดซ้ำ
    assert "_SERVER_LOCK" in sources["server.py"], "server ต้องมี lock กันสตาร์ทซ้อน"
    assert "_INSTALL_LOCK" in sources["download.py"], "download ต้องมี lock กันติดตั้งซ้อน"


def test_perf_patterns() -> None:
    ui = package_sources()["ui.py"]
    tree = ast.parse(ui)
    refresh = next(
        ast.get_source_segment(ui, n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_refresh"
    )
    assert refresh, "ไม่พบ _refresh"
    assert "_B(" not in refresh, "_refresh ต้องไม่สร้าง widget ใหม่ทุกครั้ง"
    assert 'rglob("*.gguf")' not in refresh, "_refresh ต้องใช้ _dir_stats() แบบ cache"
    assert "_dir_stats(" in refresh, "_refresh ควรใช้ _dir_stats() แบบ cache"
    assert "last_draw" in ui, "การสตรีมควร throttle การวาด HTML (~100ms)"
    rag = package_sources()["rag.py"]
    assert "min_score" in rag, "search() ควรกรองผลที่ score ต่ำเกินไป"


def test_no_dead_code() -> None:
    sources = package_sources()
    joined = "\n".join(sources.values())
    for name in ("list_model_files", "CUSTOM_INCLUDE", "TypeVar"):
        assert name not in joined, f"ยังมี dead code: {name}"


# ------------------------------------------------------------------ 3. ฟังก์ชันหลัก
def test_pick_main_gguf() -> None:
    if np is None:
        raise Skip("ต้องมี numpy")
    from strata.server import _pick_main_gguf

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

        big_hw = {"has_gpu": True, "vram_free_gb": 16.0, "ram_avail_gb": 40.0}
        chosen, mm, size = _pick_main_gguf(str(tmp), hw=big_hw, verbose=False)
        assert chosen == big, f"ควรเลือก quant ใหญ่สุดที่โหลดไหว ได้ {chosen.name}"
        assert mm is not None and "mmproj" in mm.name
        assert 7 < size < 9, f"ขนาดที่คืนมาควรใกล้ 8GB ได้ {size}"

        chosen, _, _ = _pick_main_gguf(str(tmp), hw=big_hw, prefer="q4_k_m", verbose=False)
        assert chosen == small, "prefer ต้องเลือก quant ตามที่ขอ"

        tight = {"has_gpu": True, "vram_free_gb": 4.0, "ram_avail_gb": 8.0}
        chosen, _, _ = _pick_main_gguf(str(tmp), hw=tight, verbose=False)
        assert chosen == small, f"งบจำกัดควรเลือกไฟล์เล็ก ได้ {chosen.name}"

        shard_dir = tmp / "sharded"
        mk("sharded/m-00001-of-00003.gguf", 1_000)
        mk("sharded/m-00002-of-00003.gguf", 1_000)
        try:
            _pick_main_gguf(str(shard_dir), verbose=False)
        except FileNotFoundError as e:
            assert "shard" in str(e), f"ข้อความ error ควรบอก shard ขาด: {e}"
        else:
            raise AssertionError("shard ไม่ครบต้อง raise FileNotFoundError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_safe_extract() -> None:
    from strata.utils import _safe_extract_tar, _safe_extract_zip

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
                _safe_extract_tar(tf, tmp / "out")
        except RuntimeError:
            pass
        else:
            raise AssertionError("tar ที่มี ../ ต้องถูกปฏิเสธ")

        zip_path = tmp / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../escaped-zip.txt", "pwn")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                _safe_extract_zip(zf, tmp / "out2")
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
            _safe_extract_tar(tf, tmp / "out3")
        assert (tmp / "out3/pkg/readme.txt").read_bytes() == b"ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_uv_add_validation() -> None:
    from strata import utils

    calls: list[str] = []
    original = utils.run_quiet
    utils.run_quiet = calls.append  # type: ignore[assignment]
    try:
        utils.uv_add("numpy; rm -rf /")
    except ValueError:
        pass
    else:
        raise AssertionError("uv_add ต้องปฏิเสธชื่อแพ็กเกจที่มีอักขระอันตราย")
    finally:
        utils.run_quiet = original  # type: ignore[assignment]
    assert not calls, "คำสั่งอันตรายไม่ควรถูกเรียกเลย"


def test_dir_stats_cache() -> None:
    from strata.utils import _dir_stats, _invalidate_dir_cache

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "m.gguf").write_bytes(b"x" * 100)
        first = _dir_stats(tmp)
        assert first["gguf"] == 1 and first["bytes"] == 100, f"อ่านค่าแรกผิด: {first}"
        (tmp / "second.gguf").write_bytes(b"y" * 50)
        cached = _dir_stats(tmp)
        assert cached["cached"] is True and cached["gguf"] == 1, "ควรคืนค่า cache ภายใน TTL"
        fresh = _dir_stats(tmp, ttl=0)
        assert fresh["cached"] is False and fresh["gguf"] == 2, f"ttl=0 ต้องสแกนใหม่: {fresh}"
        _invalidate_dir_cache()
        assert _dir_stats(tmp)["gguf"] == 2, "หลัง invalidate ต้องสแกนใหม่"
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
    from strata.server import _gguf_layer_count

    tmp = Path(tempfile.mkdtemp())
    try:
        good = tmp / "good.gguf"
        _write_fake_gguf(good, layers=40)
        assert _gguf_layer_count(good) == 40, "ควรอ่าน block_count ได้ถูกต้อง"
        junk = tmp / "junk.gguf"
        junk.write_bytes(b"NOTGGUF" + b"\x00" * 64)
        assert _gguf_layer_count(junk) == 0, "ไฟล์ที่ไม่ใช่ GGUF ต้องคืน 0 (ไม่ crash)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rag_chunking() -> None:
    if np is None:
        raise Skip("ต้องมี numpy")
    from strata.rag import StrataRAG

    text = ("ประโยคแรกยาวพอสมควรเกี่ยวกับแมวและ dog. " * 3) + "\n\n" + ("Second paragraph about dogs and cats. " * 6)
    chunks = StrataRAG.split_chunks(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1, "ข้อความยาวต้องถูกแบ่งหลาย chunk"
    assert all(len(c) <= 120 for c in chunks), f"มี chunk ยาวเกินกำหนด: {[len(c) for c in chunks]}"
    assert all(c.strip() for c in chunks), "ต้องไม่มี chunk ว่าง"
    words_in = re.findall(r"[A-Za-zก-๙]+", text)
    words_out = re.findall(r"[A-Za-zก-๙]+", " ".join(chunks))
    assert set(words_out) <= set(words_in), "chunk ต้องไม่ตัดคำจนเกิดคำแปลกใหม่"
    plain = StrataRAG.split_chunks(text, chunk_size=120, overlap=0)
    assert re.findall(r"[A-Za-zก-๙]+", " ".join(plain)) == words_in, "overlap=0 ต้องได้คำครบตามลำดับเดิม"
    assert StrataRAG.split_chunks("   ", chunk_size=100) == [], "ข้อความว่างต้องคืน list ว่าง"


class FakeEmbedding:
    """embedder ปลอม 3 มิติ: แมว/cat → แกน 0, หมา/dog → แกน 1, อื่น ๆ → แกน 2"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def embed(self, texts):
        for t in texts:
            low = str(t).lower()
            vec = [1.0, 0.0, 0.0] if ("แมว" in low or "cat" in low) else ([0.0, 1.0, 0.0] if ("หมา" in low or "dog" in low) else [0.0, 0.0, 1.0])
            yield np.asarray(vec, dtype=np.float32)


def test_rag_dedup_search_persist() -> None:
    if np is None:
        raise Skip("ต้องมี numpy")
    from strata.rag import StrataRAG

    rag = StrataRAG()
    rag._embedder = FakeEmbedding()
    assert rag.add_documents([{"source": "cats", "text": "แมว cat น่ารัก"}], chunk_size=500) == 1
    assert rag.add_documents([{"source": "cats", "text": "แมว cat น่ารัก"}], chunk_size=500) == 0, "ข้อมูลซ้ำต้องไม่ถูกเพิ่ม"
    rag.add_documents([{"source": "dogs", "text": "หมา dog ซื่อสัตย์"}], chunk_size=500)
    assert len(rag.chunks) == 2
    assert rag.embeddings is not None and rag.embeddings.shape == (2, 3), f"embedding ผิดรูป: {rag.embeddings}"

    hits = rag.search("แมว", top_k=5)
    assert hits and hits[0]["source"] == "cats", f"ค้นหาไม่ตรง: {hits}"
    assert hits[0]["score"] > 0.9
    assert len(hits) == 1, f"min_score ควรกรอง chunk ที่ไม่เกี่ยวออก: {hits}"

    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "store.npz"
        rag.save(path)
        other = StrataRAG()
        other._embedder = FakeEmbedding()
        assert other.load(path) == 2, "โหลดคลังกลับมาไม่ครบ"
        assert other.search("หมา")[0]["source"] == "dogs"
        assert other.add_documents([{"source": "cats", "text": "แมว cat น่ารัก"}], chunk_size=500) == 0, "หลังโหลดต้องกันซ้ำได้"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_server_lock_guard() -> None:
    """กดปุ่มซ้ำระหว่าง server กำลังทำงาน ต้องได้ error ที่อ่านรู้เรื่อง ไม่ใช่สตาร์ทซ้อน"""
    import threading

    from strata.server import _SERVER_LOCK, stop_strata_server

    stop_strata_server()  # ไม่มี server รันอยู่ -> ต้องไม่ raise

    holding = threading.Event()
    release = threading.Event()

    def holder():
        _SERVER_LOCK.acquire()
        holding.set()
        release.wait(5)
        _SERVER_LOCK.release()

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    holding.wait(5)
    try:
        try:
            stop_strata_server()
        except RuntimeError as e:
            assert "กำลัง" in str(e), f"ข้อความควรบอกว่ากำลังทำงานอยู่: {e}"
        else:
            raise AssertionError("ต้องปฏิเสธเมื่อมีคำสั่งอื่นถือ lock อยู่ (กันสตาร์ทซ้อน)")
    finally:
        release.set()
        t.join(5)


def test_network_monitor() -> None:
    """ตัววัด DL/UL: ต้องคำนวณจาก delta ได้, ทนตัวนับรีเซ็ต, และจัดรูปแบบความเร็วถูก"""
    from strata.utils import NetworkMonitor, _fmt_rate, net_bandwidth

    counters = {"v": (1_000, 500)}
    clock = {"t": 100.0}
    mon = NetworkMonitor(reader=lambda: counters["v"], clock=lambda: clock["t"])

    first = mon.sample()
    assert first["down_bps"] == 0.0 and first["up_bps"] == 0.0, "ตัวอย่างแรกยังไม่มี delta"
    assert first["ok"] is False

    clock["t"] += 2.0
    counters["v"] = (1_000 + 2 * 1024**2, 500 + 512 * 1024)   # 2 MB ใน 2 วิ, 512 KB ใน 2 วิ
    second = mon.sample()
    assert second["ok"] is True
    assert abs(second["down_bps"] - 1024**2) < 1, f"DL ผิด: {second['down_bps']}"
    assert abs(second["up_bps"] - 256 * 1024) < 1, f"UL ผิด: {second['up_bps']}"
    assert second["rx_bytes"] == counters["v"][0] and second["tx_bytes"] == counters["v"][1]

    clock["t"] += 1.0
    counters["v"] = (10, 10)          # ตัวนับถูกรีเซ็ต (เช่น คอนเทนเนอร์รีสตาร์ท)
    third = mon.sample()
    assert third["down_bps"] == 0.0 and third["up_bps"] == 0.0, "ตัวนับรีเซ็ตต้องไม่ทำให้เกิดค่าติดลบ/เลขมหาศาล"

    live = net_bandwidth()
    assert {"down_bps", "up_bps", "rx_bytes", "tx_bytes", "ok"} <= set(live), f"คีย์ไม่ครบ: {live}"
    assert live["down_bps"] >= 0 and live["up_bps"] >= 0

    assert _fmt_rate(0) == "0 B/s"
    assert _fmt_rate(900 * 1024) == "900 KB/s"
    assert _fmt_rate(4.21 * 1024**2) == "4.21 MB/s"
    assert _fmt_rate(-5) == "0 B/s", "ค่าติดลบต้องถูก clamp"


def test_notebook_repo_bootstrap() -> None:
    """เซลล์ setup ต้องหา/โคลน/อัปเดตโค้ดเองได้ และถอยไปใช้ branch ของ PR เมื่อ branch หลักยังไม่มีแพ็กเกจ"""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = {f.name for f in funcs}
    for need in ("_ensure_repo", "_refresh", "_pr_branches", "_ok"):
        assert need in names, f"ไม่พบฟังก์ชัน {need} ในเซลล์ setup (โหลดเซลล์เก่าอยู่?)"
    assert "NOTEBOOK_REV" in src, "เซลล์ setup ต้องมี marker NOTEBOOK_REV ให้เดาเวอร์ชันได้"
    keep = {"REQUIRED", "GIT_ENV", "NOTEBOOK_REV", "AUTO_UPDATE"}
    assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in keep for t in n.targets)
    ]
    assert len(assigns) == len(keep), f"ไม่พบค่าคงที่ครบในเซลล์ setup: {[a.targets[0].id for a in assigns]}"

    if Path("/content").exists():
        raise Skip("มี /content อยู่ในเครื่องนี้ (จำลอง Colab ไม่ได้)")

    old_cwd = Path.cwd()
    tmp = Path(tempfile.mkdtemp())
    clones: list[list[str]] = []
    pr_calls: list[str] = []
    ok_refs = {"pr/head", "feature/x"}  # จำลองว่า branch หลักยังไม่มีแพ็กเกจ strata/

    def materialize(target) -> None:
        (target / "strata").mkdir(parents=True, exist_ok=True)
        (target / "strata" / "__init__.py").write_text("")
        (target / "pyproject.toml").write_text("")

    class _OK:
        returncode = 0
        stdout = ""

    def fake_run(cmd, check=False, env=None, **kw):  # noqa: ANN001
        assert "clone" in cmd, f"ไม่ควรเรียกคำสั่งอื่นผ่าน subprocess.run: {cmd}"
        clones.append(list(cmd))
        ref = cmd[cmd.index("--branch") + 1] if "--branch" in cmd else ""
        if ref in ok_refs:
            materialize(Path(cmd[-1]))
        return _OK()

    class _SP:  # namespace: run ปลอม แต่ใช้ exception จริง
        run = staticmethod(fake_run)
        CalledProcessError = subprocess.CalledProcessError

    ns = {
        "Path": Path,
        "os": os,
        "shutil": shutil,
        "json": json,
        "subprocess": _SP,
        "urllib": None,
        "__name__": "shim",
    }
    body = sorted([*funcs, *assigns], key=lambda n: n.lineno)  # คงค่า `REQUIRED`/`GIT_ENV` ไว้ใช้ด้วย
    exec(compile(ast.Module(body=body, type_ignores=[]), "<shim>", "exec"), ns)  # noqa: S102
    ns["_pr_branches"] = lambda slug: (pr_calls.append(slug), ["no/strata", "pr/head"])[1]
    ensure_repo, refresh = ns["_ensure_repo"], ns["_refresh"]

    try:
        # 1) มีแพ็กเกจครบใน cwd แล้ว -> ไม่ clone ไม่ถาม GitHub
        work = tmp / "repo"
        materialize(work)
        os.chdir(work)
        try:
            assert ensure_repo("org/repo", "").resolve() == work.resolve()
        finally:
            os.chdir(old_cwd)
        assert not clones and not pr_calls, f"ไม่ควร clone/ถาม GitHub เมื่อมีโค้ดครบ: {clones}"

        # 2) มี clone เดิม (มี .git) ที่โค้ดเก่า -> อัปเดตก่อนโดยไม่ clone ใหม่
        stale = tmp / "stale"
        (stale / "Colab" / ".git").mkdir(parents=True)
        refreshed: list = []
        ns["_refresh"] = lambda target, refs: (refreshed.append(list(refs)), materialize(target), True)[-1]
        os.chdir(stale)
        try:
            got = ensure_repo("org/repo", "")
            assert got.resolve() == (stale / "Colab").resolve(), f"ปลายทางผิด: {got}"
        finally:
            os.chdir(old_cwd)
            ns["_refresh"] = refresh
        assert refreshed == [[None, "no/strata", "pr/head"]], f"ลำดับ ref สำหรับ refresh ผิด: {refreshed}"
        assert not clones, f"มี clone เดิมแล้วไม่ควร clone ซ้ำ: {clones}"

        # 2b) ปิด AUTO_UPDATE -> ใช้ของเดิมถ้าครบ โดยไม่ fetch
        refreshed.clear()
        os.chdir(stale)
        try:
            materialize(stale / "Colab")
            assert ensure_repo("org/repo", "", update=False).resolve() == (stale / "Colab").resolve()
        finally:
            os.chdir(old_cwd)
        assert not refreshed, f"ปิด AUTO_UPDATE แล้วไม่ควร refresh: {refreshed}"

        # 3) ไม่มี clone เลย -> ไล่ branch หลักก่อน แล้วถอยไป PR
        bare = tmp / "bare"
        bare.mkdir()
        os.chdir(bare)
        try:
            got = ensure_repo("org/repo", "")
        finally:
            os.chdir(old_cwd)
        assert got.resolve() == (bare / "Colab").resolve(), f"ปลายทางผิด: {got}"
        assert len(clones) == 3, f"ควรไล่ default -> PR#1 -> PR#2 ได้ {clones}"
        assert "--branch" not in clones[0], f"รอบแรกต้องเป็น branch หลัก: {clones[0]}"
        tried = [c[c.index("--branch") + 1] for c in clones[1:]]
        assert tried == ["no/strata", "pr/head"], f"ลำดับ branch ที่ลองผิด: {tried}"
        for cmd in clones:
            assert cmd[0] == "git" and cmd[1] == "clone" and "--depth" in cmd, f"คำสั่ง clone ผิด: {cmd}"
            assert any(str(c).startswith("https://github.com/org/repo") for c in cmd), f"URL ผิด: {cmd}"

        # 4) ระบุ branch เอง -> ไม่ถาม GitHub
        picks = tmp / "picks"
        picks.mkdir()
        clones.clear()
        pr_calls.clear()
        os.chdir(picks)
        try:
            ensure_repo("org/repo", "feature/x")
        finally:
            os.chdir(old_cwd)
        assert len(clones) == 1 and clones[0][clones[0].index("--branch") + 1] == "feature/x", f"{clones}"
        assert not pr_calls, "ระบุ branch เองแล้วไม่ควรถาม GitHub"

        # 5) _refresh: ref แรกพัง -> ลองตัวถัดไป, reset แล้วได้ไฟล์ครบ -> True
        fetches: list = []

        def fake_git(args, cwd=None):
            if args[0] == "fetch":
                ref = args[4] if len(args) > 4 else None
                fetches.append(ref)
                if ref not in ok_refs:
                    raise subprocess.CalledProcessError(1, args)
                return ""
            if args[0] == "reset":
                materialize(Path(cwd))
                return ""
            raise AssertionError(f"คำสั่ง git ที่ไม่คาดคิด: {args}")

        ns["_git"] = fake_git
        target = tmp / "heal"
        target.mkdir()
        assert refresh(target, [None, "pr/head"]) is True
        assert fetches == [None, "pr/head"], f"ควรลอง ref แรกแล้วตกไปตัวถัดไป: {fetches}"

        # 6) _refresh: ทุก ref พัง -> False (ให้ caller ไป clone ใหม่)
        fetches.clear()
        ns["_git"] = lambda args, cwd=None: (
            fetches.append(args[4] if len(args) > 4 else None),
            (_ for _ in ()).throw(subprocess.CalledProcessError(1, args)),
        )[1]
        assert refresh(tmp / "nowhere", [None, "gone/branch"]) is False
        assert fetches == [None, "gone/branch"], f"ควรลองทุก ref: {fetches}"
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_package_files_tracked() -> None:
    """ไฟล์ทุกตัวในแพ็กเกจต้องถูก git ติดตาม — กันเคส .gitignore กว้างเกินจน clone มาแล้วไฟล์หาย"""
    if not (ROOT / ".git").exists():
        raise Skip("ไม่มี .git ในเครื่องนี้")
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "strata"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    tracked = {t for t in tracked if t}
    on_disk = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "strata").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    missing = sorted(on_disk - tracked)
    assert not missing, f"ไฟล์ในแพ็กเกจที่ยังไม่ถูก commit (clone มาจะหาย): {missing}"

    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-z", "--stdin"],
        input="\0".join(sorted(on_disk)), capture_output=True, text=True,
    )
    for path in (p for p in ignored.stdout.split("\0") if p):
        assert path.endswith(".pyc"), f"ไฟล์แพ็กเกจถูก .gitignore บัง: {path}"


def test_flake_template() -> None:
    """เทมเพลตต้องอยู่ในโค้ด (ไม่พึ่งไฟล์ภายนอกที่หลุด .gitignore ได้) และแทน placeholder ครบ"""
    from strata import bootstrap as bs
    from strata.flake_template import FLAKE_TEMPLATE

    assert isinstance(bs.FLAKE_TEMPLATE, str), "FLAKE_TEMPLATE ต้องเป็นสตริงในโค้ด ไม่ใช่ path"
    assert not (PKG / "templates").exists(), "ไม่ควรมีโฟลเดอร์ templates/ ให้ไฟล์หลุด track อีก"
    for tok in ("@NIXPKGS_REV@", "@PY_VER@", "@CUSTOM_HF_REPO@", "@MODEL_DIR@"):
        assert tok in FLAKE_TEMPLATE, f"เทมเพลตขาด placeholder {tok}"
    tmp = Path(tempfile.mkdtemp())
    old = Path.cwd()
    try:
        os.chdir(tmp)
        out = bs.write_flake("org/repo")
        text = Path(out).read_text(encoding="utf-8")
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)
    leftovers = [tok for tok in ("@NIXPKGS_REV@", "@PY_VER@", "@CUSTOM_HF_REPO@", "@MODEL_DIR@") if tok in text]
    assert not leftovers, f"ยังมี placeholder ค้างใน flake.nix: {leftovers}"
    assert "$@" in text or "\"$@\"" in text, "สคริปต์ hf download ต้องยังรับ argument ต่อท้ายได้"
    assert bs.NIXPKGS_REV in text, "flake.nix ต้องปักหมุด nixpkgs rev"
    assert "org/repo" in text, "flake.nix ต้องใส่ repo ของ GGUF ที่กำหนด"


def main() -> int:
    print(f"🧪 smoke test: strata/{NOTEBOOK.parent.name} (python {sys.version.split()[0]})")
    check("notebook structure", test_notebook_shape)
    check("notebook cells compile", test_notebook_cells_compile)
    check("package compiles + no bad f-string", test_package_compiles)
    check("public API / catalog", test_public_api)
    check("lazy imports", test_lazy_imports)
    check("no known buggy patterns", test_no_buggy_patterns)
    check("performance patterns", test_perf_patterns)
    check("no dead code", test_no_dead_code)
    check("_pick_main_gguf", test_pick_main_gguf)
    check("safe archive extraction", test_safe_extract)
    check("uv_add input validation", test_uv_add_validation)
    check("_dir_stats cache", test_dir_stats_cache)
    check("_gguf_layer_count", test_gguf_layer_count)
    check("StrataRAG chunking", test_rag_chunking)
    check("StrataRAG dedup/search/persist", test_rag_dedup_search_persist)
    check("server lock guard", test_server_lock_guard)
    check("network monitor (DL/UL)", test_network_monitor)
    check("notebook repo bootstrap", test_notebook_repo_bootstrap)
    check("package files tracked by git", test_package_files_tracked)
    check("flake.nix from embedded template", test_flake_template)
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
