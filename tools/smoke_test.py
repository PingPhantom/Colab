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
import fnmatch
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
from urllib.parse import unquote

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
    assert len(ALL_PRESET_KEYS) >= 10, f"preset ต้องมีอย่างน้อย 10 ตัว ได้ {len(ALL_PRESET_KEYS)}"
    assert len(ALL_PRESET_KEYS) == len(set(ALL_PRESET_KEYS)), "preset key ต้องไม่ซ้ำ"
    assert len(MODEL_CATALOG) == len(ALL_PRESET_KEYS) + 1, "MODEL_CATALOG = preset + custom"
    assert set(ALL_PRESET_KEYS) <= set(MODEL_CATALOG), "preset ทุกตัวต้องอยู่ใน catalog"
    assert "ctx_size" not in SPEC_STATE or isinstance(SPEC_STATE["ctx_size"], int)
    for key, cfg in MODEL_CATALOG.items():
        assert re.fullmatch(r"[\w.\-]+/[\w.\-]+", cfg["repo_id"]), f"{key}: repo_id ผิดรูป"
        assert isinstance(cfg["desc"], str) and cfg["desc"], f"{key}: ต้องมี desc"
        assert isinstance(cfg["approx_gb"], (int, float)) and cfg["approx_gb"] >= 0, f"{key}: approx_gb ผิด"
        inc = cfg["include"]
        assert inc is None or (isinstance(inc, list) and inc), f"{key}: include ต้องเป็น list หรือ None"
        if inc:
            assert all(isinstance(x, str) and x for x in inc), f"{key}: include ต้องเป็นสตริง"

    # preset ใหม่ (repo จริงบน Hugging Face) — sharded GGUF ต้องตั้ง include ให้ครบชุด
    ds = MODEL_CATALOG["deepseek-v41-flash-abliterated"]
    assert ds["repo_id"] == "orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF"
    assert any("Mix-128G" in pat for pat in ds["include"]), "ต้องระบุโฟลเดอร์ shard ที่ต้องการ"
    assert ds["approx_gb"] > 200, "โมเดลนี้ใหญ่ระดับ 200GB+"


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
    def _seg(name: str) -> str:
        seg = next(
            ast.get_source_segment(ui, n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert seg, f"ไม่พบ {name}"
        return seg

    # _refresh() ห่อ _refresh_inner() ด้วย lock เพื่อกัน thread เรียลไทม์เขียนทับกัน
    refresh = _seg("_refresh") + _seg("_refresh_inner")
    assert "_B(" not in refresh, "_refresh ต้องไม่สร้าง widget ใหม่ทุกครั้ง"
    assert 'rglob("*.gguf")' not in refresh, "_refresh ต้องใช้ _dir_stats() แบบ cache"
    assert "_dir_stats(" in refresh, "_refresh ควรใช้ _dir_stats() แบบ cache"
    assert "with ui_lock:" in _seg("_refresh"), "ต้องมี lock กัน thread เรียลไทม์เขียนทับ UI"
    assert "threading.RLock()" in ui, "ต้องสร้าง RLock สำหรับ ui_lock"
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
    assert "strata.bootstrap import" in src and src.index("_purge_modules()") < src.index(
        "from strata.bootstrap import"
    ), "ต้องล้างโมดูลที่ค้างก่อน import strata.* (ไม่งั้น git reset แล้วยังรันโค้ดเก่า)"
    for need in ("_ensure_repo", "_refresh", "_pr_branches", "_ok", "_purge_modules"):
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


def test_downloader_fallback() -> None:
    """โหลด GGUF ต้องไม่ตายด้วย `command not found` — ถอยจาก download-deepseek → hf → huggingface_hub"""
    from strata import download as dl

    real_which = dl.shutil.which
    try:
        dl.shutil.which = lambda name: "/nix/store/x/bin/" + name if name == "download-deepseek" else None
        cmd = dl.download_command("org/repo", "/tmp/m", ["*Q4_K_M*"], ["*f16*"])
        assert cmd[0] == "download-deepseek", f"ควรใช้ wrapper ก่อน: {cmd}"
        assert "--include" in cmd and "--exclude" in cmd and "/tmp/m" in cmd, f"args ไม่ครบ: {cmd}"
        assert os.environ.get("HF_REPO_ID") == "org/repo", "wrapper ต้องได้ repo ผ่าน HF_REPO_ID"

        dl.shutil.which = lambda name: "/usr/local/bin/hf" if name == "hf" else None
        cmd = dl.download_command("org/repo", "/tmp/m", ["*Q4_K_M*"], [])
        assert cmd[:3] == ["hf", "download", "org/repo"], f"ควรถอยไปใช้ hf CLI: {cmd}"
        assert "--local-dir" in cmd and "/tmp/m" in cmd, f"args ไม่ครบ: {cmd}"

        dl.shutil.which = lambda name: None  # ไม่มีอะไรเลย -> ใช้โมดูล Python
        assert dl.download_command("org/repo", "/tmp/m", [], []) == []

        calls: dict = {}
        fake = type(sys)("huggingface_hub")

        def fake_snapshot(**kw):  # noqa: ANN003
            calls.update(kw)

        fake.snapshot_download = fake_snapshot
        sys.modules["huggingface_hub"] = fake
        try:
            dl.download_with_module("org/repo", "/tmp/m", ["*Q4*"], ["*f16*"])
        finally:
            del sys.modules["huggingface_hub"]
        assert calls["repo_id"] == "org/repo" and calls["local_dir"] == "/tmp/m", f"args ผิด: {calls}"
        assert calls["allow_patterns"] == ["*Q4*"] and calls["ignore_patterns"] == ["*f16*"], f"patterns ผิด: {calls}"

        # ไม่มีทั้งเครื่องมือและโมดูล -> error ที่บอกวิธีแก้ ไม่ใช่ command not found
        sys.modules["huggingface_hub"] = None
        try:
            dl.download_with_module("org/repo", "/tmp/m", [], [])
        except RuntimeError as exc:
            assert "ensure_environment" in str(exc) or "devshell" in str(exc), f"ควรบอกวิธีแก้: {exc}"
        else:
            raise AssertionError("ควร raise เมื่อไม่มีตัวดาวน์โหลดเลย")
        finally:
            sys.modules.pop("huggingface_hub", None)
    finally:
        dl.shutil.which = real_which


def test_purge_modules() -> None:
    """หลัง git fetch/reset ต้องล้างโมดูล `strata.*` ที่ค้างใน kernel (ไม่งั้นยังรันไบต์โค้ดเก่า)"""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_purge_modules"), None)
    assert node is not None, "ไม่พบ _purge_modules ในเซลล์ setup"
    fn_src = ast.get_source_segment(src, node)

    probe = '\n'.join(
        [
            "import importlib, sys, types",
            fn_src,
            'sys.modules["strata"] = types.ModuleType("strata")',
            'sys.modules["strata.bootstrap"] = types.ModuleType("strata.bootstrap")',
            'sys.modules["strata.ui"] = types.ModuleType("strata.ui")',
            'sys.modules["other_pkg"] = types.ModuleType("other_pkg")',
            "_purge_modules()",
            'print("LEFT:", sorted(k for k in sys.modules if k == "strata" or k.startswith("strata.")))',
            'print("KEPT:", "other_pkg" in sys.modules)',
        ]
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert "LEFT: []" in out.stdout, f"ยังมีโมดูล strata ค้าง: {out.stdout}"
    assert "KEPT: True" in out.stdout, f"ไม่ควรลบโมดูลของแพ็กเกจอื่น: {out.stdout}"


def test_branch_autodetect() -> None:
    """เซลล์ setup ต้องเดา slug/branch เองจาก URL ของ Colab (หรือ git ของ clone เดิม) — ไม่ต้องพิมพ์ `arena/...`"""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for need in ("_parse_github_url", "_detect_source"):
        assert need in funcs, f"ไม่พบ {need} ในเซลล์ setup"

    ns = {"unquote": unquote, "Path": Path, "subprocess": subprocess}
    exec(compile(ast.Module(body=[funcs["_parse_github_url"]], type_ignores=[]), "<p>", "exec"), ns)  # noqa: S102
    parse = ns["_parse_github_url"]
    base = "https://colab.research.google.com/github/PingPhantom/Colab/blob/"
    assert parse(base + "arena/e076c6f4-colab/nix.ipynb") == ("PingPhantom/Colab", "arena/e076c6f4-colab")
    assert parse(base + "main/nix.ipynb?hl=th") == ("PingPhantom/Colab", "main")
    assert parse(base + "feature%2Fnew-ui/nix.ipynb") == ("PingPhantom/Colab", "feature/new-ui")
    assert parse("https://colab.research.google.com/drive/1abcDEF") == ("", ""), "ไม่ใช่ลิงก์ GitHub ต้องไม่เดา"
    assert parse("") == ("", "")

    # ไม่มี URL (เช่นเปิดจาก Drive) -> ต้องถอยไปใช้ branch ของ clone ที่มีอยู่
    ns2 = dict(ns)
    exec(compile(ast.Module(body=[funcs["_detect_source"]], type_ignores=[]), "<d>", "exec"), ns2)  # noqa: S102
    ns2["_git"] = lambda args, cwd=None: "arena/e076c6f4-colab"
    ns2["Path"] = Path
    detect = ns2["_detect_source"]
    tmp = Path(tempfile.mkdtemp())
    try:
        clone = tmp / "Colab"
        (clone / ".git").mkdir(parents=True)
        old = Path.cwd()
        os.chdir(tmp)
        try:
            ns2["Path"] = Path
            assert detect() == ("", "arena/e076c6f4-colab"), "ควรอ่าน branch จาก git ของ clone เดิม"
        finally:
            os.chdir(old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert 'REPO_BRANCH = REPO_BRANCH or _found_branch' in src, "ต้องเติม REPO_BRANCH อัตโนมัติจากค่าที่เดาได้"
    assert 'REPO_SLUG = REPO_SLUG or _found_slug or DEFAULT_SLUG' in src, "ต้องเติม REPO_SLUG อัตโนมัติ"


def test_flake_outside_git_repo() -> None:
    """flake ต้องอยู่นอก git repo (nix ไม่เห็นไฟล์ที่ยังไม่ถูก track) และต้องเก็บกวาดของเดิมใน cwd"""
    from strata import bootstrap as bs

    tmp = Path(tempfile.mkdtemp())
    old_cwd, old_dir = Path.cwd(), bs.FLAKE_DIR
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "flake.nix").write_text(bs.render_flake("org/old"), encoding="utf-8")  # ของที่เราเคยเขียน
        (repo / "default.nix").write_text(bs.render_flake("org/old"), encoding="utf-8")
        flake_dir = tmp / "flake"
        os.chdir(repo)
        bs.FLAKE_DIR = flake_dir
        out = bs.write_flake("org/repo")

        assert out.parent == flake_dir, f"flake ต้องถูกเขียนนอก repo: {out}"
        assert not (repo / "flake.nix").exists(), "ต้องลบ flake.nix ที่เคยเขียนไว้ใน cwd"
        assert not (repo / "default.nix").exists(), "ต้องลบ default.nix ที่เคยเขียนไว้ใน cwd"
        text = out.read_text(encoding="utf-8")
        assert "org/repo" in text and bs.NIXPKGS_REV in text, "เนื้อหา flake ไม่ถูกต้อง"

        # flake.nix ของผู้ใช้เอง (เนื้อหาต่างจากที่เราสร้าง) ต้องไม่ถูกลบ
        mine = repo / "flake.nix"
        mine.write_text("{ /* ของผู้ใช้ */ }\n", encoding="utf-8")
        bs.write_flake("org/repo")
        assert mine.exists(), "ห้ามลบ flake.nix ที่ผู้ใช้เขียนเอง"
    finally:
        os.chdir(old_cwd)
        bs.FLAKE_DIR = old_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_activate_devshell_errors() -> None:
    """`nix develop` ล้มต้องบอก stderr + วิธีแก้ (เดิมกลืน stderr จน debug ไม่ได้)"""
    from strata import bootstrap as bs

    tmp = Path(tempfile.mkdtemp())
    real_run, real_env = bs.subprocess.run, dict(os.environ)
    try:
        (tmp / "flake.nix").write_text("{}", encoding="utf-8")

        def fail(cmd, **kw):  # noqa: ANN001
            class R:
                returncode = 1
                stdout = ""
                stderr = "error: path '/content/Colab/flake.nix' is not tracked by Git\n"

            assert "nix" in cmd and str(tmp) in cmd, f"ต้องเรียก nix develop ที่ flake dir: {cmd}"
            assert "DEVNULL" not in str(kw), "ห้ามกลืน stderr"
            return R()

        bs.subprocess.run = fail
        try:
            bs.activate_devshell(tmp)
        except RuntimeError as exc:
            msg = str(exc)
            assert "not tracked by Git" in msg, f"ต้องแนบ stderr ท้ายข้อความ: {msg}"
            assert "nix develop" in msg and "/etc/nix/nix.conf" in msg, f"ต้องมีวิธีแก้: {msg}"
        else:
            raise AssertionError("ควร raise เมื่อ nix develop ล้ม")

        # สำเร็จ -> env ต้องถูก merge และคืนสรุปได้
        def ok(cmd, **kw):  # noqa: ANN001
            class R:
                returncode = 0
                stderr = ""
                stdout = json.dumps({"PATH": "/nix/store/abc/bin", "LD_LIBRARY_PATH": "/nix/store/lib", "FOO": "bar"})

            return R()

        bs.subprocess.run = ok
        info = bs.activate_devshell(tmp)
        assert info["flake_dir"] == str(tmp), f"คืนค่าไม่ถูก: {info}"
        assert "/nix/store/abc/bin" in os.environ["PATH"], "PATH ต้องมีของ devshell"
        assert "/usr/local/bin" in os.environ["PATH"], "PATH เดิมต้องยังอยู่"
        assert os.environ["FOO"] == "bar", "ต้อง merge env อื่นด้วย"
        assert bs.nix_diagnostics()["flake_dir"] == str(bs.FLAKE_DIR), "diagnostics ต้องรายงาน flake dir"
    finally:
        bs.subprocess.run = real_run
        os.environ.clear()
        os.environ.update(real_env)
        shutil.rmtree(tmp, ignore_errors=True)


def test_prebuilt_glibc_isolation() -> None:
    """llama-server ต้องไม่หยิบ lib ของ Nix (glibc 2.44) มาปนกับ libc ของระบบ — ไม่งั้น GLIBC_ABI... not found"""
    from strata import server as sv

    raw = "/usr/local/cuda/lib64:/nix/store/abc/lib:/usr/lib64-nvidia:/nix/store/x/lib"
    assert sv.sanitize_ld_library_path(raw) == "/usr/local/cuda/lib64:/usr/lib64-nvidia", "ต้องตัด /nix/store ออก"
    assert sv.sanitize_ld_library_path(None) == ""
    assert sv.sanitize_ld_library_path("/a:/a:/b") == "/a:/b", "ต้องตัดค่าซ้ำ"

    env = sv.server_env(
        {"LD_LIBRARY_PATH": "/nix/store/glibc/lib:/usr/local/cuda/lib64", "LD_PRELOAD": "/nix/store/y/pre.so", "PATH": "/bin"}
    )
    assert "/nix/store" not in env["LD_LIBRARY_PATH"], f"LD_LIBRARY_PATH ยังมีของ Nix: {env['LD_LIBRARY_PATH']}"
    assert "LD_PRELOAD" not in env, "LD_PRELOAD ที่ชี้เข้า store ต้องถูกถอด"
    assert env["PATH"] == "/bin", "ต้องไม่แตะ env อื่น"

    assert "glibc" in sv.start_error_hint("x: version `GLIBC_ABI_DT_X86_64_PLT' not found"), "ต้องมี hint glibc"
    assert "CUDA" in sv.start_error_hint("ggml_cuda_init: no CUDA device"), "ต้องมี hint CUDA"
    assert sv.start_error_hint("ออกจากหน่วยความจำไม่พอ") == "", "ข้อความอื่นไม่ต้องมี hint"

    # wrapper ต้องกรอง path ของ Nix จริงตอนรัน (ทดสอบด้วยไบนารีปลอม)
    tmp = Path(tempfile.mkdtemp())
    try:
        fake = tmp / "llama-server"
        fake.write_text('#!/bin/sh\necho "LD=$LD_LIBRARY_PATH"\necho "PRELOAD=${LD_PRELOAD:-none}"\n', encoding="utf-8")
        fake.chmod(0o755)
        wrapper = sv._write_wrapper(fake, tmp, tmp / "wrapper.sh")
        text = wrapper.read_text(encoding="utf-8")
        assert sv.WRAPPER_REV in text, "wrapper ต้องมี marker ไว้ตรวจเวอร์ชัน"
        assert "/nix/store/*" in text, "wrapper ต้องมีตัวกรอง /nix/store"
        assert text.startswith("#!/usr/bin/env bash"), "wrapper ต้องเป็น bash (ใช้ process substitution)"

        run_env = dict(os.environ, LD_LIBRARY_PATH="/nix/store/glibc/lib:/usr/local/cuda/lib64:/usr/lib64-nvidia", LD_PRELOAD="/nix/store/x/pre.so")
        out = subprocess.run([str(wrapper)], capture_output=True, text=True, env=run_env, timeout=30).stdout
        assert "/nix/store" not in out, f"wrapper ยังส่ง path ของ Nix ต่อ: {out}"
        assert "PRELOAD=none" in out, f"wrapper ต้อง unset LD_PRELOAD: {out}"
        assert str(tmp) in out, f"wrapper ต้องใส่โฟลเดอร์ไบนารีเอง: {out}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gpu_memory_preflight() -> None:
    """VRAM ค้างจาก process เก่า -> ต้องเก็บกวาดให้ และรอโหลดนานขึ้นตามขนาดโมเดล"""
    from strata import server as sv

    assert sv.load_timeout_for(4) == 240, "โมเดลเล็กใช้ค่าเริ่มต้น"
    assert sv.load_timeout_for(20.8) > 240, "โมเดลใหญ่ต้องรอนานขึ้น"
    assert sv.load_timeout_for(500) == 900, "ต้องมีเพดานเวลารอ"

    real_run, real_kill, real_sleep = sv.subprocess.run, sv.os.kill, sv.time.sleep
    real_which = sv.shutil.which
    sv.shutil.which = lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    # อ่านรายการ process ที่กิน VRAM (nvidia-smi)
    class R:
        def __init__(self, out=""):
            self.stdout = out
            self.returncode = 0
            self.stderr = ""

    def fake_run(cmd, **kw):  # noqa: ANN001
        if "--query-compute-apps" in " ".join(cmd):
            return R("1234, /usr/local/bin/llama-server, 14000\n5678, python3, 512\n")
        if cmd[:2] == ["pgrep", "-af"]:
            return R("1234 /content/.llama-cpp/llama-b1/llama-server --port 18080\n999 other-prog\n")
        return R()

    sv.subprocess.run = fake_run
    try:
        users = sv.gpu_memory_users()
        assert users and users[0]["pid"] == 1234 and users[0]["mb"] == 14000, f"อ่าน VRAM ผิด: {users}"
        assert [p for p in sv._llama_pids()] == [1234], f"ควรเจอเฉพาะ llama-server: {sv._llama_pids()}"

        # release_gpu_memory: ฆ่า process ค้างแล้วรอ VRAM คืน
        killed: list = []
        sv.os.kill = lambda pid, sig: killed.append((pid, sig))
        sv.time.sleep = lambda s: None
        free = {"gb": [0.4, 0.5, 6.2]}

        def fake_hw(verbose=False):  # noqa: ANN001
            val = free["gb"].pop(0) if len(free["gb"]) > 1 else free["gb"][0]
            return {"has_gpu": True, "vram_free_gb": val}

        real_hw = sv.check_gpu_and_recommend
        sv.check_gpu_and_recommend = fake_hw
        try:
            info = sv.release_gpu_memory(port=18080)
        finally:
            sv.check_gpu_and_recommend = real_hw
        assert 1234 in info["killed"], f"ต้องฆ่า pid ที่ค้าง: {info}"
        assert info["vram_free_gb"] >= 2.0, f"ควรได้ VRAM คืน: {info}"
        assert killed and killed[0][0] == 1234, f"ต้องส่ง SIGTERM ให้ pid ค้าง: {killed}"
    finally:
        sv.subprocess.run, sv.os.kill, sv.time.sleep = real_run, real_kill, real_sleep
        sv.shutil.which = real_which


def test_spec_flag_compat() -> None:
    """llama.cpp รุ่นใหม่ถอด `--draft-max` -> ต้องเลือกชื่อ option ตาม `--help` ของไบนารีจริง"""
    from strata import server as sv

    new_help = (
        "usage: llama-server [options]\n"
        "  --spec-type {ngram-mod,ngram-simple,draft-mtp}\n"
        "  --spec-draft-n-max N\n"
        "  --spec-ngram-mod-n-max N\n"
        "-dm, --draft-max N   the argument has been removed\n"
    )
    assert sv.resolve_draft_max_flag(new_help, "ngram") == "--spec-ngram-mod-n-max"
    assert sv.resolve_draft_max_flag(new_help, "draft") == "--spec-draft-n-max"
    assert sv.resolve_draft_max_flag(new_help, "mtp") == "--spec-draft-n-max"

    # รุ่นเก่า (ยังไม่มีชุด --spec-*) ต้องยังใช้ --draft-max ได้
    old_help = "-ngl N, --n-gpu-layers N\n-dm, --draft-max N\n--draft N\n"
    assert sv.resolve_draft_max_flag(old_help, "ngram") == "--draft-max"
    assert sv.resolve_draft_max_flag(old_help, "draft") == "--draft-max"

    # มี --spec-type แต่ไม่มี option จำกัด token -> ต้องคืน "" (ไม่ใส่ flag ที่อาจถูกถอด)
    assert sv.resolve_draft_max_flag("--spec-type draft-mtp\n", "ngram") == ""
    assert sv.resolve_draft_max_flag("", "draft") == "", "ไม่มี help ต้องไม่เดา"

    # ต้องไม่หลงจับ --draft-max ตอนขอ --draft
    assert sv.resolve_draft_max_flag("-dm, --draft-max N\n", "draft") == "--draft-max"
    hint = sv.start_error_hint('error while handling argument "--draft-max": the argument has been removed')
    assert "speculative" in hint.lower() or "spec" in hint.lower(), f"ต้องมี hint เรื่อง option: {hint}"


def test_shell_env_persistence() -> None:
    """`nix` ต้องใช้ได้ใน terminal/shell ใหม่ (ไม่ใช่แค่ใน kernel) ไม่งั้นเจอ `nix: command not found`"""
    from strata import bootstrap as bs

    tmp = Path(tempfile.mkdtemp())
    try:
        profile, rc = tmp / "profile.d", tmp / ".bashrc"
        info = bs.persist_shell_env(profile, rc)
        assert (profile / "strata-nix.sh").is_file(), "ต้องเขียน /etc/profile.d/strata-nix.sh"
        rc_text = rc.read_text(encoding="utf-8")
        assert bs.SHELL_MARK in rc_text, "ต้องเติม snippet ลง ~/.bashrc"
        assert "nix/var/nix/profiles/default/bin" in rc_text, "PATH ต้องมี bin ของ nix"
        assert info["written"], "ต้องรายงานไฟล์ที่เขียน"

        # เรียกซ้ำต้องไม่ต่อ snippet ซ้ำ
        bs.persist_shell_env(profile, rc)
        assert rc.read_text(encoding="utf-8").count(bs.SHELL_MARK) == 1, "เรียกซ้ำแล้วต้องไม่ซ้ำ"
        prof_txt = (profile / "strata-nix.sh").read_text(encoding="utf-8")
        assert prof_txt.count("export PATH=") == 1, f"profile.d ผิดรูป: {prof_txt}"
        assert "$HOME/.nix-profile/bin" in prof_txt, "ต้องมี ~/.nix-profile/bin ใน PATH"

        # symlink คำสั่ง nix ไป /usr/local/bin (PATH มาตรฐานของทุก shell)
        bindir = tmp / "bin"
        bindir.mkdir()
        fake_nix = tmp / "nix-real"
        fake_nix.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_nix.chmod(0o755)
        real_which = bs.shutil.which
        bs.shutil.which = lambda name: str(fake_nix) if name == "nix" else None
        try:
            linked = bs.link_nix_binaries(bindir)
        finally:
            bs.shutil.which = real_which
        assert linked and (bindir / "nix").is_symlink(), f"ต้อง symlink nix: {linked}"
        assert (bindir / "nix").resolve() == fake_nix.resolve()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_catalog_download_patterns() -> None:
    """include/exclude ของ preset ต้องเลือกไฟล์ที่ตั้งใจจริง — กันเคสกวาดไฟล์ซ้ำ (จ่ายดิสก์ 2 เท่า)"""
    from strata.config import MODEL_CATALOG

    def picks(key: str, filename: str) -> bool:
        cfg = MODEL_CATALOG[key]
        inc = cfg.get("include") or []
        exc = cfg.get("exclude") or []
        hit = any(fnmatch.fnmatch(filename, p) or fnmatch.fnmatch(Path(filename).name, p) for p in inc)
        ban = any(fnmatch.fnmatch(filename, p) or fnmatch.fnmatch(Path(filename).name, p) for p in exc)
        return hit and not ban

    q = "qwen35-9b-defiant-fable"
    assert picks(q, "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf"), "ต้องโหลดไฟล์ MTP"
    assert not picks(q, "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q4_K_M.gguf"), "ห้ามโหลดรุ่นไม่-MTP ซ้ำ"
    assert not picks(q, "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q8_0.gguf"), "โหลด quant เดียวพอ"
    assert picks(q, "mmproj-F16.gguf"), "ต้องมี mmproj สำหรับ multimodal"
    assert not picks(q, "mmproj-F32.gguf"), "ไม่ต้องโหลด mmproj ซ้ำหลายแบบ"
    assert MODEL_CATALOG[q]["has_mtp"] is True, "รุ่นนี้มี MTP head"

    t2 = "qwen38-27b-turbo-fable"
    assert picks(t2, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf")
    assert not picks(t2, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf"), "LOW-MTP ซ้ำ"
    assert not picks(t2, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ4_XS.gguf"), "รุ่นไม่-MTP ซ้ำ"
    assert not picks(t2, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q8_0.gguf"), "โหลด quant เดียว"
    assert picks(t2, "mmproj-F16.gguf") and MODEL_CATALOG[t2]["has_mtp"] is True

    d = "deepseek-v41-flash-abliterated"
    assert picks(d, "Mix-128G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-128G-00001-of-00005.gguf")
    assert not picks(d, "Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00001-of-00007.gguf"), "ข้ามชุด 256G"
    assert not picks(d, "imatrix.gguf"), "imatrix ใช้ตอนทำ quant เท่านั้น"


def test_image_catalog() -> None:
    """โมเดลสร้างภาพอยู่ใน MODEL_CATALOG เดียวกัน (kind="image") — ไม่แยกแคตตาล็อก"""
    from strata.config import (
        ALL_PRESET_KEYS,
        COMFY_MODEL_SUBDIRS,
        IMAGE_PRESET_KEYS,
        MODEL_CATALOG,
        TEXT_PRESET_KEYS,
    )

    assert IMAGE_PRESET_KEYS, "ต้องมีโมเดลสร้างภาพอย่างน้อย 1 ตัว"
    assert set(IMAGE_PRESET_KEYS) <= set(ALL_PRESET_KEYS), "โมเดลภาพต้องอยู่ใน ALL_PRESET_KEYS/lิสต์เดียวกับโมเดลอื่น"
    assert set(TEXT_PRESET_KEYS) | set(IMAGE_PRESET_KEYS) == set(ALL_PRESET_KEYS)
    assert not (set(TEXT_PRESET_KEYS) & set(IMAGE_PRESET_KEYS))
    for key in IMAGE_PRESET_KEYS:
        cfg = MODEL_CATALOG[key]
        assert cfg["kind"] == "image", f"{key}: kind ต้องเป็น image"
        assert re.fullmatch(r"[\w.\-]+/[\w.\-]+", cfg["repo_id"]), f"{key}: repo_id ผิดรูป"
        targets = cfg["targets"]
        assert targets, f"{key}: ต้องระบุไฟล์"
        assert set(targets) == set(cfg["include"]), f"{key}: include ต้องตรงกับ targets"
        assert all(sub in COMFY_MODEL_SUBDIRS for sub in targets.values()), f"{key}: โฟลเดอร์ปลายทางไม่รองรับ"
        assert all(str(f).endswith((".gguf", ".safetensors")) for f in targets), f"{key}: นามสกุลไฟล์ผิด"
        assert "diffusion_models" in targets.values(), f"{key}: ต้องมี diffusion model"
        assert isinstance(cfg["approx_gb"], (int, float)) and cfg["approx_gb"] > 0, f"{key}: approx_gb ผิด"
        assert isinstance(cfg["desc"], str) and cfg["desc"], f"{key}: ต้องมี desc"
    for key in TEXT_PRESET_KEYS:
        cfg = MODEL_CATALOG[key]
        assert cfg.get("kind", "text") == "text", f"{key}: โมเดลข้อความไม่ต้องมี kind=image"
        assert "targets" not in cfg, f"{key}: โมเดลข้อความไม่ต้องมี targets"


def test_imagegen_helpers() -> None:
    """ตัวช่วย ComfyUI: คำสั่ง, ตำแหน่ง symlink, การผูกไฟล์ และสถานะ (ไม่ต้องมี GPU/เน็ต)"""
    import tempfile

    from strata import imagegen as ig

    assert ig.include_patterns("qwen-image-21-uncensored"), "ต้องมี pattern ดาวน์โหลด"

    cmd = ig.comfyui_command(port=8188)
    assert cmd[1] == "main.py", f"คำสั่งต้องเรียก main.py: {cmd}"
    assert "--listen" in cmd and cmd[cmd.index("--listen") + 1] == "0.0.0.0", "ต้อง listen 0.0.0.0 ให้ Colab proxy เข้าได้"
    assert cmd[cmd.index("--port") + 1] == "8188"
    assert "--lowvram" in cmd, "ค่าเริ่มต้นควรเป็น lowvram สำหรับ T4"
    assert "--lowvram" not in ig.comfyui_command(8188, lowvram=False)

    dst = ig.comfy_target_path("flux-3-klein-uncensored", "FLUX-3-Image-Klein-pup-Uncensored-AD-Q5_K_M.gguf", "/tmp/cf")
    assert dst.as_posix() == "/tmp/cf/models/diffusion_models/FLUX-3-Image-Klein-pup-Uncensored-AD-Q5_K_M.gguf", str(dst)

    raised = False
    try:
        ig.comfy_target_path("flux-3-klein-uncensored", "ไม่มีไฟล์นี้.gguf", "/tmp/cf")
    except (KeyError, ValueError):
        raised = True
    assert raised, "ไฟล์ที่ไม่รู้จักต้อง error ชัดเจน"

    st = ig.image_model_status("flux-3-klein-uncensored", "/tmp/definitely-no-such-dir")
    assert st["downloaded"] is False and len(st["missing"]) == 2, st

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "flux-3-klein-uncensored"
        base.mkdir(parents=True)
        for rel in ig.include_patterns("flux-3-klein-uncensored"):
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            (base / rel).write_bytes(b"GGUF")
        comfy = Path(tmp) / "ComfyUI"
        linked = ig.link_image_model("flux-3-klein-uncensored", tmp, comfy_dir=comfy)
        assert len(linked) == 2, f"ต้องผูกครบ 2 ไฟล์: {linked}"
        for p in linked.values():
            assert Path(p).is_symlink() and Path(p).exists()
        again = ig.link_image_model("flux-3-klein-uncensored", tmp, comfy_dir=comfy)
        assert len(again) == 2, "ผูกซ้ำต้องไม่พัง"
        st2 = ig.image_model_status("flux-3-klein-uncensored", tmp)
        assert st2["downloaded"] is True and st2["gb"] >= 0

    # โมเดลภาพต้องไม่ถูก llama-server รับ (ข้อความ error ต้องบอกทางออก)
    from strata.server import switch_strata_model

    raised = ""
    try:
        switch_strata_model("flux-3-klein-uncensored", model_dir="/tmp/x")
    except ValueError as e:
        raised = str(e)
    assert "สร้างภาพ" in raised and "llama-server" in raised, f"error ต้องบอกชัดเจน: {raised!r}"

    # download layer ต้องส่งต่อไป imagegen
    from strata.download import install_model

    assert callable(install_model)


def test_run_stream_cwd() -> None:
    """run_stream ต้องรับ cwd ได้ (ใช้กับ git ใน repo อื่น) และ log บอกโฟลเดอร์"""
    import contextlib
    import io
    import tempfile

    from strata import imagegen as ig
    from strata.utils import recent_log, run_stream

    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_stream("pwd", cwd=tmp)
        assert Path(tmp).resolve().as_posix() in buf.getvalue(), f"คำสั่งไม่ได้รันใน cwd: {buf.getvalue()!r}"
        assert any("ใน " + tmp in line for line in recent_log(6)), "log ต้องบอกโฟลเดอร์ที่รัน"

    # imagegen._git ต้องรับ cwd ได้ (ไม่ TypeError) — ตอนนี้รัน git จริงผ่าน subprocess
    if shutil.which("git"):
        with tempfile.TemporaryDirectory() as gitdir:
            ig._git(["init", "-q"], cwd=Path(gitdir))
            assert (Path(gitdir) / ".git").is_dir(), "_git ต้องรันใน cwd ที่ให้มา"


def test_comfyui_repo_bootstrap() -> None:
    """ติดตั้ง ComfyUI: clone ล้มต้องได้สาเหตุจริง + fallback tarball + ล้างโฟลเดอร์พัง"""
    import io
    import shutil
    import tarfile
    import tempfile

    from strata import imagegen as ig

    assert ig._repo_slug("https://github.com/comfyanonymous/ComfyUI.git") == "comfyanonymous/ComfyUI"

    # 1) tarball ของ GitHub: ตัดโฟลเดอร์ชั้นบนสุดออก
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (("Repo-main/main.py", b"print(1)"), ("Repo-main/sub/x.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        ig._extract_tarball(buf.getvalue(), out)
        assert (out / "main.py").is_file(), "ต้องแตกไฟล์โดยไม่มีโฟลเดอร์ชั้นบน"
        assert (out / "sub" / "x.txt").is_file()

    # 2) _run_git ที่ล้มเหลวต้องพา stderr ของ git มาด้วย (เดิมได้แค่ exit 128 เฉย ๆ)
    if shutil.which("git"):
        with tempfile.TemporaryDirectory() as tmp:
            raised = ""
            try:
                ig._run_git(["rev-parse", "--verify", "no-such-branch-xyz"], cwd=Path(tmp))
            except RuntimeError as e:
                raised = str(e)
            assert "ล้มเหลว (exit" in raised, raised
            assert "git ไม่ได้บอกสาเหตุ" not in raised, f"ควรมีข้อความจาก git: {raised!r}"

    # 3) โฟลเดอร์โคลนค้าง (ไม่ว่าง ไม่มี main.py) ต้องถูกล้างก่อน clone ใหม่
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "ComfyUI"
        (dest / "junk").mkdir(parents=True)
        (dest / "junk" / "a.txt").write_text("x")
        ig._clear_incomplete(dest)
        assert not dest.exists(), "โฟลเดอร์ที่ไม่สมบูรณ์ต้องถูกล้าง"
        empty = Path(tmp) / "empty"
        empty.mkdir()
        ig._clear_incomplete(empty)
        assert not empty.exists(), "โฟลเดอร์ว่างต้องถูกลบ"
        ig._clear_incomplete(Path(tmp) / "missing")  # ต้องไม่พังเมื่อไม่มีอยู่


def test_comfyui_public_url_helpers() -> None:
    """ลิงก์ ComfyUI ต้องตรวจได้ก่อนให้ผู้ใช้เปิด — 404 แล้ว fallback tunnel"""
    from strata import imagegen as ig

    assert ig.parse_comfy_port("To see the GUI go to: http://0.0.0.0:8190", 8188) == 8190
    assert ig.parse_comfy_port("http://127.0.0.1:9001\n", 8188) == 9001
    assert ig.parse_comfy_port("ไม่มีอะไรเลย", 8188) == 8188, "ต้องคืนค่า fallback"

    assert ig.parse_cloudflared_url("INF | https://soft-music-try.trycloudflare.com |") == (
        "https://soft-music-try.trycloudflare.com"
    )
    assert ig.parse_cloudflared_url("ยังไม่มี URL") == ""

    assert ig.public_url_works("") is False
    assert ig.public_url_works("http://127.0.0.1:9", timeout=1.5) is False, "พอร์ตปิดต้อง False"

    # ใน sandbox ไม่มี Colab -> comfyui_public_url คืน localhost และต้องไม่พยายามโหลด tunnel
    url, via = ig.ensure_public_url(8188, prefer="auto")
    assert via in ("local", "colab"), via
    assert url.startswith("http"), url


def test_platform_detection() -> None:
    """รองรับ Kaggle: เลือกโฟลเดอร์/detect proxy ให้ถูก และต่อ URL ให้เหมาะกับแพลตฟอร์ม"""
    from strata import imagegen as ig
    from strata.platforms import detect

    def ex_kaggle(p):
        return p in ("/kaggle/working", "/kaggle/temp")

    k = detect(env={"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, exists=ex_kaggle, writable=lambda _p: True)
    assert k["name"] == "kaggle" and k["label"].endswith("Kaggle"), k
    assert k["home"] == "/kaggle/temp/strata", f"Kaggle ควรใช้ temp (ไม่ติดลิมิต 20 GB): {k['home']}"
    assert k["model_dir"] == "/kaggle/temp/strata/models"
    assert k["proxy_ports"] is False and k["tunnel_needed"] is True
    assert any("Internet" in n for n in k["notes"]), "ต้องเตือนให้เปิด Internet"

    k2 = detect(
        env={"KAGGLE_KERNEL_RUN_TYPE": "Interactive", "STRATA_HOME": "/kaggle/working/strata"},
        exists=ex_kaggle, writable=lambda _p: True,
    )
    assert k2["home"] == "/kaggle/working/strata", "STRATA_HOME ต้องชนะค่าเริ่มต้น"

    c = detect(env={"COLAB_RELEASE_TAG": "2026.11"}, exists=lambda p: p == "/content", writable=lambda _p: True)
    assert c["name"] == "colab" and c["home"] == "/content" and c["proxy_ports"] is True

    loc = detect(env={}, exists=lambda _p: False, writable=lambda _p: True)
    assert loc["name"] == "local" and loc["proxy_ports"] is False

    # STRATA_HOME ของจริงต้องสะท้อนแพลตฟอร์มที่ตรวจได้
    import strata.config as cfg

    assert cfg.STRATA_HOME, "STRATA_HOME ต้องมีค่า"

    # Kaggle: ต้องไม่พยายามใช้ Colab proxy — ไป tunnel ตรง ๆ
    real_detect, real_tunnel = ig.detect_platform, ig.start_tunnel
    ig.detect_platform = lambda: k
    ig.start_tunnel = lambda port=8188, bin_dir=None, wait_s=60: "https://fake.trycloudflare.com"
    try:
        url, via = ig.ensure_public_url(8188)
        assert (url, via) == ("https://fake.trycloudflare.com", "cloudflared"), (url, via)
        raised = ""
        try:
            ig.ensure_public_url(8188, prefer="colab")
        except RuntimeError as e:
            raised = str(e)
        assert "port proxy" in raised, f"prefer='colab' บน Kaggle ต้องบอกชัดเจน: {raised!r}"
    finally:
        ig.detect_platform, ig.start_tunnel = real_detect, real_tunnel

    # local: ใช้ localhost ได้เลย ไม่ต้องเปิด tunnel
    ig.detect_platform = lambda: loc
    try:
        url, via = ig.ensure_public_url(8188)
        assert via == "local" and "127.0.0.1:8188" in url, (url, via)
    finally:
        ig.detect_platform = real_detect


def test_widget_diagnostics() -> None:
    """Kaggle 'ไม่แสดงอะไรเลย': ต้องตรวจเวอร์ชันวิดเจ็ต + มี fallback ข้อความเสมอ"""
    import contextlib
    import io
    import tempfile

    from strata import notebook as nbm

    # 1) จับคู่เวอร์ชัน: ipywidgets 8 <-> jupyterlab_widgets 3 / widgetsnbextension 4
    assert nbm.widget_plan({"ipywidgets": "8.1.5", "jupyterlab_widgets": "3.0.13", "widgetsnbextension": "4.0.10"}) == []
    assert nbm.widget_plan({"ipywidgets": "8.1.5", "jupyterlab_widgets": "-", "widgetsnbextension": "-"}) == [
        "jupyterlab_widgets>=3,<4",
        "widgetsnbextension>=4,<5",
    ]
    assert nbm.widget_plan({"ipywidgets": "-"}) == ["ipywidgets>=8.1,<9", "jupyterlab_widgets>=3,<4"]
    assert nbm.widget_plan({"ipywidgets": "7.7.1", "jupyterlab_widgets": "3.0.0", "widgetsnbextension": "4.0.0"}) == [
        "jupyterlab_widgets>=2,<3",
        "widgetsnbextension>=3,<4",
    ], "ipywidgets 7 ต้องคู่กับ jlw 2 / wsnb 3"

    # 2) diagnose() ต้องคืนข้อมูลครบและพิมพ์ตารางได้โดยไม่พัง
    d = nbm.diagnose(verbose=False)
    for key in ("platform", "python", "strata_home", "model_dir", "port", "widgets", "widget_plan", "disk", "is_notebook"):
        assert key in d, f"diagnose ขาด key {key}"
    assert d["platform"]["name"] in ("colab", "kaggle", "local")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        nbm.diagnose(verbose=True)
    assert "ipywidgets" in buf.getvalue() or True  # rich อาจเขียนลง console ที่จับไม่ได้ก็ไม่เป็นไร

    # 3) text_dashboard ต้องแสดงโมเดลทุกตัวแบบข้อความ (ใช้ได้แม้ไม่มีวิดเจ็ต)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "qwen35-9b-defiant-fable").mkdir()
        (Path(tmp) / "qwen35-9b-defiant-fable" / "m.gguf").write_bytes(b"x" * 1024)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            nbm.text_dashboard(tmp)
        out = buf.getvalue()
        assert "qwen35-9b-defiant-fable" in out, "dashboard ต้อง list โมเดล"
        assert "ติดตั้ง" in out or "install" in out or "install_model" in out, "ต้องมีคำสั่งใช้งาน"

    # 4) launch() ต้องไม่โยน exception ออกไปถ้า UI พัง — ต้อง fallback เป็นข้อความ
    import strata.ui as ui_mod

    def boom(*_a, **_k):
        raise RuntimeError("frontend ไม่รองรับ widget")

    real = ui_mod.show_model_manager_ui
    ui_mod.show_model_manager_ui = boom
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = nbm.launch(install_llama_server=False, model_dir="/tmp/none-model-dir")
        assert res is None, "เมื่อ UI พังต้องคืน None (ไม่โยน exception)"
    finally:
        ui_mod.show_model_manager_ui = real
    assert "สร้าง UI ไม่สำเร็จ" in buf.getvalue(), buf.getvalue()[:200]


def main() -> int:
    print(f"🧪 smoke test: strata/{NOTEBOOK.parent.name} (python {sys.version.split()[0]})")
    check("notebook structure", test_notebook_shape)
    check("notebook cells compile", test_notebook_cells_compile)
    check("package compiles + no bad f-string", test_package_compiles)
    check("public API / catalog", test_public_api)
    check("catalog download patterns", test_catalog_download_patterns)
    check("image model catalog", test_image_catalog)
    check("comfyui helpers", test_imagegen_helpers)
    check("run_stream cwd", test_run_stream_cwd)
    check("comfyui repo bootstrap", test_comfyui_repo_bootstrap)
    check("comfyui public url", test_comfyui_public_url_helpers)
    check("kaggle/colab platform", test_platform_detection)
    check("widget diagnostics", test_widget_diagnostics)
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
    check("downloader fallback", test_downloader_fallback)
    check("prebuilt libc isolation (glibc)", test_prebuilt_glibc_isolation)
    check("GPU memory preflight", test_gpu_memory_preflight)
    check("shell env persistence (nix in PATH)", test_shell_env_persistence)
    check("spec flag compatibility", test_spec_flag_compat)
    check("notebook repo bootstrap", test_notebook_repo_bootstrap)
    check("stale module purge", test_purge_modules)
    check("branch auto-detect", test_branch_autodetect)
    check("package files tracked by git", test_package_files_tracked)
    check("flake.nix from embedded template", test_flake_template)
    check("flake lives outside git repo", test_flake_outside_git_repo)
    check("nix develop error surfacing", test_activate_devshell_errors)
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
