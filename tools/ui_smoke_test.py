#!/usr/bin/env python3
"""UI smoke test — สร้างหน้าจอ Studio จริงด้วย ipywidgets แล้วกดปุ่มสำคัญ

ต่างจาก tools/smoke_test.py ตรงที่ไฟล์นี้ตรวจ "หน้าจอ" (widget tree) จริง:
- สร้าง UI ครบ 4 แท็บได้โดยไม่ error
- กดรีเฟรชซ้ำ ๆ แล้ว widget ของแถวโมเดลยังเป็นตัวเดิม (ไม่ถูกสร้างใหม่ทุกครั้ง = ไม่กระพริบ/ไม่กินแรง)
- ปุ่มโหลด/รัน/ลบ ส่งต่อไปยังฟังก์ชันที่ถูกต้อง (ด้วย stub)
- ปุ่มบันทึก/โหลดคลัง RAG เขียนและอ่านไฟล์ .npz ได้จริง

ต้องมี ipywidgets + numpy:  pip install ipywidgets numpy
ใช้: python3 tools/ui_smoke_test.py
"""
from __future__ import annotations

import ast
import hashlib
import html as html_lib
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "nix.ipynb"
FAILURES: list[str] = []

try:
    import ipywidgets as widgets
    import numpy as np
except ModuleNotFoundError as e:  # pragma: no cover
    print(f"⏭️  ข้าม UI test: ต้องติดตั้งก่อน ({e.name}): pip install ipywidgets numpy")
    raise SystemExit(0)


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  ✅ {name}")
    except AssertionError as e:
        FAILURES.append(f"{name}: {e}")
        print(f"  ❌ {name}: {e}")
    except Exception as e:  # noqa: BLE001
        FAILURES.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  ❌ {name}: {type(e).__name__}: {e}")


class FakeEmbedding:
    """embedder ปลอม 3 มิติ — พอสำหรับทดสอบ flow ของ UI โดยไม่ต้องโหลดโมเดลจริง"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def embed(self, texts):
        for _ in texts:
            yield np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def notebook_source() -> str:
    return "\n".join("".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code")


def build_ui(model_dir: Path, rag_path: Path):
    """exec เฉพาะส่วน UI ของ notebook ด้วย dependency ปลอม แล้วสร้างหน้าจอจริง"""
    src = notebook_source()
    tree = ast.parse(src)

    def node_of(name: str):
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return node
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
                return node
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                return node
        raise AssertionError(f"ไม่พบ {name} ใน notebook")

    calls: list[tuple] = []
    ns: dict = {
        "__name__": "ui_smoke", "Path": Path, "np": np, "widgets": widgets, "json": json, "re": re,
        "shutil": shutil, "html_lib": html_lib, "hashlib": hashlib, "glob": __import__("glob"), "time": time, "io": __import__("io"),
        "Callable": Callable, "PdfReader": object, "TextEmbedding": FakeEmbedding,
        "rprint": lambda *a, **k: None,
        "Console": type("Console", (), {"print": staticmethod(lambda *a, **k: None)}),
        "display": lambda w: ns.__setitem__("root", w),
        "check_disk_space": lambda *a, **k: {"total_gb": 100.0, "used_gb": 10.0, "free_gb": 90.0},
        "check_gpu_and_recommend": lambda verbose=False: {
            "has_gpu": True, "gpu_name": "T4", "vram_total_gb": 16.0, "vram_free_gb": 15.0,
            "ram_total_gb": 12.0, "ram_avail_gb": 10.0, "recommended_gpu_layers": 99,
            "recommended_ctx_size": 8192, "recommended_threads": 4},
        "_check_server_online": lambda port=18080, ttl=1.5: (False, "offline"),
        "_invalidate_dir_cache": lambda prefix=None: None,
        "_invalidate_server_cache": lambda port=None: None,
        "_find_draft_gguf": lambda folder: None,
        "DEFAULT_PORT": 18080, "DEFAULT_RAG_PATH": str(rag_path),
        "MODEL_DIR": str(model_dir), "DRAFT_DIR": str(model_dir / "_drafts"), "LAST_MODEL_DIR": None,
    }
    for fn in ("install_model", "install_all_models", "install_draft_model", "keep_only_models", "remove_model",
               "clean_download_cache", "start_strata_server", "stop_strata_server", "strata_server_status",
               "switch_strata_model", "get_openai_client"):
        ns[fn] = (lambda name: (lambda *a, **k: calls.append((name, a, k))))(fn)

    for name in ("ALL_PRESET_KEYS", "CUSTOM_HF_REPO", "CUSTOM_INCLUDE", "SPEC_STATE", "DRAFT_CATALOG", "MODEL_CATALOG"):
        exec(compile(ast.Module(body=[node_of(name)], type_ignores=[]), f"<{name}>", "exec"), ns)

    exec(compile(ast.Module(body=[node_of("StrataRAG")], type_ignores=[]), "<StrataRAG>", "exec"), ns)
    ns["GLOBAL_RAG"] = ns["StrataRAG"]()
    ns["_dir_stats"] = lambda p, ttl=30.0: {"bytes": 0, "gguf": 0, "cached": False}

    for name in ("_fmt_reply_html", "_L", "_B", "show_model_manager_ui"):
        exec(compile(ast.Module(body=[node_of(name)], type_ignores=[]), f"<{name}>", "exec"), ns)

    ns["show_model_manager_ui"](model_dir=str(model_dir), port=18080, rag_engine=ns["GLOBAL_RAG"])
    return ns, ns.get("root"), calls


def walk(w):
    yield w
    for ch in getattr(w, "children", ()) or ():
        yield from walk(ch)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        model_dir = tmp / "models"
        model_dir.mkdir()
        rag_path = tmp / "store.npz"
        state: dict = {}

        def build():
            ns, root, calls = build_ui(model_dir, rag_path)
            assert root is not None, "display() ไม่ได้ถูกเรียก — UI ไม่ถูกสร้าง"
            state.update(ns=ns, root=root, calls=calls, all=list(walk(root)))

        check("สร้าง UI ครบทุกแท็บ", build)

        def check_structure():
            all_w = state["all"]
            tabs = [w for w in all_w if isinstance(w, widgets.Tab)]
            assert tabs and len(tabs[0].children) == 4, "ต้องมี 4 แท็บ"
            titles = [tabs[0].get_title(i) for i in range(4)]
            assert any("แชท" in t for t in titles), f"ชื่อแท็บผิดปกติ: {titles}"
            btns = [w for w in all_w if isinstance(w, widgets.Button)]
            n_models = len(state["ns"]["ALL_PRESET_KEYS"])
            for label in ("📥 โหลด", "🚀 รัน & แชท", "📌 เก็บตัวเดียว", "🗑️ ลบ"):
                assert sum(b.description == label for b in btns) == n_models, f"ปุ่ม '{label}' ต้องมี {n_models} ปุ่ม"
            for desc in ("💾 บันทึกคลัง", "📂 โหลดคลังเดิม"):
                assert any(b.description == desc for b in btns), f"ไม่พบปุ่ม {desc}"
        check("จำนวน/widget ในแต่ละแท็บ", check_structure)

        def check_refresh_reuse():
            rows = [w for w in state["all"] if isinstance(w, widgets.VBox) and len(w.children) == len(state["ns"]["ALL_PRESET_KEYS"])]
            assert rows, "ไม่พบกล่องแถวโมเดล"
            rows_box = rows[0]
            before = [id(c) for c in rows_box.children]
            ref = next(b for b in state["all"] if isinstance(b, widgets.Button) and "รีเฟรช" in b.description)
            ref.click()
            ref.click()
            after = [id(c) for c in rows_box.children]
            assert before == after, "รีเฟรชแล้ว widget ถูกสร้างใหม่ (ควรอัปเดตค่าเดิมเท่านั้น)"
        check("รีเฟรชซ้ำโดยไม่สร้าง widget ใหม่", check_refresh_reuse)

        def click(desc: str, index: int = 0):
            btns = [b for b in state["all"] if isinstance(b, widgets.Button) and b.description == desc]
            assert len(btns) > index, f"ไม่พบปุ่ม {desc}[{index}]"
            state["calls"].clear()
            btns[index].click()
            return state["calls"]

        def check_row_actions():
            calls = click("📥 โหลด")
            assert calls and calls[0][0] == "install_model", f"ปุ่มโหลดควรเรียก install_model ได้ {calls[:1]}"
            assert calls[0][2].get("auto_start_server") is not True, "ปุ่มโหลดไม่ควรสตาร์ท server"
            calls = click("🚀 รัน & แชท")
            assert calls and calls[0][0] == "install_model" and calls[0][2].get("auto_start_server") is True, \
                f"เมื่อยังไม่มีไฟล์ ปุ่มรันต้องโหลดก่อนพร้อมสตาร์ท ได้ {calls[:1]}"
            calls = click("🗑️ ลบ")
            assert calls and calls[0][0] == "remove_model", f"ปุ่มลบควรเรียก remove_model ได้ {calls[:1]}"
        check("ปุ่มในแถวโมเดลทำงานถูก", check_row_actions)

        def check_rag_persistence():
            ns = state["ns"]
            rag = ns["GLOBAL_RAG"]
            path_in = next(w for w in walk(state["root"]) if isinstance(w, widgets.Text) and w.description == "ไฟล์คลัง:")
            path_in.value = str(rag_path)
            rag.add_documents([{"source": "note", "text": "ทดสอบบันทึกคลัง RAG"}], chunk_size=500)
            click("💾 บันทึกคลัง")
            assert rag_path.is_file(), "กดบันทึกแล้วไม่พบไฟล์คลัง"
            rag.clear()
            assert len(rag.chunks) == 0
            click("📂 โหลดคลังเดิม")
            assert len(rag.chunks) == 1, f"โหลดคลังกลับมาไม่ครบ: {len(rag.chunks)}"
            assert ns["StrataRAG"]().load(rag_path) == 1, "ไฟล์คลังต้องอ่านได้ด้วย instance ใหม่"
        check("บันทึก/โหลดคลัง RAG จากปุ่ม", check_rag_persistence)

        if FAILURES:
            print(f"\n❌ ล้มเหลว {len(FAILURES)} รายการ")
            return 1
        print("\n✅ ผ่านทั้งหมด")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
