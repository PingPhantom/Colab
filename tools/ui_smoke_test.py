#!/usr/bin/env python3
"""UI smoke test — สร้างหน้าจอ Studio จริงด้วย ipywidgets แล้วกดปุ่มสำคัญ

ตรวจว่า:
- สร้าง UI ครบ 4 แท็บด้วย `strata.ui.show_model_manager_ui()`
- กดรีเฟรชซ้ำ ๆ แล้ว widget ของแถวโมเดลยังเป็น object เดิม (ไม่ถูกสร้างใหม่ = ไม่กระพริบ/ไม่กินแรง)
- ปุ่มโหลด / รัน / ลบ ส่งต่อไปยังฟังก์ชันที่ถูกต้อง
- ปุ่มบันทึก/โหลดคลัง RAG เขียนและอ่านไฟล์ .npz ได้จริง

ต้องมี: ipywidgets + numpy      (pip install ipywidgets numpy)
ใช้: python3 tools/ui_smoke_test.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []

try:
    import ipywidgets as widgets
    import numpy as np
except ModuleNotFoundError as e:  # pragma: no cover
    print(f"⏭️  ข้าม UI test: ต้องติดตั้งก่อน ({e.name}): pip install ipywidgets numpy")
    raise SystemExit(0) from None

sys.path.insert(0, str(ROOT))

from strata import ui as strata_ui  # noqa: E402
from strata.rag import StrataRAG  # noqa: E402


class FakeEmbedding:
    """embedder ปลอม 3 มิติ — พอสำหรับทดสอบ flow ของ UI โดยไม่ต้องโหลดโมเดลจริง"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def embed(self, texts):
        for _ in texts:
            yield np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


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


def walk(w):
    yield w
    for ch in getattr(w, "children", ()) or ():
        yield from walk(ch)


def patch_ui(monkey: dict) -> dict:
    """สลับฟังก์ชันที่แตะระบบจริงใน namespace ของ strata.ui ด้วย stub แล้วคืนค่าเดิม"""
    saved = {}
    for name, value in monkey.items():
        saved[name] = getattr(strata_ui, name)
        setattr(strata_ui, name, value)
    return saved


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        model_dir = tmp / "models"
        model_dir.mkdir()
        rag_path = tmp / "store.npz"
        rag = StrataRAG()
        rag._embedder = FakeEmbedding()
        calls: list[tuple] = []
        displayed: dict = {}

        def recorder(name: str):
            def _fn(*a, **k):
                calls.append((name, a, k))
                return {"option": a[0] if a else None}

            return _fn

        monkey = {
            "install_model": recorder("install_model"),
            "install_all_models": recorder("install_all_models"),
            "install_draft_model": recorder("install_draft_model"),
            "keep_only_models": recorder("keep_only_models"),
            "remove_model": recorder("remove_model"),
            "clean_download_cache": recorder("clean_download_cache"),
            "start_strata_server": recorder("start_strata_server"),
            "stop_strata_server": recorder("stop_strata_server"),
            "strata_server_status": recorder("strata_server_status"),
            "switch_strata_model": recorder("switch_strata_model"),
            "check_gpu_and_recommend": lambda verbose=False: {
                "has_gpu": True,
                "gpu_name": "T4",
                "vram_total_gb": 16.0,
                "vram_free_gb": 15.0,
                "ram_total_gb": 12.0,
                "ram_avail_gb": 10.0,
                "recommended_gpu_layers": 99,
                "recommended_ctx_size": 8192,
                "recommended_threads": 4,
            },
            "_check_server_online": lambda port=18080, ttl=1.5: (False, "offline"),
            "_dir_stats": lambda p, ttl=30.0: {"bytes": 0, "gguf": 0, "cached": False},
            "_invalidate_dir_cache": lambda prefix=None: None,
            "_invalidate_server_cache": lambda port=None: None,
            "display": lambda w: displayed.setdefault("root", w),
        }
        saved = patch_ui(monkey)
        try:
            def build():
                root = strata_ui.show_model_manager_ui(model_dir=str(model_dir), port=18080, rag_engine=rag)
                displayed.setdefault("root", root)
                assert displayed.get("root") is not None, "UI ไม่ถูกสร้าง (display ไม่ถูกเรียก)"

            check("สร้าง UI ครบทุกแท็บ", build)
            all_w = list(walk(displayed["root"]))

            def check_structure():
                tabs = [w for w in all_w if isinstance(w, widgets.Tab)]
                assert tabs and len(tabs[0].children) == 4, "ต้องมี 4 แท็บ"
                titles = [tabs[0].get_title(i) for i in range(4)]
                assert any("แชท" in t for t in titles), f"ชื่อแท็บผิดปกติ: {titles}"
                btns = [w for w in all_w if isinstance(w, widgets.Button)]
                n_models = len(strata_ui.ALL_PRESET_KEYS)
                for label in ("📥 โหลด", "🚀 รัน & แชท", "📌 เก็บตัวเดียว", "🗑️ ลบ"):
                    assert sum(b.description == label for b in btns) == n_models, f"ปุ่ม '{label}' ต้องมี {n_models} ปุ่ม"
                for desc in ("💾 บันทึกคลัง", "📂 โหลดคลังเดิม"):
                    assert any(b.description == desc for b in btns), f"ไม่พบปุ่ม {desc}"

            check("จำนวน/widget ในแต่ละแท็บ", check_structure)

            def check_refresh_reuse():
                rows = [w for w in all_w if isinstance(w, widgets.VBox) and len(w.children) == len(strata_ui.ALL_PRESET_KEYS)]
                assert rows, "ไม่พบกล่องแถวโมเดล"
                rows_box = rows[0]
                before = [id(c) for c in rows_box.children]
                ref = next(b for b in all_w if isinstance(b, widgets.Button) and "รีเฟรช" in b.description)
                ref.click()
                ref.click()
                assert [id(c) for c in rows_box.children] == before, "รีเฟรชแล้ว widget ถูกสร้างใหม่"

            check("รีเฟรชซ้ำโดยไม่สร้าง widget ใหม่", check_refresh_reuse)

            def click(desc: str, index: int = 0):
                btns = [b for b in all_w if isinstance(b, widgets.Button) and b.description == desc]
                assert len(btns) > index, f"ไม่พบปุ่ม {desc}[{index}]"
                calls.clear()
                btns[index].click()
                return calls

            def check_row_actions():
                got = click("📥 โหลด")
                assert got and got[0][0] == "install_model", f"ปุ่มโหลดควรเรียก install_model ได้ {got[:1]}"
                assert got[0][2].get("auto_start_server") is not True, "ปุ่มโหลดไม่ควรสตาร์ท server"
                got = click("🚀 รัน & แชท")
                assert got and got[0][0] == "install_model" and got[0][2].get("auto_start_server") is True, \
                    f"เมื่อยังไม่มีไฟล์ ปุ่มรันต้องโหลดก่อนพร้อมสตาร์ท ได้ {got[:1]}"
                got = click("🗑️ ลบ")
                assert got and got[0][0] == "remove_model", f"ปุ่มลบควรเรียก remove_model ได้ {got[:1]}"

            check("ปุ่มในแถวโมเดลทำงานถูก", check_row_actions)

            def check_buttons_reenabled():
                ref = next(b for b in all_w if isinstance(b, widgets.Button) and "รีเฟรช" in b.description)
                click("📥 โหลด")  # ผ่าน _act() -> ปิดปุ่มทั้งหมดชั่วคราว
                assert ref.disabled is False, "หลังโหลดเสร็จปุ่มรีเฟรชต้องกลับมากดได้"
                tops = [
                    b for b in all_w
                    if isinstance(b, widgets.Button)
                    and b.description in ("📦 โหลดครบ 10 ตัว", "🧹 ล้างแคช", "📡 สถานะ", "🛑 หยุด Server", "🗑️ ลบทั้งหมด")
                ]
                stuck = [b.description for b in tops if b.disabled]
                assert not stuck, f"ปุ่มค้าง disabled หลังทำงานเสร็จ: {stuck}"
                click("🔄 รีเฟรช")
                assert ref.disabled is False, "กดรีเฟรชแล้วปุ่มต้องยังใช้งานได้"

            check("ปุ่มกลับมากดได้หลังทำงานเสร็จ", check_buttons_reenabled)

            def check_network_widgets():
                header = next(
                    w for w in walk(displayed["root"]) if isinstance(w, widgets.HTML) and "Strata Studio (Port" in str(w.value)
                )
                assert "⬇" in header.value and "⬆" in header.value, "header ต้องแสดง DL/UL"
                btn = next(b for b in all_w if isinstance(b, widgets.Button) and "วัดเน็ต" in b.description)
                assert btn.description == "📶 วัดเน็ตเรียลไทม์"
                btn.click()
                assert "หยุด" in btn.description, f"กดเริ่มแล้วปุ่มควรเปลี่ยนเป็นหยุด ได้ {btn.description}"
                btn.click()
                assert "หยุด" not in btn.description, f"กดหยุดแล้วปุ่มควรกลับมา ได้ {btn.description}"

            check("แสดง DL/UL ใน header + ปุ่มวัดเรียลไทม์", check_network_widgets)

            def check_live_refresh():
                from strata.utils import _remember  # noqa: PLC0415

                assert strata_ui.LIVE_REFS, "ไม่พบตัวรีเฟรชเรียลไทม์ (LIVE_REFS ว่าง)"
                ref = strata_ui.LIVE_REFS[-1]
                live_out = next(
                    w for w in walk(displayed["root"]) if isinstance(w, widgets.HTML) and "เรียลไทม์ (ทุก" in str(w.value)
                )
                btn = next(b for b in all_w if isinstance(b, widgets.Button) and "⏱ เรียลไทม์" in b.description)

                _remember("ทดสอบ log เรียลไทม์")
                before = ref.ticks
                ref.run_once()
                assert ref.ticks == before + 1, "run_once ต้องเดิน tick"
                assert "ทดสอบ log เรียลไทม์" in live_out.value, f"แผง log ไม่แสดงบรรทัดใหม่: {live_out.value[:200]}"

                btn.click()
                assert ref.running is False, "กดปิดแล้วตัวรีเฟรชต้องหยุด"
                assert "ปิด" in btn.description, f"ปุ่มควรบอกว่าปิดอยู่: {btn.description}"
                btn.click()
                assert ref.running is True, "กดเปิดอีกครั้งต้องกลับมาทำงาน"
                ref.stop()  # ไม่ให้ thread ค้างตอนจบเทสต์

            check("รีเฟรช UI เรียลไทม์ + log สด", check_live_refresh)

            def check_rag_persistence():
                path_in = next(w for w in walk(displayed["root"]) if isinstance(w, widgets.Text) and w.description == "ไฟล์คลัง:")
                path_in.value = str(rag_path)
                rag.add_documents([{"source": "note", "text": "ทดสอบบันทึกคลัง RAG"}], chunk_size=500)
                click("💾 บันทึกคลัง")
                assert rag_path.is_file(), "กดบันทึกแล้วไม่พบไฟล์คลัง"
                rag.clear()
                assert len(rag.chunks) == 0
                click("📂 โหลดคลังเดิม")
                assert len(rag.chunks) == 1, f"โหลดคลังกลับมาไม่ครบ: {len(rag.chunks)}"
                fresh = StrataRAG()
                fresh._embedder = FakeEmbedding()
                assert fresh.load(rag_path) == 1, "ไฟล์คลังต้องอ่านได้ด้วย instance ใหม่"

            check("บันทึก/โหลดคลัง RAG จากปุ่ม", check_rag_persistence)
        finally:
            patch_ui(saved)

        if FAILURES:
            print(f"\n❌ ล้มเหลว {len(FAILURES)} รายการ")
            return 1
        print("\n✅ ผ่านทั้งหมด")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
