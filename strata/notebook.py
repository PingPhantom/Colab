"""ตัวช่วยสำหรับใช้ใน Colab/Jupyter: ลงทะเบียน magic และเปิด UI ในคำสั่งเดียว"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from IPython.core.getipython import get_ipython
from IPython.core.magic import register_cell_magic
from rich.console import Console
from rich.syntax import Syntax


def register_magics() -> None:
    """ลงทะเบียน `%%ruff` — ฟอร์แมต/จัดเรียงโค้ดในเซลล์ก่อนรัน (ข้ามการรันด้วย `--no-run`)"""

    @register_cell_magic
    def ruff(line, cell):  # noqa: ANN001 - signature บังคับโดย IPython
        no_run = "--no-run" in line.split()
        config = PyprojectConfig.path()
        chk = subprocess.run(
            ["ruff", "check", "--fix", "--quiet", "--config", config, "--stdin-filename", "cell.py", "-"],
            input=cell,
            text=True,
            capture_output=True,
            check=False,
        )
        fixed = chk.stdout if chk.returncode in (0, 1) and chk.stdout else cell
        fmt = subprocess.run(
            ["ruff", "format", "--quiet", "--config", config, "--stdin-filename", "cell.py", "-"],
            input=fixed,
            text=True,
            capture_output=True,
            check=False,
        )
        code = fmt.stdout if fmt.returncode == 0 else fixed
        if code.strip() != cell.strip() or no_run:
            Console().print(Syntax(code.strip(), "python", theme="monokai", line_numbers=True))
        if not no_run:
            get_ipython().run_cell(code)


class PyprojectConfig:
    """หา pyproject.toml ที่ใกล้ที่สุด (ใช้เป็น config ของ ruff)"""

    @staticmethod
    def path() -> str:
        for base in (Path.cwd(), *Path.cwd().parents):
            candidate = base / "pyproject.toml"
            if candidate.is_file():
                return str(candidate)
        return str(Path.cwd() / "pyproject.toml")


# ---------------------------------------------------------------------------
# วิดเจ็ตไม่แสดงผล (เช่นบน Kaggle) — ตรวจ/ซ่อมเวอร์ชัน + มี fallback เป็นข้อความ
# ---------------------------------------------------------------------------
def _ver(module: str) -> str:
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - ไม่มีโมดูล/โหลดพังก็รายงานเป็น "-"
        return "-"
    return str(getattr(mod, "__version__", "?"))


def widget_versions() -> dict:
    return {
        "ipywidgets": _ver("ipywidgets"),
        "jupyterlab_widgets": _ver("jupyterlab_widgets"),
        "widgetsnbextension": _ver("widgetsnbextension"),
    }


def widget_plan(versions: dict) -> list[str]:
    """บอกว่าต้องติดตั้งอะไรเพิ่มเพื่อให้วิดเจ็ตแสดงผล (frontend/manager เวอร์ชันต้องเข้าคู่กับ ipywidgets)

    ipywidgets 8 ↔ jupyterlab_widgets 3 / widgetsnbextension 4
    ipywidgets 7 ↔ jupyterlab_widgets 2 / widgetsnbextension 3
    """
    iw = str(versions.get("ipywidgets", "-"))
    if not iw or iw == "-":
        return ["ipywidgets>=8.1,<9", "jupyterlab_widgets>=3,<4"]
    major = iw.split(".")[0]
    want = {"7": ("jupyterlab_widgets>=2,<3", "widgetsnbextension>=3,<4"),
            "8": ("jupyterlab_widgets>=3,<4", "widgetsnbextension>=4,<5")}.get(major)
    if want is None:  # เวอร์ชันใหม่กว่านี้: ปล่อยให้ pip เลือกคู่ที่เข้ากันเอง
        return ["ipywidgets", "jupyterlab_widgets"]
    plan = []
    jlw = str(versions.get("jupyterlab_widgets", "-"))
    if jlw == "-" or jlw.split(".")[0] != want[0].split(">=")[1].split(",")[0].split(".")[0]:
        plan.append(want[0])
    wsnb = str(versions.get("widgetsnbextension", "-"))
    if wsnb == "-" or wsnb.split(".")[0] != want[1].split(">=")[1].split(",")[0].split(".")[0]:
        plan.append(want[1])
    return plan


def ensure_widgets(install: bool = True) -> dict:
    """ตรวจความเข้ากันของ ipywidgets กับตัว frontend แล้วติดตั้งส่วนที่ขาดให้ (idempotent)"""
    before = widget_versions()
    plan = widget_plan(before)
    if plan and install:
        rprint = _rprint()
        rprint(f"[yellow]🧩 วิดเจ็ตอาจไม่แสดงผล: {before} → ติดตั้ง {', '.join(plan)}[/yellow]")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *plan],
            check=False, capture_output=True, text=True,
        )
        importlib.invalidate_caches()
    return {"before": before, "plan": plan, "after": widget_versions()}


def _rprint():
    from rich import print as rprint

    return rprint


def diagnose(verbose: bool = True) -> dict:
    """สรุปสภาพแวดล้อมไว้ดูเวลา "ไม่แสดงผลอะไรเลย" — คืน dict เสมอ (ใช้ในเทสต์ได้)"""
    from .config import DEFAULT_PORT, MODEL_DIR, STRATA_HOME
    from .platforms import detect

    info = detect()
    disk = {}
    try:
        import shutil as _shutil

        base = Path(MODEL_DIR) if Path(MODEL_DIR).exists() else Path(info["home"])
        du = _shutil.disk_usage(base if base.exists() else Path.home())
        disk = {"path": str(base), "free_gb": round(du.free / 1024**3, 2), "total_gb": round(du.total / 1024**3, 2)}
    except OSError:
        disk = {"path": str(MODEL_DIR), "free_gb": 0.0, "total_gb": 0.0}
    out = {
        "platform": info,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "strata_home": STRATA_HOME,
        "model_dir": MODEL_DIR,
        "port": DEFAULT_PORT,
        "widgets": widget_versions(),
        "widget_plan": widget_plan(widget_versions()),
        "disk": disk,
        "is_notebook": get_ipython() is not None,
    }
    if verbose:
        from rich.console import Console
        from rich.table import Table

        t = Table(title="strata diagnose", show_lines=False)
        t.add_column("รายการ", style="cyan")
        t.add_column("ค่า")
        t.add_row("แพลตฟอร์ม", info["label"])
        t.add_row("python", out["python"])
        t.add_row("STRATA_HOME", out["strata_home"])
        t.add_row("MODEL_DIR", out["model_dir"])
        t.add_row("พอร์ต", str(out["port"]))
        t.add_row("ipywidgets", out["widgets"]["ipywidgets"])
        t.add_row("jupyterlab_widgets", out["widgets"]["jupyterlab_widgets"])
        t.add_row("widgetsnbextension", out["widgets"]["widgetsnbextension"])
        t.add_row("ดิสก์ว่าง", f"{disk['free_gb']} / {disk['total_gb']} GB ({disk['path']})")
        t.add_row("อยู่ใน notebook", "ใช่" if out["is_notebook"] else "ไม่")
        Console().print(t)
        if out["widget_plan"]:
            Console().print(
                "[yellow]⚠️ เวอร์ชันวิดเจ็ตไม่เข้าคู่กัน → UI อาจว่างเปล่า[/yellow]\n"
                f"[yellow]   แก้: %pip install -q {' '.join(out['widget_plan'])} แล้ว Restart & Run All[/yellow]"
            )
    return out


def text_dashboard(model_dir: str | None = None) -> None:
    """fallback เป็นข้อความล้วน — ใช้เมื่อวิดเจ็ตแสดงผลไม่ได้ แต่ยังอยากเห็นสถานะ/คำสั่ง"""
    from rich.console import Console
    from rich.table import Table

    from .config import IMAGE_PRESET_KEYS, MODEL_CATALOG, MODEL_DIR
    from .utils import _dir_bytes, _fmt_bytes

    base = Path(model_dir or MODEL_DIR)
    con = Console()
    t = Table(title="📦 โมเดลในแคตตาล็อก (โหมดข้อความ)")
    t.add_column("#", justify="right")
    t.add_column("key", style="cyan")
    t.add_column("ขนาด", justify="right")
    t.add_column("สถานะ")
    for i, (key, cfg) in enumerate(MODEL_CATALOG.items(), 1):
        if key == "custom":
            continue
        sub = base / key
        have = _dir_bytes(sub) if sub.is_dir() else 0
        kind = "🎨" if key in IMAGE_PRESET_KEYS else "💬"
        t.add_row(
            str(i),
            f"{kind} {key}",
            f"~{cfg['approx_gb']:.1f} GB",
            f"✅ {_fmt_bytes(have)}" if have else "⬜ ยังไม่โหลด",
        )
    con.print(t)
    con.print(
        "[bold]คำสั่งที่ใช้ได้แม้ไม่มี UI[/bold]\n"
        "  from strata.download import install_model; install_model('qwen35-9b-defiant-fable', auto_start_server=True)\n"
        "  from strata.server import get_openai_client; print(get_openai_client().models.list())\n"
        "  from strata.imagegen import install_image_model, start_comfyui   # สร้างภาพ"
    )


def launch(model_dir: str | None = None, port: int | None = None, install_llama_server: bool = True, fallback_text: bool = True):
    """เปิด Studio UI — ถ้าวิดเจ็ตใช้ไม่ได้จะพิมพ์ dashboard ข้อความแทน (ไม่ทิ้งผู้ใช้ไว้กับจอว่าง)"""
    from .config import DEFAULT_PORT, MODEL_DIR
    from .server import install_llama_server as install_server
    from .ui import show_model_manager_ui

    try:
        ensure_widgets()
    except Exception as e:  # noqa: BLE001 - ซ่อมไม่ได้ก็ต้องเปิด UI ต่อให้ได้
        _rprint()(f"[yellow]⚠️ ตรวจวิดเจ็ตไม่สำเร็จ: {e}[/yellow]")
    if install_llama_server:
        install_server()
    try:
        return show_model_manager_ui(model_dir=model_dir or MODEL_DIR, port=port or DEFAULT_PORT)
    except Exception as e:  # noqa: BLE001 - frontend พัง/วิดเจ็ตแสดงไม่ได้
        _rprint()(f"[bold red]❌ สร้าง UI ไม่สำเร็จ: {e}[/bold red]")
        if fallback_text:
            _rprint()("[yellow]→ แสดงสถานะแบบข้อความแทน (ดูวิธีแก้ด้านล่าง)[/yellow]")
            diagnose()
            text_dashboard(model_dir or MODEL_DIR)
        return None
