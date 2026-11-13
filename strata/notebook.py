"""ตัวช่วยสำหรับใช้ใน Colab/Jupyter: ลงทะเบียน magic และเปิด UI ในคำสั่งเดียว"""

from __future__ import annotations

import subprocess
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


def launch(model_dir: str | None = None, port: int | None = None, install_llama_server: bool = True):
    """เปิด Studio UI (ค่าเริ่มต้นอ่านจาก strata.config) — คืน widget หลัก"""
    from .config import DEFAULT_PORT, MODEL_DIR
    from .server import install_llama_server as install_server
    from .ui import show_model_manager_ui

    if install_llama_server:
        install_server()
    return show_model_manager_ui(model_dir=model_dir or MODEL_DIR, port=port or DEFAULT_PORT)
