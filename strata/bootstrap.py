"""เตรียมสภาพแวดล้อมสำหรับ Colab: Nix (ปักหมุด nixpkgs) → devshell → uv venv

หลักการ: ไม่มี `curl | sh` (ดาวน์โหลดสคริปต์มาตรวจก่อนรัน + log sha256 ให้ปักหมุดได้),
ไม่ลบไฟล์ระบบของ Colab (ใช้ symlink), และเรียกซ้ำได้ปลอดภัย (idempotent)
"""

from __future__ import annotations

import glob
import hashlib
import importlib
import json
import os
import shlex
import shutil
import site
import subprocess
import sys
from pathlib import Path

import requests
from rich import print as rprint

from .config import (
    CUSTOM_HF_REPO,
    MODEL_DIR,
    NVIDIA_BIN_DIRS,
    NVIDIA_LIB_DIRS,
    STRATA_HOME,
)
from .flake_template import FLAKE_TEMPLATE
from .utils import run_quiet, run_stream

# ปักหมุด nixpkgs (nixos-unstable ณ 2026-11-11) — แก้ค่านี้เมื่อต้องการอัปเดต env
NIXPKGS_REV = "d4f25a3e562008648d45bcf2d70bcc7422cad84e"
NIX_INSTALL_URL = "https://nixos.org/nix/install"

# ใช้เมื่อ repo ไม่มี pyproject.toml (ปกติ repo นี้มี — แล้วจะใช้ `uv sync`)
FALLBACK_DEPS = [
    "numpy>=1.26,<3",
    "requests>=2.31,<3",
    "rich>=13.7,<16",
    "openai>=1.55,<4",
    "fastembed>=0.4,<1",
    "pypdf>=4.2,<8",
    "ipywidgets>=8.1,<9",
]

NIX_CONF = """experimental-features = nix-command flakes
sandbox = false
filter-syscalls = false
build-users-group =
max-jobs = auto
cores = 0
http-connections = 50
"""


def _log(msg: str) -> None:
    rprint(f"[cyan]{msg}[/cyan]")


# ---------------------------------------------------------------- Nix
def write_nix_conf() -> None:
    Path("/etc/nix").mkdir(parents=True, exist_ok=True)
    Path("/etc/nix/nix.conf").write_text(NIX_CONF, encoding="utf-8")


def ensure_paths() -> None:
    """เติม PATH/ LD_LIBRARY_PATH ของ Nix + NVIDIA (Colab เก็บ lib ไว้นอกมาตรฐาน)"""
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH") or NVIDIA_LIB_DIRS
    extra = [
        str(Path.home() / ".local/bin"),
        str(Path.home() / ".nix-profile/bin"),
        "/nix/var/nix/profiles/default/bin",
        "/usr/local/nvidia/bin",
        "/usr/local/cuda/bin",
    ]
    for p in reversed(extra):
        if p not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = f"{p}{os.pathsep}{os.environ.get('PATH', '')}"


def install_nix() -> None:
    """ติดตั้ง Nix แบบ single-user โดยดาวน์โหลดสคริปต์มาตรวจก่อน (ไม่ pipe เข้า sh ตรง ๆ)

    ตั้ง env `NIX_INSTALL_SHA256` เพื่อบังคับให้สคริปต์ตรงกับ hash ที่ตรวจไว้แล้ว
    """
    if shutil.which("nix"):
        return
    _log("⏳ กำลังติดตั้ง Nix Package Manager...")
    resp = requests.get(NIX_INSTALL_URL, timeout=120)
    resp.raise_for_status()
    script = resp.content
    digest = hashlib.sha256(script).hexdigest()
    rprint(f"[dim]   installer sha256 = {digest}[/dim]")

    expected = os.environ.get("NIX_INSTALL_SHA256", "").strip().lower()
    if expected and digest != expected:
        raise RuntimeError(f"sha256 ของสคริปต์ติดตั้ง Nix ไม่ตรงที่คาดไว้ (ได้ {digest}) — ยกเลิกเพื่อความปลอดภัย")
    if not script.startswith(b"#!") or b"nix" not in script[:8192].lower():
        raise RuntimeError("เนื้อหาที่ดาวน์โหลดมาไม่ใช่สคริปต์ติดตั้ง Nix — ยกเลิกเพื่อความปลอดภัย")

    installer = Path("/tmp/nix-install.sh")
    installer.write_bytes(script)
    installer.chmod(0o700)
    run_quiet(f"groupadd -r nixbld 2>/dev/null || true && sh {shlex.quote(str(installer))} --no-daemon")


def write_flake(repo_id: str = CUSTOM_HF_REPO) -> Path:
    """เขียน flake.nix จาก template (ปักหมุด nixpkgs rev + model dir ให้ตรงกับ config)"""
    target = Path("flake.nix")
    py_ver = f"python{sys.version_info.major}{sys.version_info.minor}"
    content = (
        FLAKE_TEMPLATE
        .replace("@NIXPKGS_REV@", NIXPKGS_REV)
        .replace("@PY_VER@", py_ver)
        .replace("@CUSTOM_HF_REPO@", repo_id)
        .replace("@MODEL_DIR@", MODEL_DIR)
    )
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")
    Path("default.nix").unlink(missing_ok=True)
    return target


def link_tools() -> None:
    """วาง symlink ของ tool จาก devshell ไว้ใน /usr/local/bin (ทับได้ ไม่ลบของเดิม)"""
    # bash/sh: symlink ไปของจริงแทนการลบ (เดิมลบทิ้ง ซึ่งเสี่ยงกับ runtime ของ Colab)
    for name, real in (("bash", "/bin/bash"), ("sh", "/bin/sh")):
        dst = Path("/usr/local/bin") / name
        try:
            if Path(real).exists() and (not dst.exists() or dst.resolve() != Path(real)):
                dst.unlink(missing_ok=True)
                dst.symlink_to(real)
        except OSError:
            pass
    for tool in ("uv", "ruff", "jq", "rg", "hf", "download-deepseek"):
        src = shutil.which(tool)
        if not src:
            continue
        dst = Path("/usr/local/bin") / tool
        try:
            if not dst.exists() or dst.resolve() != Path(src).resolve():
                dst.unlink(missing_ok=True)
                dst.symlink_to(src)
        except OSError:
            pass


def activate_devshell() -> None:
    """ดึง env จาก `nix develop` มาใส่ process ปัจจุบัน (PATH/LD_LIBRARY_PATH รวมกับของเดิม)"""
    _log("⏳ กำลังโหลด Nix devshell...")
    env_json = subprocess.check_output(
        ["nix", "develop", "--command", "python3", "-c", "import os, json; print(json.dumps(dict(os.environ)))"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    for key, val in json.loads(env_json).items():
        if key in ("_", "SHLVL", "PWD", "OLDPWD"):
            continue
        if key == "PATH":
            os.environ["PATH"] = f"/usr/bin:/bin:{val}:{NVIDIA_BIN_DIRS}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin"
        elif key == "LD_LIBRARY_PATH":
            os.environ["LD_LIBRARY_PATH"] = f"{val}:{NVIDIA_LIB_DIRS}"
        else:
            os.environ[key] = val


# ---------------------------------------------------------------- Python venv
def ensure_venv() -> None:
    """สร้าง/ซิงก์ .venv ด้วย uv — ใช้ pyproject.toml ของ repo ถ้ามี (dependency ถูก pin ไว้แล้ว)"""
    if Path("pyproject.toml").exists():
        _log("⏳ กำลังติดตั้งแพ็กเกจ Python ตาม pyproject.toml (uv sync)...")
        run_stream("set -e\nuv sync --quiet\n")
    else:
        _log("⏳ ไม่พบ pyproject.toml — สร้างโปรเจกต์ใหม่แล้วติดตั้ง dependency ชุดมาตรฐาน...")
        deps = " ".join(shlex.quote(d) for d in FALLBACK_DEPS)
        run_stream(
            "set -e\n"
            "if [ ! -f pyproject.toml ]; then uv init --vcs none --no-readme --quiet; fi\n"
            f"uv add --quiet {deps}\n"
        )
    run_quiet("uv export --format requirements-txt --no-hashes --no-emit-project --quiet -o requirements.txt || true")


def activate_venv() -> str:
    """ทำให้ kernel ปัจจุบัน import ของใน .venv ได้ทันที (ไม่ต้อง restart runtime)"""
    venv_path = Path(".venv").resolve()
    sites = glob.glob(str(venv_path / "lib" / "python*" / "site-packages"))
    if not sites:
        raise RuntimeError(f"ไม่พบ site-packages ใน {venv_path} — uv sync ล้มเหลว?")
    venv_site = sites[0]
    site.addsitedir(venv_site)
    while venv_site in sys.path:
        sys.path.remove(venv_site)
    sys.path.insert(0, venv_site)
    os.environ["VIRTUAL_ENV"] = str(venv_path)
    os.environ["PATH"] = f"{venv_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    importlib.invalidate_caches()
    return venv_site


def prepare_dirs() -> None:
    for d in (MODEL_DIR, f"{MODEL_DIR}/_drafts", STRATA_HOME):
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- entrypoint
def ensure_environment(skip_devshell: bool = False) -> dict:
    """เรียกครั้งเดียวใน notebook — ปลอดภัยถ้าเรียกซ้ำ

    คืน dict สรุปสถานะเพื่อเอาไป log/ดีบักได้
    """
    prepare_dirs()
    ensure_paths()
    write_nix_conf()
    install_nix()
    write_flake()
    if not skip_devshell:
        activate_devshell()
    link_tools()
    ensure_venv()
    venv_site = activate_venv()
    rprint(f"[bold green]✅ สภาพแวดล้อมพร้อม[/bold green] (nixpkgs {NIXPKGS_REV[:7]} • {venv_site})")
    return {"nixpkgs": NIXPKGS_REV, "venv": venv_site, "model_dir": MODEL_DIR}
