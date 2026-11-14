"""สร้างภาพด้วย ComfyUI + GGUF (คนละเอนจินกับ llama.cpp)

llama.cpp สร้างภาพไม่ได้ — `mmproj` มีไว้ให้โมเดล *อ่าน* ภาพ
โมดูลนี้จึงจัดการฝั่ง image generation: ดาวน์โหลดโมเดล GGUF/safetensors,
symlink เข้า `ComfyUI/models/...`, ติดตั้ง ComfyUI(+ComfyUI-GGUF) และสตาร์ท/หยุดเซิร์ฟเวอร์
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from rich import print as rprint

from .config import (
    COMFY_DIR,
    COMFY_MODEL_SUBDIRS,
    COMFY_PORT,
    IMAGE_PRESET_KEYS,
    MODEL_CATALOG,
    MODEL_DIR,
)
from .download import download_model_files
from .utils import _dir_bytes, _remember, run_stream

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_NODE_REPO = "https://github.com/city96/ComfyUI-GGUF"
GGUF_NODE_DIRNAME = "ComfyUI-GGUF"

_COMFY_PROC: subprocess.Popen | None = None

_COMFY_LOG: object | None = None  # ไฟล์ log ที่เปิดค้างไว้ตอนสตาร์ท (ปิดตอน stop)


# ---------------------------------------------------------------------------
# แคตตาล็อก / ไฟล์
# ---------------------------------------------------------------------------
def image_preset(key: str) -> dict:
    """entry ของโมเดลภาพ (kind="image") — อยู่ใน MODEL_CATALOG เดียวกับโมเดลข้อความ"""
    cfg = MODEL_CATALOG.get(key)
    if cfg is None or cfg.get("kind") != "image":
        raise ValueError(f"'{key}' ไม่ใช่โมเดลสร้างภาพ (มี: {', '.join(IMAGE_PRESET_KEYS)})")
    return cfg


def include_patterns(key: str) -> list[str]:
    """pattern สำหรับดาวน์โหลด = ชื่อไฟล์ในเรพอกดตรง ๆ (ไม่มี wildcard)"""
    return list(image_preset(key)["targets"])


def _comfy_models_root(comfy_dir: str | Path = COMFY_DIR) -> Path:
    return Path(comfy_dir) / "models"


def comfy_target_path(key: str, rel: str, comfy_dir: str | Path = COMFY_DIR) -> Path:
    """ตำแหน่งปลายทางใน ComfyUI/models/<subdir>/<ชื่อไฟล์>"""
    sub = image_preset(key)["targets"][rel]
    if sub not in COMFY_MODEL_SUBDIRS:
        raise ValueError(f"โฟลเดอร์ปลายทาง '{sub}' ไม่รองรับ (ต้องเป็น {COMFY_MODEL_SUBDIRS})")
    return _comfy_models_root(comfy_dir) / sub / Path(rel).name


def image_model_status(key: str, model_dir: str | Path | None = None) -> dict:
    """ดูว่าโหลดครบหรือยัง (ไม่แตะ network) — model_dir คือโฟลเดอร์รากที่มี <key>/ อยู่ข้างใน"""
    cfg = image_preset(key)
    base = Path(model_dir or MODEL_DIR) / key
    missing = [rel for rel in cfg["targets"] if not (base / rel).is_file()]
    gb = _dir_bytes(base) / (1024**3) if base.is_dir() else 0.0
    return {
        "key": key,
        "dir": str(base),
        "downloaded": not missing,
        "missing": missing,
        "gb": round(gb, 2),
        "approx_gb": cfg["approx_gb"],
    }


# ---------------------------------------------------------------------------
# ติดตั้งโมเดลภาพ + ผูกเข้า ComfyUI
# ---------------------------------------------------------------------------
def link_image_model(key: str, model_dir: str | Path | None = None, comfy_dir: str | Path = COMFY_DIR) -> dict[str, str]:
    """symlink ไฟล์โมเดลเข้าโฟลเดอร์ที่ ComfyUI มองเห็น (ทำซ้ำได้ ไม่พัง)"""
    cfg = image_preset(key)
    base = (Path(model_dir or MODEL_DIR) / key)
    linked: dict[str, str] = {}
    for rel in cfg["targets"]:
        src = (base / rel).resolve()
        if not src.is_file():
            continue
        dst = comfy_target_path(key, rel, comfy_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and Path(os.readlink(dst)) == src:
                linked[rel] = str(dst)
                continue
            dst.unlink()
        os.symlink(src, dst)
        linked[rel] = str(dst)
    return linked


def install_image_model(key: str, model_dir: str | Path | None = None, force: bool = False, **_: object) -> dict:
    """ดาวน์โหลดไฟล์ของโมเดลภาพลง <model_dir>/<key> (เหมือนโมเดลอื่น) แล้ว symlink เข้า ComfyUI/models"""
    cfg = image_preset(key)
    base = Path(model_dir or MODEL_DIR) / key
    base.mkdir(parents=True, exist_ok=True)
    rprint(f"[bold magenta]🎨 โมเดลสร้างภาพ:[/bold magenta] [cyan]{key}[/cyan] -> {cfg['repo_id']}")
    res = download_model_files(
        include=include_patterns(key), exclude=None, model_dir=str(base), repo_id=cfg["repo_id"], force=force
    )
    linked = link_image_model(key, base.parent)
    st = image_model_status(key, base.parent)
    if st["missing"]:
        raise FileNotFoundError(f"ยังขาดไฟล์: {', '.join(st['missing'])} (ลองดาวน์โหลดใหม่หรือตรวจชื่อไฟล์ในเรพอ)")
    rprint(f"[bold green]✅ พร้อมสร้างภาพ[/bold green] ({st['gb']:.2f} GB • ผูกเข้า ComfyUI {len(linked)} ไฟล์)")
    return {**res, **st, "linked": linked}


# ---------------------------------------------------------------------------
# ComfyUI runtime
# ---------------------------------------------------------------------------
def comfyui_command(port: int = COMFY_PORT, lowvram: bool = True, comfy_dir: str | Path = COMFY_DIR) -> list[str]:
    """คำสั่งสตาร์ท ComfyUI — คั่นด้วย --listen 0.0.0.0 เพื่อให้ Colab proxy เข้าถึงได้"""
    cmd = [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", str(port), "--preview-method", "none"]
    if lowvram:
        cmd.append("--lowvram")
    return cmd


TARBALL_BRANCHES = ("master", "main")  # ลองทีละชื่อ branch ตอนโหลด tarball


def _say(text: str) -> None:
    """เขียนทั้งลง log สดของ UI และ stdout"""
    _remember(text)
    print(text)


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 900) -> None:
    """รัน git แบบเก็บ stderr ไว้ในข้อความ error — clone ล้มจะได้รู้สาเหตุจริงไม่ใช่แค่ exit 128"""
    cmd = ["git", *args]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"รัน `{' '.join(cmd)}` ไม่ได้: {e}") from e
    for line in ((p.stdout or "") + (p.stderr or "")).splitlines()[-15:]:
        _say(line)
    if p.returncode != 0:
        raw = (p.stderr or p.stdout or "").strip().splitlines()
        tail = " | ".join(x.strip() for x in raw[-3:] if x.strip()) or "(git ไม่ได้บอกสาเหตุ)"
        raise RuntimeError(f"`{' '.join(cmd)}` ล้มเหลว (exit {p.returncode}): {tail}")


def _git(args: list[str], cwd: Path | None = None) -> None:
    """alias เดิม (เก็บไว้ให้โค้ด/เทสต์เก่าเรียกได้)"""
    _run_git(args, cwd=cwd)


def _repo_slug(repo_url: str) -> str:
    """https://github.com/owner/name(.git) -> owner/name"""
    s = repo_url.rstrip("/").removesuffix(".git")
    return s.split("github.com/")[-1] if "github.com/" in s else s


def _extract_tarball(data: bytes, dest: Path) -> None:
    """แตก tar.gz ของ GitHub (ตัดโฟลเดอร์ชั้นบนสุดออก) ลง dest แบบปลอดภัย"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue  # entry ของโฟลเดอร์บนสุด / ไฟล์เปล่า
            m.name = parts[1]
            members.append(m)
        try:
            tar.extractall(dest, members=members, filter="data")  # py>=3.12
        except TypeError:  # python เก่าไม่มี filter=
            tar.extractall(dest, members=members)


def _fetch_tarball(repo_url: str, dest: Path) -> None:
    """ดาวน์โหลดซอร์สเป็น tarball จาก codeload (ใช้เมื่อ git clone ไม่ผ่าน)"""
    slug = _repo_slug(repo_url)
    last = ""
    for br in TARBALL_BRANCHES:
        url = f"https://codeload.github.com/{slug}/tar.gz/refs/heads/{br}"
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001 - รายงานสาเหตุสุดท้ายให้ผู้ใช้เห็น
            last = f"{br}: {e}"
            continue
        _say(f"⬇️ ดาวน์โหลด tarball {slug}@{br} (~{len(data) / 1024**2:.0f} MB)")
        _extract_tarball(data, dest)
        return
    raise RuntimeError(f"ดาวน์โหลด {slug} เป็น tarball ไม่สำเร็จ ({last})")


def _clear_incomplete(dest: Path) -> None:
    """ล้างโฟลเดอร์ที่โคลนค้าง/พัง (git clone ซ้ำจะได้ไม่ error 128 'not an empty directory')"""
    dest = Path(dest)
    if not dest.exists():
        return
    try:
        empty = not any(dest.iterdir())
    except OSError:
        empty = False
    if empty:
        dest.rmdir()
        return
    rprint(f"[yellow]🧹 ล้างโฟลเดอร์เดิมที่ไม่สมบูรณ์: {dest}[/yellow]")
    shutil.rmtree(dest, ignore_errors=True)


def _clone_or_tarball(repo_url: str, dest: Path, marker: str = "main.py") -> None:
    """clone แบบ depth 1 → ถ้าไม่ผ่านให้ fallback เป็น tarball อัตโนมัติ"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run_git(["clone", "--depth", "1", repo_url, str(dest)])
        if (dest / marker).exists():
            return
        rprint(f"[yellow]⚠️ clone เสร็จแต่ไม่พบ {marker} — ลองดาวน์โหลด tarball แทน[/yellow]")
    except RuntimeError as e:
        rprint(f"[yellow]⚠️ git clone ไม่สำเร็จ: {e}[/yellow]")
        rprint("[yellow]→ ลองดาวน์โหลดเป็น tarball จาก codeload แทน[/yellow]")
    _clear_incomplete(dest)
    _fetch_tarball(repo_url, dest)
    if not (dest / marker).exists():
        raise RuntimeError(f"ติดตั้ง {_repo_slug(repo_url)} แล้วแต่ไม่พบ {marker} ใน {dest}")


def ensure_comfyui(pull: bool = True, comfy_dir: str | Path = COMFY_DIR) -> Path:
    """โคลน/อัปเดต ComfyUI + custom node GGUF แล้วติดตั้ง dependency (ข้ามถ้ามีอยู่แล้ว)"""
    root = Path(comfy_dir)
    if not (root / "main.py").is_file():
        _clear_incomplete(root)
        _clone_or_tarball(COMFY_REPO, root, marker="main.py")
    elif pull:
        try:
            _run_git(["pull", "--ff-only"], cwd=root)
        except RuntimeError as e:
            rprint(f"[yellow]⚠️ อัปเดต ComfyUI ไม่ได้ (ใช้ของเดิมต่อ): {e}[/yellow]")
    node = root / "custom_nodes" / GGUF_NODE_DIRNAME
    if not (node / "nodes.py").is_file():
        _clear_incomplete(node)
        _clone_or_tarball(GGUF_NODE_REPO, node, marker="nodes.py")
    req = root / "requirements.txt"
    try:
        if req.is_file():
            run_stream(f"{sys.executable} -m pip install -q -r {req}")
        run_stream(f"{sys.executable} -m pip install -q gguf")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ติดตั้ง dependency ของ ComfyUI ไม่สำเร็จ (exit {e.returncode}) — "
            "ลองรัน `%s -m pip install -r %s` เองแล้วดู error แรก" % (sys.executable, req)
        ) from e
    rprint(f"[bold green]✅ ComfyUI พร้อม[/bold green] ที่ {root}")
    return root


def comfyui_online(port: int = COMFY_PORT) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            return r.status < 500
    except (urllib.error.URLError, OSError):
        return False


def comfyui_public_url(port: int = COMFY_PORT) -> str:
    """URL ที่เปิดจากเบราว์เซอร์ได้ — บน Colab ต้องผ่าน proxyPort"""
    try:  # ใน Colab เท่านั้น
        from google.colab.output import eval_js  # type: ignore

        return str(eval_js(f"google.colab.kernel.proxyPort({int(port)})"))
    except Exception:
        return f"http://127.0.0.1:{port}"


def start_comfyui(port: int = COMFY_PORT, lowvram: bool = True, wait_s: int = 180, comfy_dir: str | Path = COMFY_DIR) -> str:
    """สตาร์ท ComfyUI (ถ้ายังไม่รัน) แล้วคืน URL สำหรับเปิดในเบราว์เซอร์"""
    global _COMFY_PROC
    if comfyui_online(port):
        url = comfyui_public_url(port)
        rprint(f"[bold green]ℹ️ ComfyUI รันอยู่แล้ว:[/bold green] {url}")
        return url
    root = Path(comfy_dir)
    if not (root / "main.py").is_file():
        raise FileNotFoundError(f"ยังไม่มี ComfyUI ที่ {root} — เรียก ensure_comfyui() ก่อน")
    global _COMFY_LOG
    log = root / "comfyui.log"
    _COMFY_LOG = log.open("w", encoding="utf-8")
    _COMFY_PROC = subprocess.Popen(
        comfyui_command(port, lowvram=lowvram, comfy_dir=root),
        cwd=str(root),
        stdout=_COMFY_LOG,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if comfyui_online(port):
            url = comfyui_public_url(port)
            rprint(f"[bold green]🎨 ComfyUI พร้อมสร้างภาพ:[/bold green] {url}")
            return url
        if _COMFY_PROC.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-8:] if log.is_file() else []
            raise RuntimeError("ComfyUI ออกก่อนพร้อมใช้งาน:\n" + "\n".join(tail))
        time.sleep(2)
    raise TimeoutError(f"ComfyUI ไม่ขึ้นภายใน {wait_s}s — ดู log ที่ {log}")


def _close_comfy_log() -> None:
    global _COMFY_LOG
    if _COMFY_LOG is not None:
        with contextlib.suppress(Exception):
            _COMFY_LOG.close()  # type: ignore[attr-defined]
        _COMFY_LOG = None


def stop_comfyui() -> bool:
    global _COMFY_PROC
    if _COMFY_PROC is not None and _COMFY_PROC.poll() is None:
        _COMFY_PROC.terminate()
        try:
            _COMFY_PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _COMFY_PROC.kill()
        _COMFY_PROC = None
        _close_comfy_log()
        rprint("[bold yellow]⏹ หยุด ComfyUI แล้ว[/bold yellow]")
        return True
    _COMFY_PROC = None
    if shutil.which("pkill"):
        return False
    return False


def comfyui_status(port: int = COMFY_PORT) -> dict:
    return {
        "online": comfyui_online(port),
        "port": port,
        "url": comfyui_public_url(port) if comfyui_online(port) else "",
        "installed": (Path(COMFY_DIR) / "main.py").is_file(),
        "models_gb": round(_dir_bytes(Path(COMFY_DIR) / "models") / (1024**3), 2) if Path(COMFY_DIR).is_dir() else 0.0,
    }


__all__ = [
    "_extract_tarball",
    "_fetch_tarball",
    "_run_git",
    "comfy_target_path",
    "comfyui_command",
    "comfyui_online",
    "comfyui_public_url",
    "comfyui_status",
    "ensure_comfyui",
    "image_model_status",
    "image_preset",
    "include_patterns",
    "install_image_model",
    "link_image_model",
    "start_comfyui",
    "stop_comfyui",
]

if __name__ == "__main__":  # pragma: no cover - CLI เร็ว ๆ
    for k in IMAGE_PRESET_KEYS:
        st = image_model_status(k)
        print(f"{k}: {st['gb']}/{st['approx_gb']} GB • ครบ={st['downloaded']}")
