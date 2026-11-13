"""ยูทิลิตี้พื้นฐาน: รันคำสั่ง shell อย่างปลอดภัย, วัดขนาดโฟลเดอร์แบบ cache, แตก archive อย่างปลอดภัย."""
from __future__ import annotations

import importlib
import re
import shlex
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests


def run_quiet(cmd: str):
    p = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(p.returncode, cmd)

def run_stream(cmd: str):
    p = subprocess.Popen(["/bin/bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if p.stdout:
        for line in p.stdout:
            print(line, end="")
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)

def uv_add(*packages: str):
    if not packages:
        return
    for pkg in packages:
        if not re.fullmatch(r"[A-Za-z0-9_.\-\[\]=<>!,~*+/]+", pkg):
            raise ValueError(f"ชื่อแพ็กเกจไม่ถูกต้อง: {pkg!r}")
    s = " ".join(shlex.quote(p) for p in packages)
    run_quiet(f"uv add --quiet {s} && uv export --format requirements-txt --no-hashes --no-emit-project --quiet -o requirements.txt")
    importlib.invalidate_caches()
    print(f"✅ ติดตั้ง {s} เรียบร้อย!")

def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if not p.is_symlink() and p.is_file())

_DIR_CACHE: dict[str, tuple[float, int, int]] = {}

_DIR_CACHE_TTL = 30.0

def _dir_stats(path: Path, ttl: float = _DIR_CACHE_TTL) -> dict:
    """ขนาดโฟลเดอร์ + จำนวนไฟล์ .gguf แบบ cache — กัน UI เดินดิสก์ใหม่ทุกครั้งที่กดปุ่ม/ส่งแชท"""
    key = str(Path(path))
    now = time.monotonic()
    hit = _DIR_CACHE.get(key)
    if hit and ttl > 0 and (now - hit[0]) < ttl:
        return {"bytes": hit[1], "gguf": hit[2], "cached": True}
    p = Path(path)
    total = gguf = 0
    if p.is_file():
        total = p.stat().st_size
    elif p.exists():
        for f in p.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                continue
            if f.suffix.lower() == ".gguf":
                gguf += 1
    _DIR_CACHE[key] = (now, total, gguf)
    return {"bytes": total, "gguf": gguf, "cached": False}

def _invalidate_dir_cache(prefix: Path | None = None):
    if prefix is None:
        _DIR_CACHE.clear()
        return
    pref = str(Path(prefix)).rstrip("/")
    for k in [k for k in _DIR_CACHE if k == pref or k.startswith(pref + "/")]:
        _DIR_CACHE.pop(k, None)

def _fmt_bytes(n: int) -> str:
    return f"{n / (1024**3):.2f} GB" if n >= 1024**3 else (f"{n / (1024**2):.2f} MB" if n >= 1024**2 else f"{n / 1024:.1f} KB")

def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    members = []
    for m in tf.getmembers():
        target = (root / m.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"ไฟล์ใน tar ชี้นอกโฟลเดอร์ปลายทาง: {m.name}")
        if m.isfile() or m.isdir():
            members.append(m)
    if sys.version_info >= (3, 12):
        tf.extractall(root, filter="data")
    else:
        tf.extractall(root, members=members)

def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for info in zf.infolist():
        target = (root / info.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"ไฟล์ใน zip ชี้นอกโฟลเดอร์ปลายทาง: {info.filename}")
    zf.extractall(root)

def _download_and_extract(url: str, name: str, dest: Path):
    arc = dest / name
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(arc, "wb") as f:
            for ch in r.iter_content(chunk_size=1024 * 1024):
                if ch:
                    f.write(ch)
    try:
        if name.endswith(".tar.gz"):
            with tarfile.open(arc, "r:gz") as tf:
                _safe_extract_tar(tf, dest)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(arc, "r") as zf:
                _safe_extract_zip(zf, dest)
    finally:
        arc.unlink(missing_ok=True)
