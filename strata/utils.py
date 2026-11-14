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
from collections import deque
from pathlib import Path

import requests

LOG_RING: deque = deque(maxlen=600)  # บรรทัด log ล่าสุด (ให้ UI แสดงแบบเรียลไทม์)


def recent_log(n: int = 12) -> list[str]:
    """n บรรทัดล่าสุดจากคำสั่งที่รันผ่าน run_quiet/run_stream"""
    return list(LOG_RING)[-int(n):]


def _remember(line: str) -> None:
    LOG_RING.append(str(line).rstrip("\n"))


def run_quiet(cmd: str):
    p = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(p.returncode, cmd)

def run_stream(cmd: str, cwd: str | Path | None = None):
    """รันคำสั่งแบบสตรีม log — `cwd` ใช้กำหนดโฟลเดอร์ทำงาน (เช่น git ใน repo อื่น)"""
    where = f" (ใน {cwd})" if cwd else ""
    _remember(f"$ {cmd}{where}")
    p = subprocess.Popen(
        ["/bin/bash", "-c", cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    if p.stdout:
        for line in p.stdout:
            _remember(line)
            print(line, end="")
    p.wait()
    _remember(f"(exit {p.returncode})")
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

# ---------------------------------------------------------------------------
# Network bandwidth (DL/UL) — อ่านตัวนับไบต์จาก /proc/net/dev แล้วคำนวณส่วนต่าง
# ---------------------------------------------------------------------------
def _read_net_counters() -> tuple[int, int]:
    """คืน (rx_bytes, tx_bytes) รวมทุก interface ยกเว้น loopback; (0, 0) ถ้าอ่านไม่ได้"""
    try:
        lines = Path("/proc/net/dev").read_text(encoding="utf-8", errors="ignore").splitlines()[2:]
    except OSError:
        return 0, 0
    rx = tx = 0
    for line in lines:
        iface, sep, data = line.partition(":")
        if not sep or iface.strip() == "lo":
            continue
        parts = data.split()
        if len(parts) < 9:
            continue
        try:
            rx += int(parts[0])
            tx += int(parts[8])
        except ValueError:
            continue
    return rx, tx


class NetworkMonitor:
    """คำนวณความเร็วเน็ต (bytes/s) จากส่วนต่างของตัวนับ — เรียกถี่แค่ไหนก็ได้ ไม่ต้องมี thread"""

    def __init__(self, reader=_read_net_counters, clock=time.monotonic):
        self._read = reader
        self._clock = clock
        self._prev: tuple[float, int, int] | None = None

    def sample(self) -> dict:
        now = self._clock()
        try:
            rx, tx = self._read()
        except Exception:
            rx = tx = 0
        prev = self._prev
        self._prev = (now, rx, tx)
        down = up = 0.0
        if prev is not None:
            dt = now - prev[0]
            if dt > 0 and rx >= prev[1] and tx >= prev[2]:  # กันตัวนับรีเซ็ต/reload
                down = (rx - prev[1]) / dt
                up = (tx - prev[2]) / dt
        return {
            "down_bps": down,
            "up_bps": up,
            "rx_bytes": rx,
            "tx_bytes": tx,
            "ok": prev is not None,
        }

    def reset(self) -> None:
        self._prev = None


_NET_MONITOR = NetworkMonitor()


def net_bandwidth() -> dict:
    """ความเร็วดาวน์โหลด/อัปโหลดปัจจุบัน (bytes/s) พร้อมยอดรวมสะสม"""
    return _NET_MONITOR.sample()


def _fmt_rate(bps: float) -> str:
    """จัดรูปแบบความเร็ว: `0 B/s`, `812 KB/s`, `4.21 MB/s`"""
    bps = max(0.0, float(bps))
    if bps < 1024:
        return f"{bps:.0f} B/s"
    if bps < 1024**2:
        return f"{bps / 1024:.0f} KB/s"
    if bps < 1024**3:
        return f"{bps / 1024**2:.2f} MB/s"
    return f"{bps / 1024**3:.2f} GB/s"


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
