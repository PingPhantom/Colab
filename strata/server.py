"""จัดการ llama.cpp server: ติดตั้งไบนารี, เลือก quant, สตาร์ท/หยุด และ speculative decoding."""
from __future__ import annotations

import functools
import os
import re
import shutil
import signal
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from rich import print as rprint
from rich.console import Console

from .config import (
    ALL_PRESET_KEYS,
    DEFAULT_PORT,
    LLAMA_LOG_FILE,
    MODEL_CATALOG,
    MODEL_DIR,
    NVIDIA_LIB_DIRS,
    SPEC_STATE,
    S,
)
from .utils import _download_and_extract

if TYPE_CHECKING:  # openai ถูก import ตอนเรียกใช้จริง (ให้ import โมดูลนี้ได้เร็ว)
    from openai import AsyncOpenAI, OpenAI

LLAMA_SERVER_PROC: subprocess.Popen | None = None

NIX_STORE = "/nix/store/"
LLAMA_CPP_DIR = "/content/.llama-cpp"
WRAPPER_REV = "# strata-wrapper-v2"  # v2 = กรอง /nix/store ออกจาก LD_LIBRARY_PATH (glibc ของ Nix ชนกับระบบ)


def sanitize_ld_library_path(value: str | None) -> str:
    """ตัด path ของ Nix ออกจาก LD_LIBRARY_PATH

    ไบนารี prebuilt (llama-server) ต้องใช้ glibc ของระบบ — ถ้าไปหยิบ libc/libdl ของ Nix
    จะได้ error `version 'GLIBC_ABI_DT_X86_64_PLT' not found`
    """
    keep: list = []
    for part in (value or "").split(os.pathsep):
        p = part.strip()
        if not p or p.startswith(NIX_STORE) or p in keep:
            continue
        keep.append(p)
    return os.pathsep.join(keep)


def server_env(base: dict | None = None) -> dict:
    """env สำหรับรันไบนารี prebuilt: ไม่มี lib ของ Nix และไม่มี LD_PRELOAD ที่ชี้เข้า store"""
    env = dict(os.environ if base is None else base)
    env["LD_LIBRARY_PATH"] = sanitize_ld_library_path(env.get("LD_LIBRARY_PATH"))
    if NIX_STORE in (env.get("LD_PRELOAD") or ""):
        env.pop("LD_PRELOAD", None)
    return env


def start_error_hint(tail: str) -> str:
    """แนบคำใบ้ให้แก้ได้เองเมื่อ server ตายเพราะ lib ไม่เข้ากัน"""
    text = tail or ""
    if "GLIBC" in text:
        return (
            "\n💡 ไบนารี prebuilt หยิบ glibc ของ Nix มาผสมกับของระบบ — โค้ดนี้ตัด path `/nix/store` "
            "ออกจาก LD_LIBRARY_PATH ให้แล้ว ถ้ายังเจอให้ลบ /usr/local/bin/llama-server แล้วกดติดตั้งใหม่ "
            "(wrapper จะถูกสร้างพร้อมตัวกรอง)"
        )
    if "CUDA" in text or "cuda" in text:
        return "\n💡 ตรวจว่า Colab เปิด GPU อยู่ (Runtime → Change runtime type → GPU) และไดรเวอร์ CUDA ยังทำงาน"
    return ""


def check_disk_space(target_dir: str = "/content", required_gb: float | None = None) -> dict:
    cp = Path(target_dir)
    while not cp.exists() and cp != cp.parent:
        cp = cp.parent
    u = shutil.disk_usage(cp)
    t_gb, u_gb, f_gb = u.total / (1024**3), u.used / (1024**3), u.free / (1024**3)
    col = "green" if (required_gb is None or f_gb >= required_gb) else "bold red"
    extra = "" if required_gb is None else (f" — [bold green]✅ พอ ({required_gb:.1f}GB)[/bold green]" if f_gb >= required_gb else f" — [bold red]❌ ขาด {required_gb - f_gb:.1f}GB[/bold red]")
    Console().print(f"💾 [bold]Disk ({cp}):[/bold] ว่าง [{col}]{f_gb:.2f} GB[/{col}] / {t_gb:.2f} GB{extra}")
    return {"total_gb": t_gb, "used_gb": u_gb, "free_gb": f_gb}

def check_gpu_and_recommend(gb_per_layer: float = 0.5, kv_headroom_gb: float = 2.0, verbose: bool = True) -> dict:
    cores = os.cpu_count() or 2
    threads = max(1, cores - 1) if cores > 2 else cores
    ram_t, ram_a = 0.0, 0.0
    if Path("/proc/meminfo").exists():
        for ln in Path("/proc/meminfo").read_text().splitlines():
            pts = ln.split()
            if len(pts) >= 2 and ln.startswith("MemTotal:"):
                _, kb, *_ = pts; ram_t = int(kb) / (1024**2)
            elif len(pts) >= 2 and ln.startswith("MemAvailable:"):
                _, kb, *_ = pts; ram_a = int(kb) / (1024**2)
    smi = next((c for c in ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi", shutil.which("nvidia-smi")) if c and Path(c).exists()), None)
    gpus = []
    if smi:
        try:
            out = subprocess.check_output([smi, "--query-gpu=index,name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL).strip()
            for ln in out.splitlines():
                if ln.strip():
                    i, nm, tot, fr, cc = [x.strip() for x in ln.split(",")]
                    gpus.append({"index": int(i), "name": nm, "vram_total_gb": float(tot) / 1024, "vram_free_gb": float(fr) / 1024, "compute_cap": cc})
        except Exception:
            pass
    if not gpus:
        return {"has_gpu": False, "gpu_name": "CPU Only", "vram_total_gb": 0.0, "vram_free_gb": 0.0, "ram_total_gb": ram_t, "ram_avail_gb": ram_a, "recommended_gpu_layers": 0, "recommended_ctx_size": 4096, "recommended_threads": threads}
    tv, fv = sum(g["vram_total_gb"] for g in gpus), sum(g["vram_free_gb"] for g in gpus)
    names = ", ".join(f"{g['name']} ({g['vram_total_gb']:.0f}GB)" for g in gpus)
    layers = max(1, int(max(0.0, fv - kv_headroom_gb) / gb_per_layer))
    ctx = 16384 if tv >= 35 else (8192 if tv >= 20 else 4096)
    if verbose:
        Console().print(f"🎮 [bold green]GPU:[/bold green] {names} | VRAM ว่าง {fv:.2f}/{tv:.2f} GB | RAM ว่าง {ram_a:.1f} GB")
    return {"has_gpu": True, "gpu_name": names, "vram_total_gb": tv, "vram_free_gb": fv, "ram_total_gb": ram_t, "ram_avail_gb": ram_a, "recommended_gpu_layers": layers, "recommended_ctx_size": ctx, "recommended_threads": threads}

def _write_wrapper(srv_bin: Path, dest: Path, wrapper: Path = Path("/usr/local/bin/llama-server")) -> Path:
    """สร้าง wrapper ที่ตัด path ของ Nix ออกจาก LD_LIBRARY_PATH ก่อนรันไบนารี prebuilt"""
    lib_dirs = os.pathsep.join(sorted({str(p.parent) for p in dest.rglob("*.so*")}))
    base = [d for d in (NVIDIA_LIB_DIRS, lib_dirs, str(srv_bin.parent)) if d]
    body = (
        "#!/usr/bin/env bash\n"
        f"{WRAPPER_REV}\n"
        "# ลบ lib ของ Nix ออกจาก LD_LIBRARY_PATH (glibc ของ Nix ไม่เข้ากับไบนารี prebuilt)\n"
        "unset LD_PRELOAD\n"
        '_keep=""\n'
        'while IFS= read -r _p; do\n'
        '  case "$_p" in /nix/store/*) continue;; esac\n'
        '  _keep="${_keep:+$_keep:}$_p"\n'
        'done < <(printf "%s\\n" "${LD_LIBRARY_PATH:-}" | tr ":" "\\n")\n'
        f'export LD_LIBRARY_PATH="{os.pathsep.join(base)}${{_keep:+:$_keep}}"\n'
        f'exec "{srv_bin}" "$@"\n'
    )
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.unlink(missing_ok=True)
    wrapper.write_text(body, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def install_llama_server(install_dir: str = LLAMA_CPP_DIR) -> str:
    wrapper = Path("/usr/local/bin/llama-server")
    if wrapper.exists():
        wtxt = wrapper.read_text(encoding="utf-8", errors="ignore")
        if ("\\" + S) in wtxt:
            wrapper.write_text(wtxt.replace("\\" + S, S), encoding="utf-8"); wrapper.chmod(0o755)
        old_bins = sorted(q for q in Path(install_dir).rglob("llama-server") if q.is_file())
        if WRAPPER_REV not in wtxt and old_bins:  # wrapper เก่า (ไม่มีตัวกรอง /nix/store) -> เขียนใหม่
            rprint("[dim]🔧 อัปเดต wrapper llama-server (ตัด lib ของ Nix ออกจาก LD_LIBRARY_PATH)[/dim]")
            return str(_write_wrapper(old_bins[0], Path(install_dir), wrapper))
        if subprocess.run([str(wrapper), "--version"], capture_output=True).returncode == 0:
            return str(wrapper)
    rprint("[bold cyan]⏳ กำลังติดตั้ง Prebuilt llama-server (CUDA 12)...[/bold cyan]")
    hw = check_gpu_and_recommend(verbose=False)
    gapi = requests.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10", timeout=20, headers={"Accept": "application/vnd.github+json"})
    if gapi.status_code != 200:
        raise RuntimeError(f"ดึงรายการ release ของ llama.cpp ไม่สำเร็จ (HTTP {gapi.status_code}) — อาจติด rate limit ของ GitHub ลองรันใหม่ภายหลัง")
    rels = gapi.json()
    _pin = os.environ.get("LLAMA_CPP_TAG", "").strip()
    if _pin:
        rels = [r for r in rels if r.get("tag_name") == _pin] or rels
        rprint(f"[dim]📌 ใช้ release ที่กำหนด: {_pin}[/dim]")
    main_asset, cudart_asset = None, None
    prefs = ["bin-ubuntu-cuda-12", "bin-ubuntu-vulkan-x64", "bin-ubuntu-x64"] if hw["has_gpu"] else ["bin-ubuntu-x64"]
    for pat in prefs:
        for rel in rels:
            assets = rel.get("assets", [])
            for a in assets:
                nm = a.get("name", "")
                if nm.startswith("llama-") and pat in nm and "x64" in nm and (nm.endswith(".tar.gz") or nm.endswith(".zip")):
                    main_asset = a
                    if "cuda-12" in pat:
                        cudart_asset = next((c for c in assets if c.get("name", "").startswith("cudart-") and "ubuntu-cuda-12" in c.get("name", "") and "x64" in c.get("name", "")), None)
                    break
            if main_asset:
                break
        if main_asset:
            break
    if not main_asset:
        raise RuntimeError("ไม่พบไฟล์ Prebuilt llama-server")
    dest = Path(install_dir); shutil.rmtree(dest, ignore_errors=True); dest.mkdir(parents=True, exist_ok=True)
    if cudart_asset:
        _download_and_extract(cudart_asset["browser_download_url"], cudart_asset["name"], dest)
    _download_and_extract(main_asset["browser_download_url"], main_asset["name"], dest)
    srv_bins = [p for p in dest.rglob("llama-server") if p.is_file()]
    if not srv_bins:
        raise RuntimeError("ไม่พบไบนารี llama-server")
    srv_bin = next(iter(srv_bins)); srv_bin.chmod(0o755)
    _write_wrapper(srv_bin, dest, wrapper)
    rprint(f"[bold green]✅ ติดตั้ง llama-server สำเร็จ ({main_asset['name']})![/bold green]")
    return str(wrapper)

def get_openai_client(port: int = DEFAULT_PORT, api_key: str = "strata", async_mode: bool = False, timeout: float = 300.0) -> OpenAI | AsyncOpenAI:
    url = f"http://127.0.0.1:{port}/v1"
    from openai import AsyncOpenAI, OpenAI

    return AsyncOpenAI(base_url=url, api_key=api_key, timeout=timeout) if async_mode else OpenAI(base_url=url, api_key=api_key, timeout=timeout)

_SRV_CACHE: dict[int, tuple[float, tuple[bool, str]]] = {}

_SRV_CACHE_TTL = 1.5

def _probe_server(port: int) -> tuple[bool, str]:
    try:
        if requests.get(f"http://127.0.0.1:{port}/health", timeout=1.5).status_code != 200:
            return False, "loading"
        rm = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=1.5)
        if rm.status_code == 200 and "json" in rm.headers.get("content-type", "").lower():
            d = rm.json()
            if isinstance(d, dict) and isinstance(d.get("data"), list) and d["data"]:
                return True, str(next(iter(d["data"])).get("id", "default"))
    except Exception:
        pass
    return False, "offline"

def _check_server_online(port: int = DEFAULT_PORT, ttl: float = _SRV_CACHE_TTL) -> tuple[bool, str]:
    """เช็กสุขภาพ server แบบ cache สั้น ๆ (UI เรียกบ่อย แต่ไม่ควรยิง HTTP ซ้ำทุกครั้ง) — ttl=0 เพื่อบังคับเช็กใหม่"""
    now = time.monotonic()
    hit = _SRV_CACHE.get(port)
    if hit and ttl > 0 and (now - hit[0]) < ttl:
        return hit[1]
    res = _probe_server(port)
    _SRV_CACHE[port] = (now, res)
    return res

def _invalidate_server_cache(port: int | None = None):
    if port is None:
        _SRV_CACHE.clear()
    else:
        _SRV_CACHE.pop(port, None)

_GGUF_LAYERS: dict[str, int] = {}

def _gguf_read_value(f, vtype: int):
    scalars = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if vtype in scalars:
        fmt = scalars[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 8:
        ln = struct.unpack("<Q", f.read(8))[0]
        return f.read(ln).decode("utf-8", errors="ignore")
    if vtype == 9:
        etype, cnt = struct.unpack("<IQ", f.read(12))
        for _ in range(min(cnt, 100000)):
            _gguf_read_value(f, etype)
        return None
    raise ValueError(f"GGUF metadata type {vtype} ไม่รู้จัก")

def _gguf_layer_count(path: str | Path) -> int:
    """อ่าน <arch>.block_count จากหัวไฟล์ GGUF (คืน 0 ถ้าอ่านไม่ได้) เพื่อคำนวณ --n-gpu-layers ให้ตรงจริง"""
    key = str(path)
    if key in _GGUF_LAYERS:
        return _GGUF_LAYERS[key]
    n_layers = 0
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return 0
            ver = struct.unpack("<I", f.read(4))[0]
            if ver < 2 or ver > 3:
                return 0
            struct.unpack("<Q", f.read(8))[0]  # tensor_count
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                klen = struct.unpack("<Q", f.read(8))[0]
                if klen > 1_000_000:
                    return 0
                kname = f.read(klen).decode("utf-8", errors="ignore")
                vtype = struct.unpack("<I", f.read(4))[0]
                val = _gguf_read_value(f, vtype)
                if kname.endswith(".block_count") and isinstance(val, int) and val > 0:
                    n_layers = val
                    break
    except (OSError, struct.error, ValueError):
        return 0
    if n_layers:
        _GGUF_LAYERS[key] = n_layers
    return n_layers

def _pick_main_gguf(model_dir: str, hw: dict | None = None, prefer: str | None = None, verbose: bool = True) -> tuple[Path, Path | None, float]:
    """เลือก GGUF หลัก: prefer (ถ้าระบุ) > quant ใหญ่สุดที่โหลดไหว > quant เล็กสุด; shard ไม่ครบ = error ชัดเจน"""
    root = Path(model_dir)
    all_ggufs = [p for p in sorted(root.rglob("*.gguf")) if "_drafts" not in p.parts]
    if not all_ggufs:
        raise FileNotFoundError(f"ไม่พบไฟล์ .gguf ใน {model_dir}")
    mmprojs = [p for p in all_ggufs if "mmproj" in p.name.lower()]
    skip = ("mmproj", "imatrix", "-draft", "draft-")
    mains = [p for p in all_ggufs if not any(k in p.name.lower() for k in skip)]
    if not mains:
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลหลักใน {model_dir}")
    s_re = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf\Z", re.IGNORECASE)
    cands: list[tuple[float, Path]] = []
    broken: list[str] = []
    seen: set[Path] = set()
    for p in mains:
        m = s_re.match(p.name)
        if m:
            pfx, idx, tot_s = m.groups()
            if idx != "00001":
                continue
            shards = [p.parent / f"{pfx}-{i:05d}-of-{tot_s}.gguf" for i in range(1, int(tot_s) + 1)]
            missing = [s.name for s in shards if not s.exists()]
            if missing:
                broken.append(f"{p.name} (ขาด {len(missing)} shard)")
                continue
            if shards[0] in seen:
                continue
            seen.add(shards[0])
            cands.append((sum(s.stat().st_size for s in shards) / (1024**3), p))
        elif "-of-" not in p.name.lower():
            if p in seen:
                continue
            seen.add(p)
            cands.append((p.stat().st_size / (1024**3), p))
    if broken and verbose:
        rprint(f"[bold yellow]⚠️ ข้ามไฟล์ที่โหลดไม่ครบ: {', '.join(broken)}[/bold yellow]")
    if not cands:
        raise FileNotFoundError(f"ไฟล์ .gguf ใน {model_dir} ไม่สมบูรณ์ (shard ขาด) — โหลดใหม่ก่อน: {', '.join(broken) or 'ไม่พบไฟล์หลัก'}")
    mm_pick = next(iter(mmprojs), None)
    if len(cands) > 1 and verbose:
        rprint("[dim]📦 quant ที่พบ: " + ", ".join(f"{p.name}={sz:.2f}GB" for sz, p in sorted(cands, key=lambda c: c[0])) + "[/dim]")
    if prefer:
        pr = prefer.lower()
        hit = next(((sz, p) for sz, p in cands if pr in p.name.lower()), None)
        if hit:
            if verbose:
                rprint(f"[cyan]🎯 ใช้ quant ตามที่ระบุ: {hit[1].name} ({hit[0]:.2f} GB)[/cyan]")
            return hit[1], mm_pick, hit[0]
        if verbose:
            rprint(f"[yellow]⚠️ ไม่พบ quant ที่ตรงกับ '{prefer}' — เลือกอัตโนมัติแทน[/yellow]")
    hw = hw or check_gpu_and_recommend(verbose=False)
    budget = max(0.5, hw["vram_free_gb"] - 2.2) if hw["has_gpu"] else max(1.0, hw["ram_avail_gb"] * 0.9)
    fitting = [c for c in cands if c[0] <= budget]
    sz, chosen = max(fitting) if fitting else min(cands)
    if verbose:
        rprint(f"[dim]   → เลือก {chosen.name} ({sz:.2f} GB / งบหน่วยความจำ {budget:.2f} GB)[/dim]")
    return chosen, mm_pick, sz

def _llama_pids() -> list:
    """pid ของ llama-server ที่ค้างอยู่ (ของโปรเจกต์นี้) — ใช้คืน VRAM ก่อนสตาร์ทใหม่"""
    try:
        out = subprocess.run(["pgrep", "-af", "llama-server"], capture_output=True, text=True).stdout
    except (FileNotFoundError, OSError):
        return []
    pids: list = []
    for line in out.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if LLAMA_SERVER_PROC and pid == LLAMA_SERVER_PROC.pid:
            continue
        if LLAMA_CPP_DIR in cmd or "llama-server" in Path(cmd.split(" ")[0]).name:
            pids.append(pid)
    return pids


def gpu_memory_users() -> list:
    """(pid, ชื่อ process, MB) ที่กำลังกิน VRAM อยู่ — ไว้แสดงตอน VRAM ไม่พอ"""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    users = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            users.append({"pid": int(parts[0]), "name": parts[1], "mb": int(float(parts[2] or 0))})
    return users


def load_timeout_for(size_gb: float) -> int:
    """เวลารอโหลดโมเดล — โมเดลใหญ่ (หรือ offload ลง RAM) ใช้เวลามากกว่า 4 นาที"""
    return int(min(900, max(240, 240 + (float(size_gb) - 8.0) * 40)))


def release_gpu_memory(port: int | None = None, wait_s: float = 12.0) -> dict:
    """ฆ่า llama-server ที่ค้าง (ของโปรเจกต์นี้) แล้วรอให้ VRAM ถูกคืน"""
    killed = _kill_server_on_port(int(port)) if port else []
    extra = [pid for pid in _llama_pids() if pid not in killed]
    for pid in extra:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    if extra:
        time.sleep(1.0)
        for pid in extra:
            try:
                os.kill(pid, 0)  # ยังอยู่?
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue
    deadline = time.monotonic() + wait_s
    free = 0.0
    while time.monotonic() < deadline:
        hw = check_gpu_and_recommend(verbose=False)
        free = float(hw["vram_free_gb"])
        if not hw["has_gpu"] or free >= 2.0:
            break
        time.sleep(1.0)
    return {"killed": killed + extra, "vram_free_gb": free}


def _kill_server_on_port(port: int) -> list[int]:
    """ฆ่าเฉพาะ process llama-server ที่ฟังพอร์ตที่กำหนด (ไม่แตะ process อื่นในเครื่อง)"""
    killed: list[int] = []
    try:
        out = subprocess.run(["pgrep", "-af", f"llama-server.*--port {port}"], capture_output=True, text=True).stdout
    except (FileNotFoundError, OSError):
        return killed
    for line in out.splitlines():
        pid_s = line.strip().split(" ", 1)[0]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if LLAMA_SERVER_PROC and pid == LLAMA_SERVER_PROC.pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM); killed.append(pid)
        except OSError:
            pass
    if killed:
        time.sleep(1.0)
    return killed

def stop_strata_server(port: int | None = None):
    global LLAMA_SERVER_PROC
    if LLAMA_SERVER_PROC and LLAMA_SERVER_PROC.poll() is None:
        LLAMA_SERVER_PROC.terminate()
        try:
            LLAMA_SERVER_PROC.wait(timeout=8)
        except subprocess.TimeoutExpired:
            LLAMA_SERVER_PROC.kill()
    LLAMA_SERVER_PROC = None
    _kill_server_on_port(int(port or DEFAULT_PORT))
    _invalidate_server_cache(int(port or DEFAULT_PORT))
    SPEC_STATE["active_label"] = "⏹️ ปิด Speculative"
    rprint("[bold yellow]🛑 หยุดการทำงานของ API Server เรียบร้อย[/bold yellow]")

def strata_server_status(port: int = DEFAULT_PORT):
    on, mid = _check_server_online(port)
    rprint(f"[bold {'green' if on else 'red'}]{'🟢 Online' if on else '🔴 Offline'} (:{port} • {mid} • {SPEC_STATE['active_label']})[/bold {'green' if on else 'red'}]")
    if LLAMA_LOG_FILE.exists():
        print("\n".join(LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]))

def start_strata_server(model_dir: str | None = MODEL_DIR, port: int = DEFAULT_PORT, ctx_size: int | None = None, restart: bool = True, spec_mode: str | None = None, draft_max: int | None = None, draft_model: str | None = None, kv_q8: bool | None = None, quant: str | None = None, extra_args: list[str] | None = None, **_) -> OpenAI:
    global LLAMA_SERVER_PROC
    if not model_dir:
        raise ValueError("กรุณาระบุ model_dir")
    srv_bin = install_llama_server()
    if restart:
        stop_strata_server(port=port)
    hw = check_gpu_and_recommend(verbose=False)
    if hw["has_gpu"] and hw["vram_free_gb"] < 2.0:
        # VRAM ค้างจากรอบก่อน (process ตายแต่ยังไม่คืนหน่วยความจำ) -> เก็บกวาดก่อน
        rprint(f"[yellow]🧹 VRAM ว่างแค่ {hw['vram_free_gb']:.2f} GB — เก็บกวาด llama-server ที่ค้างอยู่[/yellow]")
        info = release_gpu_memory(port)
        hw = check_gpu_and_recommend(verbose=False)
        if info["killed"]:
            rprint(f"[dim]   ปิด process ค้าง: {', '.join(map(str, info['killed']))}[/dim]")
        if hw["vram_free_gb"] < 2.0:
            holders = ", ".join(f"{u['name']}(pid {u['pid']}, {u['mb']} MB)" for u in gpu_memory_users()[:5])
            rprint(
                f"[bold red]⚠️ VRAM ยังว่างแค่ {hw['vram_free_gb']:.2f} GB"
                f"{' — ตัวที่ถืออยู่: ' + holders if holders else ''}[/bold red]\n"
                "[yellow]   จะสตาร์ทแบบ offload ลง RAM (ช้ากว่าปกติมาก) — ลอง Runtime → Restart session "
                "ถ้าต้องการคืน VRAM ทั้งหมด[/yellow]"
            )
    ctx_size = int(ctx_size or hw["recommended_ctx_size"])
    SPEC_STATE["ctx_size"] = ctx_size
    main_gguf, mmproj_gguf, size_gb = _pick_main_gguf(model_dir, hw=hw, prefer=quant)
    SPEC_STATE["last_model_dir"] = str(model_dir)
    usable_vram = max(0.0, float(hw["vram_free_gb"]) - 2.2) if hw["has_gpu"] else 0.0
    if hw["has_gpu"] and size_gb > usable_vram:
        rprint(
            f"[yellow]⚠️ โมเดล {size_gb:.1f} GB ใหญ่กว่า VRAM ที่ใช้ได้ ({usable_vram:.1f} GB) — "
            "จะ offload บางส่วนลง RAM ทำให้ช้าลง (เลือก quant เล็กลงถ้าไม่ไหว)[/yellow]"
        )
    load_timeout = load_timeout_for(size_gb)
    rprint(f"[dim]🧠 ctx-size = {ctx_size} tokens (ปรับตาม VRAM อัตโนมัติ) • รอโหลดสูงสุด {load_timeout}s[/dim]")
    help_txt = subprocess.run([srv_bin, "--help"], capture_output=True, text=True).stdout
    alias_name = Path(model_dir).name

    smode = (spec_mode if spec_mode is not None else str(SPEC_STATE["mode"])).lower().strip()
    dmax_in = int(draft_max if draft_max is not None else SPEC_STATE["draft_max"])
    dmodel_in = (draft_model if draft_model is not None else str(SPEC_STATE["draft_model"])).strip()
    use_kv_q8 = bool(kv_q8 if kv_q8 is not None else SPEC_STATE["kv_q8"])
    has_draft = bool(dmodel_in and Path(dmodel_in).is_file())
    cat_mtp = bool(MODEL_CATALOG.get(alias_name, {}).get("has_mtp")) or ("mtp" in main_gguf.name.lower())

    if smode in ("auto", "draft"):
        eff_mode = "draft_file" if has_draft else ("mtp" if cat_mtp else "ngram")
    elif smode == "mtp":
        eff_mode = "draft_file_mtp" if has_draft else "mtp"
    elif smode == "ngram":
        eff_mode = "ngram"
    else:
        eff_mode = "off"

    def _build_cmd(target_eff: str) -> tuple[list[str], str]:
        cmd = [srv_bin, "--model", str(main_gguf), "--alias", alias_name, "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(ctx_size), "--threads", str(hw["recommended_threads"])]
        if hw["has_gpu"]:
            if "--fit" in help_txt:
                cmd.extend(["--fit", "on", "--fit-target", "1024"])
            else:
                usable = max(1.0, hw["vram_free_gb"] - 2.2)
                n_layers = _gguf_layer_count(main_gguf)
                if not n_layers:
                    n_layers = max(8, int(size_gb / 0.5))
                    gb_per_layer = 0.5
                else:
                    gb_per_layer = size_gb / n_layers
                n_gpu = n_layers if size_gb <= usable else max(1, min(n_layers, int(usable / max(gb_per_layer, 0.01))))
                cmd.extend(["--n-gpu-layers", str(n_gpu)])
                rprint(f"[dim]   🧮 offload {n_gpu}/{n_layers} layer (โมเดล {size_gb:.2f}GB • ใช้ได้ {usable:.2f}GB)[/dim]")
            if "--flash-attn" in help_txt:
                cmd.extend(["--flash-attn", "on"])
        else:
            cmd.extend(["--n-gpu-layers", "0"])
        if "--jinja" in help_txt:
            cmd.append("--jinja")
        if use_kv_q8 and "--cache-type-k" in help_txt:
            cmd.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])

        dmax_flag = "--draft-max" if "--draft-max" in help_txt else "--draft"
        no_mm = False; lbl = "⏹️ ปิด Speculative"
        if target_eff == "mtp" and "--spec-type" in help_txt:
            dm = dmax_in if dmax_in > 0 else 2
            cmd.extend(["--spec-type", "draft-mtp", dmax_flag, str(dm)]); no_mm = True; lbl = f"🚀 MTP (max={dm})"
        elif target_eff == "ngram" and "--spec-type" in help_txt:
            ng = "ngram-mod" if "ngram-mod" in help_txt else ("ngram-simple" if "ngram-simple" in help_txt else "ngram-cache")
            dm = dmax_in if dmax_in > 0 else 16
            cmd.extend(["--spec-type", ng, dmax_flag, str(dm)]); lbl = f"📚 N-gram ({ng}, max={dm})"
        elif target_eff in ("draft_file", "draft_file_mtp") and has_draft:
            dm = dmax_in if dmax_in > 0 else (3 if target_eff == "draft_file_mtp" else 6)
            cmd.extend(["--model-draft", dmodel_in, "-ngld", "99", dmax_flag, str(dm)])
            if target_eff == "draft_file_mtp" and "--spec-type" in help_txt:
                cmd.extend(["--spec-type", "draft-mtp"]); no_mm = True
            lbl = f"⚡ Draft ({Path(dmodel_in).name[:16]}.., max={dm})"
        if use_kv_q8:
            lbl += " + KV:Q8"
        if no_mm:
            if "--no-mmproj" in help_txt:
                cmd.append("--no-mmproj")
        elif mmproj_gguf and hw["vram_free_gb"] >= size_gb + 3.0:
            cmd.extend(["--mmproj", str(mmproj_gguf)])
        cmd.extend(str(a) for a in (extra_args or []))
        return cmd, lbl

    def _spawn(target_eff: str) -> tuple[bool, str]:
        global LLAMA_SERVER_PROC
        cmd_run, lbl_run = _build_cmd(target_eff)
        rprint(f"[bold cyan]🚀 เปิด llama-server (:{port}):[/bold cyan] [yellow]{main_gguf.name}[/yellow] ({size_gb:.2f}GB) • [magenta]{lbl_run}[/magenta]")
        with open(LLAMA_LOG_FILE, "w", encoding="utf-8") as log_f:
            LLAMA_SERVER_PROC = subprocess.Popen(
                cmd_run, stdout=log_f, stderr=subprocess.STDOUT, text=True, env=server_env()
            )
        deadline = time.monotonic() + load_timeout
        while time.monotonic() < deadline:
            if LLAMA_SERVER_PROC.poll() is not None:
                tail = "\n".join(LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:])
                return False, (
                    f"process จบตัวก่อนพร้อม (exit={LLAMA_SERVER_PROC.returncode})\n{tail}{start_error_hint(tail)}"
                )
            is_ok, loaded_id = _check_server_online(port, ttl=0.0)
            if is_ok:
                SPEC_STATE["active_label"] = lbl_run
                elapsed = int(load_timeout - (deadline - time.monotonic()))
                rprint(f"[bold green]✅ Server พร้อมใน {elapsed}s! (model='{loaded_id}' • {lbl_run})[/bold green]")
                return True, loaded_id
            elapsed = int(load_timeout - (deadline - time.monotonic()))
            if elapsed and elapsed % 10 == 0:
                rprint(f"[dim]   ⏳ กำลังโหลดโมเดลลง GPU/RAM... ({elapsed}s)[/dim]")
            time.sleep(1.0)
        if LLAMA_SERVER_PROC and LLAMA_SERVER_PROC.poll() is None:
            LLAMA_SERVER_PROC.terminate()
            try:
                LLAMA_SERVER_PROC.wait(timeout=10)
            except subprocess.TimeoutExpired:
                LLAMA_SERVER_PROC.kill()
        tail = "\n".join(LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:])
        return False, f"โหลดโมเดลเกิน {load_timeout} วินาที\n{tail}{start_error_hint(tail)}"

    fallbacks = [eff_mode] + (["draft_file", "ngram"] if eff_mode == "draft_file_mtp" else (["ngram"] if eff_mode in ("mtp", "draft_file") else []))
    last_err = ""
    for idx, m_try in enumerate(dict.fromkeys(fallbacks)):
        if idx > 0:
            rprint(f"[bold yellow]⚠️ โหมดก่อนหน้าไม่พร้อม กำลังสลับเป็น '{m_try}' อัตโนมัติ...[/bold yellow]")
            _kill_server_on_port(port)
        ok, info = _spawn(m_try)
        if ok:
            return get_openai_client(port=port)  # type: ignore[return-value]
        last_err = info
    raise RuntimeError(f"llama-server สตาร์ทไม่สำเร็จทุกโหมด:\n{last_err}")

def switch_strata_model(option: str, model_dir: str = MODEL_DIR, port: int = DEFAULT_PORT) -> OpenAI:
    if option not in MODEL_CATALOG:
        raise ValueError(f"เลือกโมเดลจาก: {ALL_PRESET_KEYS}")
    return start_strata_server(model_dir=str(Path(model_dir) / option), port=port, restart=True)

# ---------------------------------------------------------------------------
# กันกดซ้ำ/สตาร์ทซ้อน: ทุกคำสั่งที่แตะ server ต้องผ่าน lock นี้
# ---------------------------------------------------------------------------
_SERVER_LOCK = threading.RLock()


def _guarded(name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _SERVER_LOCK.acquire(blocking=False):
            raise RuntimeError(f"{name}: มีคำสั่งอื่นกำลังจัดการ server อยู่ — รอให้เสร็จแล้วลองใหม่")
        try:
            return fn(*args, **kwargs)
        finally:
            _SERVER_LOCK.release()

    return wrapper


start_strata_server = _guarded("start_strata_server", start_strata_server)
stop_strata_server = _guarded("stop_strata_server", stop_strata_server)


def get_last_model_dir() -> str | None:
    """โมเดลที่สตาร์ทล่าสุด (เก็บใน SPEC_STATE แทน global แยก)"""
    return SPEC_STATE.get("last_model_dir")
