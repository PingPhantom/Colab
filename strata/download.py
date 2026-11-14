"""ดาวน์โหลด/ติดตั้ง/ลบโมเดล GGUF จาก Hugging Face (download-deepseek → hf → huggingface_hub)."""
from __future__ import annotations

import fnmatch
import functools
import os
import re
import shlex
import shutil
import threading
from pathlib import Path

import requests
from rich import print as rprint

from .config import (
    ALL_PRESET_KEYS,
    CUSTOM_HF_REPO,
    DEFAULT_PORT,
    DRAFT_CATALOG,
    DRAFT_DIR,
    MODEL_CATALOG,
    MODEL_DIR,
    ModelOptionType,
)
from .server import check_disk_space, start_strata_server, stop_strata_server, switch_strata_model
from .utils import _dir_bytes, _fmt_bytes, run_stream

DOWNLOADER_HINT = (
    "ไม่พบตัวดาวน์โหลด GGUF (download-deepseek / hf / huggingface_hub) — ปกติแปลว่าเซลล์ 1 ยังรันไม่จบ "
    "หรือ PATH ของ nix devshell หลุดไปแล้ว ให้รัน `from strata.bootstrap import ensure_environment; "
    "ensure_environment()` (หรือ `activate_devshell()` + `link_tools()`) แล้วลองโหลดใหม่"
)


def download_command(repo_id: str, model_dir: str, inc: list[str], exc: list[str]) -> list[str]:
    """สร้างคำสั่งดาวน์โหลดจากเครื่องมือที่หาเจอ — คืน [] ถ้าต้องใช้ huggingface_hub แทน"""
    if shutil.which("download-deepseek"):
        args = [shlex.quote(str(model_dir))]
        for p in inc:
            args += ["--include", shlex.quote(p)]
        for p in exc:
            args += ["--exclude", shlex.quote(p)]
        os.environ["HF_REPO_ID"] = str(repo_id)  # wrapper ใน devshell อ่าน repo จาก env นี้
        return ["download-deepseek", *args]
    if shutil.which("hf"):
        args = ["hf", "download", shlex.quote(str(repo_id)), "--local-dir", shlex.quote(str(model_dir))]
        for p in inc:
            args += ["--include", shlex.quote(p)]
        for p in exc:
            args += ["--exclude", shlex.quote(p)]
        return args
    return []


def download_with_module(repo_id: str, model_dir: str, inc: list[str], exc: list[str]) -> None:
    """ทางสุดท้าย: ใช้ huggingface_hub ในโปรเซสนี้เลย (ไม่ต้องพึ่ง PATH ของ devshell)"""
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as e:
        raise RuntimeError(DOWNLOADER_HINT) from e
    rprint(f"[dim]📥 ใช้ huggingface_hub (Python) ดาวน์โหลด {repo_id} → {model_dir}[/dim]")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        allow_patterns=inc or None,
        ignore_patterns=exc or None,
    )


def download_model_files(include: list[str] | str | None = None, exclude: list[str] | str | None = None, model_dir: str = MODEL_DIR, repo_id: str = CUSTOM_HF_REPO, force: bool = False) -> dict:
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", str(repo_id)):
        raise ValueError(f"repo_id ต้องอยู่ในรูป org/model (ได้รับ {repo_id!r})")
    inc = [include] if isinstance(include, str) else list(include or [])
    exc = [exclude] if isinstance(exclude, str) else list(exclude or [])
    try:
        r = requests.get(f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true", timeout=15)
        r.raise_for_status()
        entries = [x for x in r.json() if isinstance(x, dict) and x.get("type") == "file"]
    except requests.RequestException as e:
        raise RuntimeError(f"ตรวจรายการไฟล์ของ {repo_id} บน Hugging Face ไม่สำเร็จ: {e}") from e
    need, pending = 0, []
    for it in entries:
        fn = it["path"]
        if inc and not any(fnmatch.fnmatch(fn, p) or fnmatch.fnmatch(Path(fn).name, p) for p in inc):
            continue
        if exc and any(fnmatch.fnmatch(fn, p) or fnmatch.fnmatch(Path(fn).name, p) for p in exc):
            continue
        lf = Path(model_dir) / fn
        if lf.exists() and lf.stat().st_size == it.get("size", 0):
            continue
        need += int(it.get("size", 0) or 0); pending.append(fn)
    req_gb = need / (1024**3)
    if not pending and not force:
        rprint(f"[bold green]✅ ไฟล์ครบแล้วใน {model_dir} — ไม่ต้องโหลดเพิ่ม[/bold green]")
        return {"downloaded": 0, "required_gb": 0.0}
    di = check_disk_space(model_dir, required_gb=req_gb)
    if di["free_gb"] < req_gb and not force:
        raise RuntimeError(f"ดิสก์ไม่พอ: ต้องการ {req_gb:.2f} GB แต่ว่าง {di['free_gb']:.2f} GB (ใช้ force=True เพื่อข้ามการตรวจ)")
    rprint(f"[dim]📥 ต้องโหลด {len(pending)} ไฟล์ (~{_fmt_bytes(need)}) จาก {repo_id}[/dim]")
    cmd = download_command(repo_id, str(model_dir), inc, exc)
    if cmd:
        run_stream(" ".join(cmd))
    else:
        download_with_module(repo_id, str(model_dir), inc, exc)
    return {"downloaded": len(pending), "required_gb": round(req_gb, 2)}

def _find_draft_gguf(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    cands = sorted([p for p in folder.rglob("*.gguf") if "mmproj" not in p.name.lower() and "imatrix" not in p.name.lower()], key=lambda p: p.stat().st_size)
    return next(iter(cands), None)

def install_draft_model(draft_key_or_spec: str, draft_base_dir: str = DRAFT_DIR) -> tuple[str, bool]:
    base = Path(draft_base_dir)
    if draft_key_or_spec in DRAFT_CATALOG:
        cfg = DRAFT_CATALOG[draft_key_or_spec]; sub = base / draft_key_or_spec
        rprint(f"[bold magenta]⚡ กำลังโหลดไฟล์ Draft:[/bold magenta] [cyan]{draft_key_or_spec}[/cyan] -> {cfg['repo_id']}")
        download_model_files(include=cfg["include"], exclude=cfg["exclude"], model_dir=str(sub), repo_id=cfg["repo_id"])
        found = _find_draft_gguf(sub)
        if not found:
            raise FileNotFoundError(f"ไม่พบไฟล์ Draft .gguf ใน {sub}")
        rprint(f"[bold green]✅ พร้อมใช้ Draft:[/bold green] {found}")
        return str(found), bool(cfg.get("is_mtp", False))
    spec = draft_key_or_spec.strip()
    if Path(spec).is_file():
        return spec, ("mtp" in Path(spec).name.lower())
    repo_part, pat_part = spec.split(":", 1) if ":" in spec else (spec, "*Q4_K_M*.gguf")
    repo_part, pat_part = repo_part.strip(), pat_part.strip() or "*Q4_K_M*.gguf"
    if "/" not in repo_part:
        raise ValueError("ระบุเป็น org/repo หรือ org/repo:*Q4_K_M*.gguf หรือพาธไฟล์ .gguf")
    sub = base / ("custom_" + re.sub(r"[^a-zA-Z0-9_.-]+", "_", repo_part))
    download_model_files(include=[pat_part], exclude=["*mmproj*"], model_dir=str(sub), repo_id=repo_part)
    found = _find_draft_gguf(sub)
    if not found:
        raise FileNotFoundError(f"ไม่พบไฟล์ .gguf ({pat_part}) ใน {repo_part}")
    return str(found), ("mtp" in found.name.lower())

def install_model(option: ModelOptionType = "qwen38-27b-cyber", model_dir: str = MODEL_DIR, repo_id: str | None = None, include: list[str] | str | None = None, exclude: list[str] | str | None = None, auto_start_server: bool = False, port: int = DEFAULT_PORT, force: bool = False) -> dict | list[dict]:
    if option == "all":
        return install_all_models(model_dir=model_dir, force=force, port=port)
    if option not in MODEL_CATALOG:
        raise ValueError(f"เลือกโมเดลจาก: {[*MODEL_CATALOG.keys(), 'all']}")
    cfg = MODEL_CATALOG[option]; sub = str(Path(model_dir) / option)
    if cfg.get("kind") == "image":  # โมเดลสร้างภาพ: ดาวน์โหลดเหมือนกัน แต่ผูกเข้า ComfyUI ด้วย
        from .imagegen import install_image_model

        return install_image_model(option, model_dir=model_dir, force=force)
    rprint(f"[bold magenta]📦 เตรียมโมเดล:[/bold magenta] [cyan]{option}[/cyan] -> {repo_id or cfg['repo_id']}")
    download_model_files(include=include if include is not None else cfg["include"], exclude=exclude if exclude is not None else cfg["exclude"], model_dir=sub, repo_id=repo_id or cfg["repo_id"], force=force)
    if auto_start_server:
        start_strata_server(model_dir=sub, port=port)
    return {"option": option, "model_dir": sub}

def install_all_models(options: list[str] | None = None, model_dir: str = MODEL_DIR, force: bool = False, start_after: str | None = None, port: int = DEFAULT_PORT) -> list[dict]:
    keys = options or ALL_PRESET_KEYS
    est = sum(MODEL_CATALOG[k]["approx_gb"] for k in keys if k in MODEL_CATALOG)
    di = check_disk_space(model_dir, required_gb=est)
    if di["free_gb"] < est and not force:
        raise RuntimeError(f"พื้นที่ไม่พอโหลดครบ {len(keys)} โมเดล (~{est:.1f} GB)")
    res = []
    for i, k in enumerate(keys, 1):
        rprint(f"\n[bold cyan]📥 [{i}/{len(keys)}] ดาวน์โหลด {k}[/bold cyan]")
        try:
            info = install_model(k, model_dir=model_dir, force=force, port=port)  # type: ignore[arg-type]
            if isinstance(info, dict):
                res.append({**info, "status": "success"})
        except Exception as e:
            rprint(f"[bold red]❌ {k} ล้มเหลว: {e}[/bold red]"); res.append({"option": k, "status": f"failed: {e}"})
    if start_after and start_after in MODEL_CATALOG:
        switch_strata_model(start_after, model_dir=model_dir, port=port)
    return res

def clean_download_cache(model_dir: str = MODEL_DIR, include_global_hf_cache: bool = True) -> float:
    freed, base = 0, Path(model_dir)
    if base.exists():
        for cd in base.rglob(".cache"):
            if cd.is_dir():
                freed += _dir_bytes(cd); shutil.rmtree(cd, ignore_errors=True)
        for pat in ("*.incomplete", "*.part", "*.tmp"):
            for tf in base.rglob(pat):
                if tf.is_file():
                    freed += tf.stat().st_size; tf.unlink(missing_ok=True)
    if include_global_hf_cache:
        hc = Path.home() / ".cache/huggingface/hub"
        if hc.exists():
            freed += _dir_bytes(hc); shutil.rmtree(hc, ignore_errors=True)
    rprint(f"[bold green]🧹 ล้างแคชสำเร็จ คืนพื้นที่ {_fmt_bytes(freed)}[/bold green]")
    return round(freed / (1024**3), 2)

def remove_model(options: str | list[str], model_dir: str = MODEL_DIR, clear_cache: bool = True, stop_server: bool = False) -> dict:
    base = Path(model_dir)
    if not base.exists():
        return {"removed": [], "freed_gb": 0.0}
    if stop_server or options == "all":
        stop_strata_server()
    targets = [d.name for d in base.iterdir() if d.is_dir()] if options == "all" else ([options] if isinstance(options, str) else list(options))
    before = shutil.disk_usage(base).free; removed = []
    base_abs = base.resolve()
    for opt in targets:
        tp = (base / opt).resolve()
        if tp.parent != base_abs or not tp.is_dir():
            rprint(f"[bold red]⛔ ข้าม '{opt}': ไม่ใช่โฟลเดอร์โมเดลที่ถูกต้อง[/bold red]")
            continue
        sz = _dir_bytes(tp); shutil.rmtree(tp, ignore_errors=True); removed.append(opt)
        rprint(f"[bold yellow]🗑️ ลบ '{opt}' ({_fmt_bytes(sz)})[/bold yellow]")
    if clear_cache:
        clean_download_cache(model_dir=model_dir, include_global_hf_cache=False)
    freed = max(0, shutil.disk_usage(base if base.exists() else Path("/content")).free - before)
    rprint(f"[bold green]✅ คืนพื้นที่ดิสก์รวม {_fmt_bytes(freed)}[/bold green]")
    return {"removed": removed, "freed_gb": round(freed / (1024**3), 2)}

def keep_only_models(keep: str | list[str], model_dir: str = MODEL_DIR) -> dict:
    base = Path(model_dir)
    if not base.exists():
        return {"kept": [], "removed": [], "freed_gb": 0.0}
    ks = {keep} if isinstance(keep, str) else set(keep)
    rem = [d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("_") and d.name not in ks]
    return remove_model(rem, model_dir=model_dir) if rem else {"kept": sorted(ks), "removed": [], "freed_gb": 0.0}

# ---------------------------------------------------------------------------
# กันติดตั้งซ้อน (เช่น ดับเบิลคลิกปุ่มโหลด)
# ---------------------------------------------------------------------------
_INSTALL_LOCK = threading.RLock()


def _install_guarded(name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _INSTALL_LOCK.acquire(blocking=False):
            raise RuntimeError(f"{name}: กำลังติดตั้ง/ลบโมเดลอยู่ — รอให้เสร็จแล้วลองใหม่")
        try:
            return fn(*args, **kwargs)
        finally:
            _INSTALL_LOCK.release()

    return wrapper


install_model = _install_guarded("install_model", install_model)
install_all_models = _install_guarded("install_all_models", install_all_models)
remove_model = _install_guarded("remove_model", remove_model)
