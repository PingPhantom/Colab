"""ตรวจจับสภาพแวดล้อมที่รันอยู่ (Colab / Kaggle / เครื่องตัวเอง) พร้อมค่าเริ่มต้นที่เหมาะกับแต่ละที่

ทำไมต้องมี: Kaggle ต่างจาก Colab หลายจุด
- โฟลเดอร์เขียนได้จำกัด (``/kaggle/working`` 20 GB) แต่มีดิสก์ชั่วคราวใหญ่กว่าที่ ``/kaggle/temp``
- **ไม่มี** port proxy ให้เปิดเบราว์เซอร์ (Colab มี ``proxyPort``) → ต้องใช้ tunnel
- มี GPU ได้ 2×T4 / P100 และรันเป็น root
"""

from __future__ import annotations

import os
from pathlib import Path

COLAB_HOME = "/content"
KAGGLE_WORK = "/kaggle/working"
KAGGLE_TEMP = "/kaggle/temp"


def _default_exists(path: str) -> bool:
    try:
        return Path(path).is_dir()
    except OSError:  # pragma: no cover - ระบบไฟล์พัง
        return False


def _default_writable(path: str) -> bool:
    try:
        return os.access(path, os.W_OK)
    except OSError:  # pragma: no cover
        return False


def detect(env: dict[str, str] | None = None, exists=None, writable=None) -> dict:
    """คืนข้อมูลสภาพแวดล้อม — `env`/`exists`/`writable` ใส่เข้ามาเพื่อเทสต์ได้โดยไม่ต้องมี Colab/Kaggle จริง"""
    env = dict(os.environ if env is None else env)
    exists = exists or _default_exists
    writable = writable or _default_writable

    is_kaggle = any(k.startswith("KAGGLE_") for k in env) or exists(KAGGLE_WORK)
    is_colab = "COLAB_RELEASE_TAG" in env or "COLAB_GPU" in env or (exists(COLAB_HOME) and not is_kaggle)

    if is_kaggle:
        name = "kaggle"
    elif is_colab:
        name = "colab"
    else:
        name = "local"

    home = env.get("STRATA_HOME") or _suggest_home(name, exists, writable)
    return {
        "name": name,
        "label": {"colab": "🟡 Colab", "kaggle": "🐨 Kaggle", "local": "💻 Local"}[name],
        "home": home,
        "model_dir": env.get("STRATA_MODEL_DIR") or f"{home}/models",
        # Colab เท่านั้นที่มี proxyPort ให้เปิดเว็บจากเบราว์เซอร์ได้
        "proxy_ports": name == "colab",
        "tunnel_needed": name != "colab",
        "workdir": KAGGLE_WORK if name == "kaggle" and exists(KAGGLE_WORK) else ("/content" if name == "colab" else str(Path.home())),
        "notes": _notes(name),
    }


def _suggest_home(name: str, exists, writable=None) -> str:
    """โฟลเดอร์เก็บโมเดล/log — เลือกที่ "เขียนได้และใหญ่พอ" ของแต่ละแพลตฟอร์ม"""
    if name == "colab":
        return COLAB_HOME if exists(COLAB_HOME) else str(Path.home() / ".strata")
    writable = writable or _default_writable
    if name == "kaggle":
        # /kaggle/working จำกัด 20 GB (เป็น output) — โมเดล GGUF ตัวละ 3-17 GB จึงใช้ temp เป็นค่าเริ่มต้น
        if exists(KAGGLE_TEMP) and writable(KAGGLE_TEMP):
            return f"{KAGGLE_TEMP}/strata"
        if exists(KAGGLE_WORK) and writable(KAGGLE_WORK):
            return f"{KAGGLE_WORK}/strata"
        return str(Path.home() / ".strata")
    return str(Path.home() / ".strata")


def _notes(name: str) -> list[str]:
    if name == "kaggle":
        return [
            "Kaggle ต้องเปิด Internet ใน notebook settings ก่อน (ไม่งั้นโหลดโมเดล/Nix ไม่ได้)",
            f"โมเดลเก็บที่ {KAGGLE_TEMP}/strata โดยค่าเริ่มต้น (หายเมื่อจบ session) — "
            f"ถ้าอยากเก็บถาวรให้ตั้ง STRATA_HOME={KAGGLE_WORK} (จำกัด 20 GB)",
            "Kaggle ไม่มี port proxy: ลิงก์ ComfyUI จะเปิดผ่าน cloudflared tunnel อัตโนมัติ",
        ]
    if name == "colab":
        return ["Colab ใช้ proxyPort เปิด ComfyUI ได้; ดิสก์ /content ~110 GB"]
    return ["เครื่องตัวเอง: ต้องมี GPU + llama.cpp หรือปล่อยให้สคริปต์โหลดไบนารีให้"]


def summary() -> str:
    """ข้อความสั้น ๆ ใช้แสดงใน UI/notebook"""
    info = detect()
    return f"{info['label']} • home={info['home']} • tunnel={'ต้องใช้' if info['tunnel_needed'] else 'ไม่ต้อง'}"


__all__ = ["COLAB_HOME", "KAGGLE_TEMP", "KAGGLE_WORK", "detect", "summary"]
