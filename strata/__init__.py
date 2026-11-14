"""Strata Studio — จัดการโมเดล GGUF บน llama.cpp พร้อม RAG และ UI สำหรับ Colab

โมดูลถูกออกแบบให้ import ได้เร็ว: ตัวที่ต้องใช้ไลบรารีหนัก (fastembed, pypdf, openai,
ipywidgets) จะถูก import ตอนใช้งานจริง

    from strata.bootstrap import ensure_environment   # เตรียม Nix + venv (ครั้งเดียว)
    from strata.notebook import launch                # เปิด UI
    launch()
"""

from __future__ import annotations

from .config import (
    ALL_PRESET_KEYS,
    DEFAULT_PORT,
    DEFAULT_RAG_PATH,
    DRAFT_CATALOG,
    MODEL_CATALOG,
    MODEL_DIR,
    SPEC_STATE,
)

__version__ = "0.3.0"

__all__ = [
    "ALL_PRESET_KEYS",
    "DEFAULT_PORT",
    "DEFAULT_RAG_PATH",
    "DRAFT_CATALOG",
    "MODEL_CATALOG",
    "MODEL_DIR",
    "SPEC_STATE",
    "__version__",
]
