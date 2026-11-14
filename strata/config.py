"""ค่าคงที่และ catalog ของ Strata Studio — import ได้โดยไม่ต้องมีไลบรารีภายนอก."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# ตัวอักษร $ ใช้ประกอบสคริปต์ shell ใน f-string (เลี่ยงไม่ให้ Python/shell แทนค่าเอง)
S = chr(36)


DEFAULT_PORT: int = int(os.environ.get("STRATA_PORT", "18080"))

NVIDIA_BIN_DIRS = "/usr/local/nvidia/bin:/usr/local/cuda/bin"

NVIDIA_LIB_DIRS = "/usr/lib64-nvidia:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

ModelOptionType = Literal[
    "none", "all", "sakura-minimax-h3", "qwen38-27b-cyber",
    "qwen38-27b-rvn-heretic", "qwen38-27b-coletti", "muse-glimmer-30b-heretic",
    "minicpm-v47-35b", "lfm25-24b-a2b", "qwen35-9b-defiant-fable",
    "ornith-15-9b", "nemotron-35-bolt-74b",
    "deepseek-v41-flash-abliterated", "qwen38-27b-turbo-fable",
    "deepseek-v4-pro-4b-heretic", "custom",
]

ALL_PRESET_KEYS: list[str] = [
    "sakura-minimax-h3", "qwen38-27b-cyber", "qwen38-27b-rvn-heretic",
    "qwen38-27b-coletti", "muse-glimmer-30b-heretic", "minicpm-v47-35b",
    "lfm25-24b-a2b", "qwen35-9b-defiant-fable", "ornith-15-9b",
    "nemotron-35-bolt-74b", "deepseek-v41-flash-abliterated",
    "qwen38-27b-turbo-fable", "deepseek-v4-pro-4b-heretic",
    "qwen-image-21-uncensored", "flux-3-klein-uncensored",
]

MODEL_OPTION: ModelOptionType = "none"

def _default_home() -> str:
    """โฟลเดอร์เก็บโมเดล/log: Colab=/content, Kaggle=/kaggle/temp/strata, อื่น ๆ ~/.strata

    เลือกตามแพลตฟอร์มจริง (ดู `strata/platforms.py`) — override ได้ด้วย env `STRATA_HOME`
    """
    if os.environ.get("STRATA_HOME"):
        return os.environ["STRATA_HOME"]
    from .platforms import detect  # import ตอนใช้ กัน circular import

    return detect()["home"]


STRATA_HOME: str = _default_home()
MODEL_DIR: str = os.environ.get("STRATA_MODEL_DIR", f"{STRATA_HOME}/models")

DRAFT_DIR: str = os.environ.get("STRATA_DRAFT_DIR", f"{MODEL_DIR}/_drafts")

CUSTOM_HF_REPO: str = "AtomicChat/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Uncensored-GGUF"

SPEC_STATE: dict = {
    "mode": "off", "draft_max": 0, "draft_model": "",
    "kv_q8": False, "active_label": "⏹️ ปิด Speculative",
}

DRAFT_CATALOG: dict[str, dict] = {
    "qwen38-coletti-mtp": {
        "repo_id": "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF",
        "include": ["*draft*Q8_0*", "*draft*Q4*"], "exclude": ["*mmproj*"],
        "is_mtp": True, "desc": "🚀 Qwen3.8-27B MTP Draft (Coletti)",
    },
    "qwen35-0.8b": {
        "repo_id": "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
        "include": ["*Q4_K_M*.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "desc": "⚡ Qwen3.5-0.8B Q4_K_M (~0.5GB • Bartowski)",
    },
    "qwen3-0.6b": {
        "repo_id": "unsloth/Qwen3-0.6B-GGUF",
        "include": ["*Q4_K_M*.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "desc": "⚡ Qwen3-0.6B Q4_K_M (~0.4GB • Unsloth)",
    },
    "qwen3-coder-0.8b": {
        "repo_id": "DavidAU/Qwen3-Zero-Coder-Reasoning-V2-0.8B-NEO-EX-GGUF",
        "include": ["*Q4_K_M*.gguf", "*IQ4_XS*.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "desc": "💻 Qwen3-Zero-Coder-0.8B (DavidAU)",
    },
}

MODEL_CATALOG: dict[str, dict] = {
    "sakura-minimax-h3": {"repo_id": "webmp3/Sakura-MiniMax-H3-Uncensored-GGUF", "include": ["*Q4_K*", "*IQ4*"], "exclude": None, "approx_gb": 15.5, "has_mtp": False, "desc": "🌸 Sakura-MiniMax-H3-Uncensored (26B) Multimodal"},
    "qwen38-27b-cyber": {"repo_id": "cyjin-yl/Qwen3.8-27B-Uncensored-Cyber-agentic-imatrix-GGUF", "include": ["*IQ4_XS*", "*mmproj*"], "exclude": None, "approx_gb": 15.2, "has_mtp": False, "desc": "💻 Qwen3.8-27B-Uncensored-Cyber Coding & Agent"},
    "qwen38-27b-rvn-heretic": {"repo_id": "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF", "include": ["RVN-Q4_K_M.gguf", "*mmproj*"], "exclude": ["*mtp*", "*multilingual*"], "approx_gb": 16.5, "has_mtp": False, "desc": "🔥 Qwen3.8-27B-RVN-Heretic (RVN Q4_K_M) ARA 3-pass"},
    "qwen38-27b-coletti": {"repo_id": "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF", "include": ["*IQ4_XS*", "*mmproj*"], "exclude": ["*noMTP*", "*draft*"], "approx_gb": 15.3, "has_mtp": True, "desc": "🧠 Qwen3.8-27B-Uncensored Coletti (IQ4_XS + MTP)"},
    "muse-glimmer-30b-heretic": {"repo_id": "0bserverx/Muse-Glimmer-30B-Heretic-Uncensored-GGUF", "include": ["*IQ4_XS*", "*mmproj*"], "exclude": None, "approx_gb": 15.3, "has_mtp": False, "desc": "✨ Muse-Glimmer-30B-Heretic (IQ4_XS) Multimodal"},
    "minicpm-v47-35b": {"repo_id": "AtomicChat/MiniCPM-V-4.7-35B-A3B-Uncensored-GGUF", "include": ["*IQ4_XS*", "*Q4_K_M*", "*mmproj*"], "exclude": None, "approx_gb": 20.5, "has_mtp": False, "desc": "👁️ MiniCPM-V-4.7-35B-A3B (3B active MoE) Vision"},
    "lfm25-24b-a2b": {"repo_id": "AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF", "include": ["*Q4_K_M*", "*IQ4_XS*"], "exclude": None, "approx_gb": 14.5, "has_mtp": False, "desc": "⚡ LFM2.5-24B-A2B (2B active MoE) เร็วสูง หลายภาษา"},
    # repo มีทั้งรุ่น MTP และไม่-MTP: `*Q4_K_M*` เดิมกวาดมาทั้งคู่ (~13.8GB) -> เลือกเฉพาะไฟล์ MTP ตัวเดียว
    "qwen35-9b-defiant-fable": {
        "repo_id": "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF",
        "include": ["*-NEO-MAX-MTP-Q4_K_M.gguf", "mmproj-F16.gguf"],
        "exclude": ["*plusIQ*", "mmproj-*F32*", "mmproj-*BF16*"],
        "approx_gb": 7.4,
        "has_mtp": True,
        "desc": "📖 Qwen3.5-9B-Defiant-Fable Heretic NEO (MTP Q4_K_M 7.0GB + mmproj) T4 ลื่น",
    },
    "ornith-15-9b": {"repo_id": "mradermacher/Ornith-1.5-9B-uncensored-GGUF", "include": ["*Q4_K_M*", "*mmproj-Q8_0*"], "exclude": None, "approx_gb": 6.4, "has_mtp": False, "desc": "🦅 Ornith-1.5-9B-uncensored (Q4_K_M) ลง T4 ได้ทั้งตัว"},
    "nemotron-35-bolt-74b": {"repo_id": "AtomicChat/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Uncensored-GGUF", "include": ["*IQ2*", "*IQ3_XXS*", "*mmproj*"], "exclude": None, "approx_gb": 24.5, "has_mtp": False, "desc": "🚀 Nemotron-3.5-Bolt-74B-A9B (9B active MoE)"},
    # DavidAU "TURBO Cold-Fusion 735-882" — repo มีทั้งรุ่น MTP / LOW-MTP / ไม่-MTP
    # (ชื่อไฟล์ลงท้าย ...-MAX-MTP-<quant>.gguf) ใช้ MTP IQ4_XS 17.0GB + mmproj F16
    "qwen38-27b-turbo-fable": {
        "repo_id": "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF",
        "include": ["*-MAX-MTP-IQ4_XS.gguf", "mmproj-F16.gguf"],
        "exclude": ["*LOW-MTP*", "mmproj-*BF16*", "mmproj-*F32*"],
        "approx_gb": 16.8,
        "has_mtp": True,
        "desc": "🧊 Qwen3.8-27B-TurboFCFusion 735-882 Heretic CODER (MTP IQ4_XS 16.7GB)",
    },
    # ⚠️ โมเดลระดับ data-center: Mix-128G ≈ 225 GB (Mix-256G ≈ 346 GB) — ต้องมีดิสก์ ≥ 250 GB
    # และ RAM/VRAM มาก (A100/H100 หรือ offload ลง CPU) ไม่เหมาะกับ Colab ฟรี (ดิสก์ ~110 GB)
    "deepseek-v41-flash-abliterated": {
        "repo_id": "orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF",
        "include": ["Mix-128G/*", "mmproj-*BF16.gguf"],
        "exclude": ["imatrix.gguf"],
        "approx_gb": 224.6,
        "has_mtp": False,
        "desc": "🐋 DeepSeek-V4.1-Flash-1017 (abliterated, MoE, 5 shards ~225GB — A100/H100 เท่านั้น)",
    },
    "deepseek-v4-pro-4b-heretic": {
        "repo_id": "mradermacher/DeepSeek-V4-Pro-Qwen3.5-4B-UNCENSORED-heretic-GGUF",
        "include": ["*heretic.Q5_K_M.gguf", "*mmproj-Q8_0.gguf"],
        "exclude": ["*f16*"],
        "approx_gb": 3.5,
        "has_mtp": False,
        "desc": "🐋 DeepSeek-V4-Pro-Qwen3.5-4B-heretic (Q5_K_M 3.2GB + mmproj) เร็วมากบน T4",
    },
    # 🎨 โมเดลสร้างภาพ (kind="image") — อยู่ในแคตตาล็อก/ลิสต์เดียวกับโมเดลข้อความ แต่รันด้วย ComfyUI
    "qwen-image-21-uncensored": {
        "repo_id": "sdadfas/Qwen-Image-2.1-Uncensored-GGUF",
        "kind": "image",
        "include": [
            "qwen-image-2.1-UC-Q4_K_M.gguf",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
        "targets": {
            "qwen-image-2.1-UC-Q4_K_M.gguf": "diffusion_models",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors": "text_encoders",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "vae",
        },
        "exclude": None,
        "approx_gb": 14.6,
        "has_mtp": False,
        "desc": "🎨 Qwen-Image-2.1 Uncensored (Q4_K_M + Qwen3VL-8B int8 + VAE) สร้างภาพผ่าน ComfyUI",
    },
    "flux-3-klein-uncensored": {
        "repo_id": "AtomicChat/FLUX-3-Image-Klein-pup-Uncensored-GGUF",
        "kind": "image",
        "include": [
            "FLUX-3-Image-Klein-pup-Uncensored-AD-Q5_K_M.gguf",
            "mmproj-FLUX-3-Image-Klein-pup-Uncensored-F16.gguf",
        ],
        "targets": {
            "FLUX-3-Image-Klein-pup-Uncensored-AD-Q5_K_M.gguf": "diffusion_models",
            "mmproj-FLUX-3-Image-Klein-pup-Uncensored-F16.gguf": "text_encoders",
        },
        "exclude": None,
        "approx_gb": 7.5,
        "has_mtp": False,
        "desc": "⚡ FLUX-3-Image-Klein pup Uncensored (Q5_K_M + encoder) สร้างภาพผ่าน ComfyUI",
    },
    "custom": {"repo_id": CUSTOM_HF_REPO, "include": None, "exclude": None, "approx_gb": 0.0, "has_mtp": False, "desc": "⚙️ กำหนด Repo เอง"},
}

LLAMA_LOG_FILE = Path(os.environ.get("STRATA_LOG_FILE", f"{STRATA_HOME}/llama_server.log"))

DEFAULT_RAG_PATH = os.environ.get("STRATA_RAG_STORE", f"{STRATA_HOME}/rag_store.npz")

# ---------------------------------------------------------------------------
# ตัวช่วยฝั่ง "สร้างภาพ" — โมเดลภาพอยู่ใน MODEL_CATALOG เดียวกัน (kind="image")
# แต่รันด้วย ComfyUI (+custom node ComfyUI-GGUF) ไม่ใช่ llama-server
# ---------------------------------------------------------------------------
COMFY_DIR = Path(os.environ.get("STRATA_COMFY_DIR", f"{STRATA_HOME}/ComfyUI"))

COMFY_PORT = int(os.environ.get("STRATA_COMFY_PORT", "8188"))

COMFY_MODEL_SUBDIRS = ("diffusion_models", "text_encoders", "vae")

IMAGE_PRESET_KEYS: list[str] = [k for k, v in MODEL_CATALOG.items() if v.get("kind") == "image"]

TEXT_PRESET_KEYS: list[str] = [k for k in ALL_PRESET_KEYS if MODEL_CATALOG[k].get("kind") != "image"]
