# 🎛️ Strata Studio

<a href="https://colab.research.google.com/github/PingPhantom/Colab/blob/main/nix.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

จัดการโมเดล **GGUF** บน [llama.cpp](https://github.com/ggml-org/llama.cpp) พร้อม **StrataRAG** (คลังความรู้จาก PDF/ข้อความ)
และ UI 4 แท็บสำหรับ Google Colab — ใช้ GPU T4 ฟรีก็รันได้

> โค้ดทั้งหมดอยู่ในแพ็กเกจ [`strata/`](./strata) ส่วน `nix.ipynb` เหลือแค่เซลล์ติดตั้ง + เปิด UI

---

## ✨ ความสามารถ

- **12 โมเดล preset + custom repo** — ดาวน์โหลด/ลบ/เก็บตัวเดียว, เช็กดิสก์และ VRAM ก่อนโหลด
  (รวม `orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF` — MoE ~225GB สำหรับ A100/H100)
- **Speculative decoding** — MTP, external draft model, N-gram พร้อม fallback อัตโนมัติเมื่อโหมดใดโหลดไม่ได้
- **เลือก quant อัตโนมัติ** — อ่านจำนวน layer จริงจากหัวไฟล์ GGUF แล้วคำนวณ `--n-gpu-layers` ให้พอดี VRAM
- **StrataRAG** — chunk ตามประโยค, กันข้อมูลซ้ำ, ค้นหาแบบ cosine + `min_score`, บันทึก/โหลดคลังเป็น `.npz`
- **UI 4 แท็บ** — โมเดล&ดิสก์ • แชท+RAG • คลัง PDF • เปรียบเทียบหลายโมเดล (Arena)
- **วัดความเร็วเน็ต DL/UL** — แสดงในหัว UI (⬇/⬆ พร้อมยอดรวม) และเปิดแผงวัดเรียลไทม์ (อัปเดตทุก 1 วิ) ในแท็บ 1
- **API แบบ OpenAI-compatible** ที่ `http://127.0.0.1:18080/v1`
- **ไม่แตะไฟล์ระบบของ Colab** — ใช้ symlink แทนการลบ, ตรวจสคริปต์ติดตั้ง Nix ก่อนรัน, แตก archive อย่างปลอดภัย

### ⏱ แสดงผลแบบเรียลไทม์
- หัว UI (สถานะ server, ดิสก์, VRAM, RAG, ⬇/⬆) อัปเดตเองทุก **2 วินาที** ไม่ต้องกดรีเฟรช
- แผง `⏱ เรียลไทม์` ท้ายแท็บ 1 แสดง log สด: คำสั่งที่รันผ่าน `run_stream()` + ท้าย log ของ `llama-server`
- ทุก ~6 วินาทีจะเดินดิสก์/ขนาดโมเดลใหม่ (ใช้ `_dir_stats()` แบบ cache)

---

## 🚀 เริ่มใช้ (Colab)

1. เปิด [`nix.ipynb`](./nix.ipynb) ใน Colab (เลือก Runtime → GPU) — เปิดไฟล์เดี่ยว ๆ ได้เลย
   เซลล์แรก **เดา repo + branch ให้เอง** (จาก URL ของ Colab → git ของ clone เดิม → branch ของ PR ที่เปิดอยู่)
   และ **clone/อัปเดตโค้ดให้อัตโนมัติ** (`AUTO_UPDATE=True` = `git fetch --depth 1` + `reset --hard` ให้ตรง GitHub ทุกครั้ง)
   — จะระบุเองก็ได้โดยเติม `REPO_SLUG`/`REPO_BRANCH` ในฟอร์ม
2. รันเซลล์ **1) ติดตั้งสภาพแวดล้อม** — ติดตั้ง Nix (ปักหมุด nixpkgs), devshell, และ `.venv` ตาม `pyproject.toml`
3. รันเซลล์ **2) เปิด Studio UI** — เปิด UI ที่พอร์ต `18080` (ไม่ชนกับพอร์ต 8080 ของ Colab)
4. แท็บแรก → กด **📥 โหลด** โมเดลที่ต้องการ → **🚀 รัน & แชท** → ไปคุยที่แท็บ 2

> **เจอ `ModuleNotFoundError: No module named 'strata'` อยู่?**
> แปลว่าเซลล์ที่รันเป็นเวอร์ชันก่อนแยกแพ็กเกจ (เซลล์ใหม่จะมีข้อความ `🧩 strata notebook v3-auto-clone`)
> ปิด runtime แล้วเปิด notebook ใหม่จากลิงก์ด้านล่าง หรือถ้าอยากใช้เซสชันเดิมต่อ ให้วาง 3 บรรทัดนี้ในเซลล์ใหม่:
>
> ```python
> %cd /content
> !git clone --depth 1 --branch arena/e076c6f4-colab https://github.com/PingPhantom/Colab.git Colab 2>/dev/null || git -C Colab pull --ff-only
> import os, sys; os.chdir("/content/Colab"); sys.path.insert(0, "/content/Colab")
> ```
>
> แล้วรันเซลล์ 1 ต่อได้เลย (ขั้นตอนที่ทำไปแล้วจะถูกข้าม)

> **ติด ๆ กัน: เซลล์ที่รันอยู่เป็นเวอร์ชันเก่า (banner ไม่ใช่ `v4-pr-fallback`)?**
> ไม่ต้องโหลด notebook ใหม่ — วาง 9 บรรทัดนี้ในเซลล์ใหม่แทนเซลล์ 1 ได้เลย:
>
> ```python
> %cd /content
> !rm -rf /content/Colab
> !git clone --depth 1 --branch arena/e076c6f4-colab https://github.com/PingPhantom/Colab.git /content/Colab
> import os, sys
> os.chdir("/content/Colab")
> sys.path.insert(0, os.getcwd())
> from strata.bootstrap import ensure_environment
> env = ensure_environment()
> print("✅ nixpkgs", env["nixpkgs"][:7])
> ```
>
> แล้วเซลล์ 2 ใช้แค่ 3 บรรทัด: `from strata.notebook import launch, register_magics` → `register_magics()` → `ui = launch()`
>
> ⚠️ หลังจากวาง snippet แล้ว **อย่ารันเซลล์ 1 เวอร์ชันเก่าซ้ำ** — มันจะลบ `/content/Colab` แล้ว clone branch หลักทับ (ถ้าจำเป็นให้ลบเซลล์นั้นทิ้ง)

> ยังไม่ merge PR ก็ใช้ได้: เว้น `REPO_BRANCH` ว่างไว้ เซลล์ 1 จะหา branch ที่มีแพ็กเกจ `strata/` เอง
> (ถาม GitHub API ว่า PR ไหนเปิดอยู่ แล้วไล่ `git clone --branch` ทีละตัว)
> หรือเปิดตรง ๆ: `https://colab.research.google.com/github/PingPhantom/Colab/blob/arena/e076c6f4-colab/nix.ipynb`

ใช้ในเครื่องตัวเอง (Linux + Nix):

```bash
nix develop                 # ได้ python + uv + hf + ruff + download-deepseek
uv sync                     # ติดตั้ง dependency ตาม pyproject.toml
jupyter lab                 # แล้วเปิด nix.ipynb
```

---

## 🧩 โครงสร้างโค้ด

```
strata/
├── config.py       # ค่าคงที่: พอร์ต, path, MODEL_CATALOG, DRAFT_CATALOG, SPEC_STATE (ไม่พึ่ง lib ภายนอก)
├── bootstrap.py    # เตรียม Nix + flake (ปักหมุด rev) + uv venv ให้ kernel ใช้ได้ทันที
├── utils.py        # รัน shell อย่างปลอดภัย, cache ขนาดโฟลเดอร์, แตก tar/zip อย่างปลอดภัย
├── download.py     # ดาวน์โหลด/ติดตั้ง/ลบโมเดล + draft model
├── server.py       # ติดตั้ง llama.cpp, เลือก quant/layer, สตาร์ท/หยุด server, OpenAI client
├── rag.py          # StrataRAG: chunking, embedding, search, persist
├── ui.py           # ipywidgets UI 4 แท็บ
├── notebook.py     # ตัวช่วยใน notebook: register_magics(), launch()
└── templates/
    └── flake.nix   # devshell template (แทนที่ rev/py version ตอน bootstrap)
tools/
├── smoke_test.py      # เทสต์โครงสร้าง + ฟังก์ชันหลัก (ไม่ต้องมี GPU/Colab)
└── ui_smoke_test.py   # สร้าง UI จริงด้วย ipywidgets แล้วกดปุ่ม
```

### ตัวอย่างเรียกใช้ตรง ๆ

```python
from strata.download import install_model, remove_model
from strata.server import start_strata_server, stop_strata_server, get_openai_client

install_model("qwen35-9b-defiant-fable", auto_start_server=True)
client = get_openai_client()                    # OpenAI-compatible ที่ :18080
print(client.chat.completions.create(
    model="qwen35-9b-defiant-fable",
    messages=[{"role": "user", "content": "สวัสดี"}],
).choices[0].message.content)

stop_strata_server()
```

```python
from strata.utils import net_bandwidth, _fmt_rate

net = net_bandwidth()                      # อ่านจาก /proc/net/dev (Linux/Colab)
print(f"⬇ {_fmt_rate(net['down_bps'])} ⬆ {_fmt_rate(net['up_bps'])}"
      f" (รวม ⬇{net['rx_bytes'] / 1024**2:.0f} MB)")
```

```python
from strata.rag import GLOBAL_RAG

GLOBAL_RAG.add_files("/content/document.pdf", chunk_size=500)
GLOBAL_RAG.save("/content/rag_store.npz")
prompt, docs = GLOBAL_RAG.build_augmented_prompt("สรุปสาระสำคัญให้หน่อย", top_k=3)
```

---

## ⚙️ ตั้งค่าผ่าน environment variable

| ตัวแปร | ค่าเริ่มต้น | ใช้ทำอะไร |
| --- | --- | --- |
| `STRATA_PORT` | `18080` | พอร์ต API server / UI |
| `STRATA_HOME` | `/content` | โฟลเดอร์เก็บ log และคลัง RAG |
| `STRATA_MODEL_DIR` | `$STRATA_HOME/models` | ที่เก็บโมเดล GGUF |
| `STRATA_DRAFT_DIR` | `$STRATA_MODEL_DIR/_drafts` | ที่เก็บ draft model |
| `STRATA_RAG_STORE` | `$STRATA_HOME/rag_store.npz` | ไฟล์คลัง RAG |
| `STRATA_LOG_FILE` | `$STRATA_HOME/llama_server.log` | log ของ llama-server |
| `LLAMA_CPP_TAG` | release ล่าสุด | ปักหมุดเวอร์ชัน llama.cpp (เช่น `b7000`) |
| `NIX_INSTALL_SHA256` | – | บังคับ checksum ของสคริปต์ติดตั้ง Nix |
| `HF_REPO_ID` | repo ตาม catalog | override repo ตอนเรียก `download-deepseek` |

---

## 💻 ข้อกำหนดเครื่อง

| รายการ | ขั้นต่ำ | แนะนำ |
| --- | --- | --- |
| GPU | T4 16 GB (Colab ฟรี) | L4 / A100 |
| RAM | 12 GB | 25 GB+ |
| ดิสก์ | 20 GB (โมเดล 9B ~6.4 GB) | 100 GB+ ถ้าโหลดหลายตัว |
| Python | 3.10+ (Colab 3.11/3.12) | 3.12 |

โมเดล 27B–35B (IQ4_XS ~15 GB) ต้องใช้ `--n-gpu-layers` บางส่วนหรือ KV cache แบบ Q8 ซึ่งโค้ดเลือกให้อัตโนมัติ

---

## 🧪 เทสต์

```bash
pip install numpy ipywidgets requests rich   # dependency ที่เทสต์ต้องใช้
python3 tools/smoke_test.py       # โครงสร้าง + ฟังก์ชันหลัก (chunking, quant picker, GGUF parser, ...)
python3 tools/ui_smoke_test.py    # สร้าง widget tree จริงแล้วกดปุ่ม (รีเฟรช/โหลด/RAG save-load)
ruff check strata tools           # lint
```

เทสต์ไม่ต้องใช้ GPU และไม่รัน side effect (ไม่ติดตั้ง Nix / ไม่ดาวน์โหลดโมเดล)

---

## 🛠️ แก้ปัญหาที่พบบ่อย

| อาการ | สาเหตุ/วิธีแก้ |
| --- | --- |
| `FileNotFoundError: strata/templates/flake.nix` | แก้แล้ว: เทมเพลตย้ายไปอยู่ในโค้ด (`strata/flake_template.py`) ไม่มีไฟล์ภายนอกให้หายอีก — ถ้า clone เดิมยังพัง ให้รันเซลล์ 1 v6 (จะ `git reset --hard` ให้เอง) |
| `RuntimeError: clone ... แล้วไม่พบแพ็กเกจ strata/` | ทุก branch ที่ลองไม่มีโค้ด — ใส่ชื่อ branch ในช่อง `REPO_BRANCH` ให้ตรง (เช่น branch ของ PR) หรือ merge PR เข้า branch หลัก |
| อยากใช้ branch อื่น (frok/PR) | เติม `REPO_SLUG=owner/repo` และ `REPO_BRANCH=ชื่อ branch` ในฟอร์มเซลล์ 1 — เว้นว่าง = เดาให้ |
| `the argument has been removed` (`--draft-max`) | llama.cpp รุ่นใหม่เปลี่ยนเป็น `--spec-draft-n-max`/`--spec-ngram-mod-n-max` — โค้ดอ่าน `--help` ของไบนารีแล้วเลือกชื่อให้เอง และถ้ายังพังจะลองสตาร์ทแบบปิด spec ให้ |
| ปุ่มไม่ตอบหลังโหลดโมเดลเสร็จ | เดิม `_act()` ปิดปุ่มทั้งหมดแล้วไม่เปิดคืน (แก้แล้ว: `_set_busy()` เปิดคืนใน `finally`) — ถ้าเจอในเซสชันเก่าให้รีเฟรชหน้าเว็บ/รันเซลล์ 2 ใหม่ |
| VRAM เต็มทั้งที่ server Offline | process `llama-server` รอบก่อนยังไม่คืนหน่วยความจำ — โค้ดใหม่เก็บกวาดให้อัตโนมัติ (`release_gpu_memory()`) และบอกว่าใครถือ VRAM อยู่ (`gpu_memory_users()`); ถ้ายังไม่พอให้ Runtime → Restart session |
| `version 'GLIBC_ABI_DT_X86_64_PLT' not found` ตอนสตาร์ท llama-server | ไบนารี prebuilt ไปหยิบ glibc ของ Nix — โค้ดใหม่ตัด `/nix/store` ออกจาก `LD_LIBRARY_PATH` ใน wrapper + `server_env()` ให้แล้ว (ลบ `/usr/local/bin/llama-server` แล้วติดตั้งใหม่ถ้าเป็น wrapper เก่า) |
| `nix: command not found` ใน terminal/shell ใหม่ | `ensure_environment()` เขียน `/etc/profile.d/strata-nix.sh` + `~/.bashrc` และ symlink คำสั่ง nix ไป `/usr/local/bin` ให้แล้ว — เปิด shell ใหม่ หรือ `source ~/.bashrc`; ถ้ายังไม่เห็นให้ใช้พาธเต็ม `/nix/var/nix/profiles/default/bin/nix` |
| `nix develop` ล้มเหลว (exit 1) | ข้อความ error จะแนบ stderr ท้ายให้แล้ว + พิมพ์ `nix_diagnostics()`; สาเหตุพบบ่อย: flake อยู่ใน git repo ที่ยังไม่ track ไฟล์ (โค้ดใหม่เขียน flake ไปที่ `$STRATA_HOME/flake` แล้ว) |
| `/bin/bash: download-deepseek: command not found` | devshell ยังไม่ถูก activate — รัน `from strata.bootstrap import ensure_environment; ensure_environment()` แล้วดู `env['missing_tools']` (โค้ดใหม่จะถอยไปใช้ `hf`/`huggingface_hub` ให้เอง) |
| `ModuleNotFoundError: No module named 'strata'` | เกิดเมื่อรันเซลล์ที่ 2 ก่อนเซลล์ที่ 1 หรือ clone ไม่สำเร็จ — รันเซลล์ 1 ใหม่ (จะมีข้อความ `📂 ใช้โค้ดจาก: ...`) และถ้าทดสอบ branch/PR ให้ใส่ชื่อ branch ในช่อง `REPO_BRANCH` |
| `ติดตั้ง Nix` ค้าง/ช้า | Colab บางรอบเน็ตช้า ลองรันเซลล์ 1 ใหม่ (ขั้นตอนที่ทำแล้วจะถูกข้าม) |
| `nix develop` ฟ้อง hash mismatch | ลบ `flake.lock` แล้วรันใหม่ หรือ `nix flake update` |
| `Server พอร์ต 18080 ยังไม่เปิด` | กด 🚀 รัน & แชท ที่แท็บ 1 (ต้องมีไฟล์โมเดลก่อน) |
| โหลดโมเดลแล้วช้า/ค้าง | ดู log ที่ `/content/llama_server.log` หรือปุ่ม 📡 สถานะ |
| RAG ตอบไม่ตรง | เพิ่ม `chunk_size`/`top_k` หรือลด `min_score` ใน `GLOBAL_RAG.search()` |
| ดิสก์เต็ม | แท็บ 1 → 🧹 ล้างแคช / 🗑️ ลบโมเดลที่ไม่ใช้ |

---

## 📄 เอกสารเพิ่มเติม

- [`REVIEW.md`](./REVIEW.md) — บันทึกการรีวิว/แก้ไข (P0–P2) และสิ่งที่ยังเหลือ
- [`nix.ipynb`](./nix.ipynb) — notebook สำหรับ Colab

## License

MIT
