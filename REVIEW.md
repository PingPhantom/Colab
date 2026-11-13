# รีวิว & แผนปรับปรุง (Strata Studio)

เอกสารนี้สรุปสิ่งที่ตรวจพบในโปรเจกต์ — เดิมเป็นโค้ดก้อนเดียวใน `nix.ipynb` (~1,400 บรรทัด)
ปัจจุบันอยู่ในแพ็กเกจ [`strata/`](./strata) — พร้อมสิ่งที่แก้ไปแล้วและสิ่งที่เหลือ

รันทดสอบซ้ำได้ด้วย:

```bash
python3 tools/smoke_test.py       # โครงสร้าง notebook + ฟังก์ชันหลัก (ไม่ต้องมี Colab)
python3 tools/ui_smoke_test.py    # สร้างหน้าจอจริงด้วย ipywidgets แล้วกดปุ่ม (ต้องมี ipywidgets + numpy)
```

---

## ✅ แก้แล้วในรอบนี้ (P0 + P1 ที่เสี่ยงสูง)

| # | ปัญหาเดิม | ผลกระทบ | วิธีแก้ |
|---|---|---|---|
| 1 | f-string มี backslash ใน `{expression}` (เดิมบรรทัด 750, ป้าย `KV:Q8_0`) | `SyntaxError` ทั้งเซลล์บน Python 3.11 → notebook รันไม่ขึ้นเลย | แยก string ออกจาก expression (`kv_badge = ...`) — ตอนนี้คอมไพล์ผ่านทั้ง 3.11/3.12 |
| 2 | `_pick_main_gguf()` เรียงขนาดแล้วหยิบไฟล์ **เล็กสุด** เสมอ | เลือก quant ผิดจากที่ตั้งใจ (เช่นได้ IQ4_XS ทั้งที่โหลด Q4_K_M ไว้), เงียบเมื่อ shard ขาด | เลือก quant ใหญ่สุดที่ยังอยู่ในงบ VRAM/RAM, รับพารามิเตอร์ `prefer`/`quant`, รายงาน quant ที่พบ, shard ไม่ครบ = `FileNotFoundError` พร้อมบอกว่าขาดกี่ไฟล์ |
| 3 | ลูป fallback ของ speculative decoding ถูกข้ามเมื่อ timeout | `TimeoutError` โยนออกทันที ทำให้ fallback (`ngram`) ไม่มีวันทำงานในเคสที่ช้าที่สุด | `_spawn()` คืน `(False, log)` แทนการ raise + terminate process ที่ค้าง แล้วไปโหมดถัดไปได้จริง (และกันโหมดซ้ำด้วย `dict.fromkeys`) |
| 4 | `ctx_size` จาก `check_gpu_and_recommend()` ไม่เคยถูกใช้ (hardcode 4096) | ใช้ VRAM ไม่คุ้ม โดยเฉพาะ T4 16GB | `ctx_size` เป็น `None` = ใช้ค่าที่แนะนำตาม VRAM, เก็บลง `SPEC_STATE["ctx_size"]`, log ให้เห็น |
| 5 | `download_model_files()` กลืน `RequestException` แล้วไปต่อ | ไม่รู้ว่า HF API ล้ม (404/rate limit) แล้วพังไกลจากต้นเหตุ | ตรวจ `raise_for_status()` และ raise `RuntimeError` พร้อมชื่อ repo, เพิ่ม validation `repo_id` ต้องเป็น `org/model`, ข้ามการโหลดถ้าไฟล์ครบแล้ว |
| 6 | `tarfile/zipfile.extractall()` ไม่มี filter | Path traversal จาก archive ที่ถูกแก้ (ตัวติดตั้ง llama-server) | เพิ่ม `_safe_extract_tar()` / `_safe_extract_zip()` ตรวจปลายทางก่อนแตกไฟล์, py≥3.12 ใช้ `filter="data"` |
| 7 | `pkill -f llama-server` | ฆ่า process อื่นที่ชื่อพ้อง และฆ่า server ที่เพิ่งสตาร์ท | `_kill_server_on_port(port)` ใช้ `pgrep -af` กรองเฉพาะ `--port <port>` + `os.kill(SIGTERM)` |
| 8 | ต่อคำสั่ง shell ด้วย string interpolation (`uv_add`, `download-deepseek`) | Shell injection ผ่านชื่อแพ็กเกจ/glob ที่ผู้ใช้กรอก | `shlex.quote()` ทุกอาร์กิวเมนต์ + whitelist ชื่อแพ็กเกจด้วย `re.fullmatch` |
| 9 | `remove_model()` ไม่ตรวจ path | ส่ง `../../..` ลบโฟลเดอร์นอก `model_dir` ได้ | ตรวจ `tp.parent == base.resolve()` ก่อนลบ |
| 10 | GitHub Releases API ไม่ตรวจสถานะ | rate limit ทำให้พังด้วย `KeyError`/`AttributeError` ที่อ่านไม่รู้เรื่อง | ตรวจ `status_code`, ข้อความบอกสาเหตุ, เพิ่ม env `LLAMA_CPP_TAG` สำหรับ pin เวอร์ชัน |
| 11 | `max_tokens` (สูงสุด 4096) + history ไม่จำกัด ชนกับ ctx 4096 | request ล้ม/ตัดคำกลางประโยคเมื่อคุยยาว | clamp `max_tokens` ตาม ctx จริง และตัด history ตามงบโทเคน |
| 12 | ไม่มีเทสต์เลย | แก้แล้วพังโดยไม่รู้ตัว | `tools/smoke_test.py`: ตรวจโครง notebook, คอมไพล์ทั้งเซลล์, จับ f-string ที่พังบน 3.11, ทดสอบ `_pick_main_gguf` / safe extract / `uv_add` |

---

## ⚡ แก้แล้วรอบที่ 2 (P2 — ประสิทธิภาพ & ความถูกต้อง)

| # | ปัญหาเดิม | ผลกระทบ | วิธีแก้ |
|---|---|---|---|
| 1 | `_refresh()` สร้าง widget แถวโมเดลใหม่ **ทั้ง 10 แถวทุกครั้ง** + `rglob` + `stat` ทุกไฟล์ + `nvidia-smi` + HTTP | UI กระตุกทุกครั้งที่ส่งแชท ปุ่มที่กำลังกดหลุดสถานะ | สร้างแถวครั้งเดียวแล้วอัปเดตแค่ `value`/`description`/`disabled`; ขนาดโฟลเดอร์ใช้ `_dir_stats()` cache 30 วิ (invalidate หลังติดตั้ง/ลบ); health check cache 1.5 วิ (`_probe_server` แยกไว้, `ttl=0` เมื่อต้องค่าใหม่) |
| 2 | วาด HTML ใหม่ทุก token (`_fmt_reply_html(''.join(buf))`) | O(n²) — ตอบยาว ๆ แล้วกระตุกหนัก | throttle ~100ms หรือเมื่อขึ้นบรรทัดใหม่ + วาดรอบสุดท้ายครั้งเดียว (ทำทั้งแท็บแชทและ Arena) |
| 3 | Arena hardcode `max_tokens=512` และยืม `Temp` จากแท็บแชท | เทียบโมเดลไม่ยุติธรรม ปรับแยกไม่ได้ | เพิ่ม `a_temp_sl` / `a_max_sl` ของ Arena เอง + clamp ตาม ctx จริง |
| 4 | Chunking ตัดตามตัวอักษรดิบ ๆ (ตัดกลางคำ) | คุณภาพ RAG ต่ำ ค้นหาเพี้ยน | `split_chunks()` ตัดที่ขอบย่อหน้า/ประโยคก่อน, overlap เริ่มที่ขอบคำเสมอ, ประโยคยาวเกินค่อยตัดเป็นท่อน |
| 5 | `np.vstack` ทุกครั้งที่เพิ่มเอกสาร (O(n²)) | นำเข้าไฟล์เยอะ ๆ ช้าลงเรื่อย ๆ | เก็บเป็น list ของ matrix แล้ว `np.concatenate` ครั้งเดียวแบบ lazy (`embeddings` property + setter คง API เดิม) |
| 6 | `search()` คืน top-k เสมอ แม้ score ต่ำ/ติดลบ | prompt ปนข้อมูลที่ไม่เกี่ยว | เพิ่ม `min_score=0.05` (ปรับได้) และ `build_augmented_prompt(min_score=...)`, จัด rank ใหม่หลังกรอง |
| 7 | `add_pdf_bytes()` เรียก `extract_text()` ซ้ำ 2 รอบ/หน้า | PDF ยาว ๆ ช้าเป็นเท่าตัว | extract ครั้งเดียว/หน้า + รายงานจำนวนหน้าที่ไม่มีข้อความ |
| 8 | ไม่มี dedup / ไม่จำกัดขนาด | chunk ซ้ำ ถังบวมจน context ล้น | กันซ้ำด้วย SHA-1 ของ `source|text`, `max_chunks=4000`, ข้ามไฟล์เกิน 25 MB |
| 9 | คลัง RAG หายทุกครั้งที่ restart | ต้อง ingest ใหม่ทุก session | `save()`/`load()` เป็น `.npz` (embeddings + chunks) + ปุ่ม 💾 บันทึกคลัง / 📂 โหลดคลังเดิม (ค่าเริ่มต้น `/content/rag_store.npz`) |
| 10 | `--n-gpu-layers` เดาจาก `gb_per_layer=0.5` และสมมติ 48 layer | โมเดล 27B offload ผิดสัดส่วน ใช้ VRAM ไม่คุ้ม | อ่าน `<arch>.block_count` จากหัวไฟล์ GGUF จริง (`_gguf_layer_count`) แล้วคำนวณ layer ที่ใส่ GPU ได้ + log ให้เห็น |

### เทสต์ที่เพิ่มในรอบนี้
- `tools/smoke_test.py` +4 เคส: cache ของ `_dir_stats`, `_gguf_layer_count` (GGUF สังเคราะห์), `StrataRAG.split_chunks` (ความยาว/ลำดับคำ/overlap), dedup + search + save/load
- `tools/ui_smoke_test.py` (ใหม่): สร้างหน้าจอจริงด้วย ipywidgets แล้วตรวจ 4 แท็บ, ปุ่มครบตามจำนวนโมเดล, **กดรีเฟรชซ้ำแล้ว widget ต้องเป็นตัวเดิม**, ปุ่มโหลด/รัน/ลบเรียกฟังก์ชันถูก, และปุ่มบันทึก/โหลดคลัง RAG เขียน-อ่านไฟล์จริง
- `test_performance_patterns`: กันไม่ให้ `_refresh` กลับไปสร้าง widget/`rglob` เอง และกันไม่ให้ถอด throttle/`min_score` ออก

---

## 🏗️ แก้แล้วรอบที่ 3 (P3 — โครงสร้าง, dependency, เอกสาร)

| # | ปัญหาเดิม | วิธีแก้ |
|---|---|---|
| 1 | โค้ด ~1,400 บรรทัดในเซลล์เดียว แก้/ทดสอบ/debug ยาก | แยกเป็นแพ็กเกจ `strata/`: `config` • `utils` • `bootstrap` • `download` • `server` • `rag` • `ui` • `notebook` (+ `templates/flake.nix`) โดยคงพฤติกรรมเดิมไว้ (แยกด้วย AST จากโค้ดที่เทสต์ผ่านแล้ว) |
| 2 | `nix.ipynb` หนัก 611 KB (โค้ดจริง ~70 KB ที่เหลือเป็น output + widget state) | notebook เหลือ **4 เซลล์ ~5.5 KB** (intro → setup → launch → คู่มือ) ล้าง output/widget state ทั้งหมด + เทสต์ดักไม่ให้กลับมา |
| 3 | dependency ลอยตัว (`uv add` ไม่ระบุเวอร์ชัน, `nixpkgs` = nixos-unstable) | `pyproject.toml` ระบุช่วงเวอร์ชันครบ + `uv sync` แทน `uv add`; flake ปักหมุด `NIXPKGS_REV` (nixos-unstable @ 2026-11-11) |
| 4 | ติดตั้ง Nix ด้วย `curl ... \| sh` | ดาวน์โหลดสคริปต์ → ตรวจเนื้อหา + log `sha256` → รองรับ `NIX_INSTALL_SHA256` บังคับ checksum → ค่อยรันจากไฟล์ |
| 5 | ลบ `/usr/local/bin/bash`, `/usr/local/bin/sh` ทิ้ง | เปลี่ยนเป็น `symlink` ไป `/bin/bash`, `/bin/sh` (ไม่ทำลาย runtime ของ Colab) |
| 6 | Dead code (`asyncio`, `TypeVar`, `list_model_files`, `CUSTOM_INCLUDE`, alias `launch_strata`) | ลบออกทั้งหมด + เทสต์ `no dead code` กันกลับมา |
| 7 | Global state กระจัดกระจาย (`LAST_MODEL_DIR`, `LLAMA_SERVER_PROC`) เสี่ยงกดซ้ำ/สตาร์ทซ้อน | `LLAMA_SERVER_PROC` อยู่ใน `server` เท่านั้น, `last_model_dir` ย้ายเข้า `SPEC_STATE`, เพิ่ม `_SERVER_LOCK` / `_INSTALL_LOCK` (non-blocking → แจ้งเตือนแทนการสตาร์ทซ้อน) |
| 8 | `import` ทั้งเซลล์ดึง `fastembed`/`pypdf`/`openai` ทันที (ช้าและพังถ้าไม่ติดตั้ง) | import แบบ lazy ในฟังก์ชัน/`TYPE_CHECKING` → `import strata.server` เร็วและไม่ต้องมี heavy deps |
| 9 | README เป็นแค่ `# Colab` | เขียนใหม่: ฟีเจอร์, quickstart, โครงสร้างโค้ด, ตัวอย่าง API/RAG, ตาราง env var, ข้อกำหนดเครื่อง, การทดสอบ, troubleshooting |
| 10 | path/พอร์ต hardcode ใน `/content` | override ได้ผ่าน `STRATA_PORT`, `STRATA_HOME`, `STRATA_MODEL_DIR`, `STRATA_DRAFT_DIR`, `STRATA_RAG_STORE`, `STRATA_LOG_FILE` |

### เทสต์หลัง refactor
- `tools/smoke_test.py` — 16 เคส: โครง notebook (ต้องไม่มี output/state), คอมไพล์ทุกไฟล์ในแพ็กเกจ, f-string safety, public API/catalog, lazy import, รูปแบบโค้ดที่เคยพัง, perf patterns, dead code, quant picker, safe extract, `uv_add`, dir cache, GGUF parser, RAG chunking/dedup/persist, **server lock guard**
- `tools/ui_smoke_test.py` — import `strata.ui` จริงแล้วสร้าง widget tree, กดรีเฟรชซ้ำ/widget ต้องเป็น object เดิม, กดโหลด/รัน/ลบ, บันทึก-โหลดคลัง RAG
- `ruff check strata tools` — ผ่านทั้งหมด (E702 ถูก ignore เพราะเป็นโค้ดที่ port มาจาก notebook)

### สิ่งที่ยังไม่ได้ทำ (ตั้งใจ)
- RAG ยังไม่มี rerank (cross-encoder) และยังไม่อ่าน PDF ที่เป็นภาพสแกน (ต้อง OCR)
- ยังไม่ได้รัน `Restart & Run All` บน Colab GPU จริงในรอบนี้ (sandbox ไม่มี GPU) — แนะนำรันหนึ่งรอบหลัง merge
- ยังไม่ได้รัน `ruff format` ทั้งโปรเจกต์ (จะเกิด diff ยักษ์ในโค้ดที่ port มา) — ทำแยกได้เมื่อต้องการ

---

## 📶 เพิ่มเติมตามคำขอ: Network bandwidth (DL/UL)

- `strata/utils.py` — `NetworkMonitor` (คำนวณความเร็วจาก delta ของ `/proc/net/dev`, ทนตัวนับรีเซ็ต), `net_bandwidth()`, `_fmt_rate()` (`812 KB/s`, `4.21 MB/s`)
- `strata/ui.py` — ชิป ⬇/⬆ ในหัว UI (มี tooltip ยอดรวม) และแผงวัดเรียลไทม์ในแท็บ 1 (ปุ่ม 📶 เปิด/หยุด, thread อัปเดตทุก 1 วิ, ปิดเองเมื่อกดหยุด/kernel ตาย)
- เทสต์: `tools/smoke_test.py::test_network_monitor` (delta/รีเซ็ต/ฟอร์แมต) และ `tools/ui_smoke_test.py` (ชิปใน header + toggle เปิด/ปิด)

---

## 🔧 แก้เพิ่ม: Colab เปิด notebook เดี่ยว ๆ แล้ว `ModuleNotFoundError: No module named 'strata'`

เดิมทั้งเซลล์รวมโค้ดไว้ในตัว notebook แต่หลังแยกเป็นแพ็กเกจ `strata/` การเปิด notebook จาก GitHub
ใน Colab จะไม่มีโฟลเดอร์โค้ดอยู่ในเครื่อง (cwd = `/content`) จึง import ไม่เจอ

- เซลล์ 1 มี `_ensure_repo()`: หา `strata/__init__.py` จาก cwd ขึ้นไป ถ้าไม่พบจะ `git clone --depth 1`
  ไปที่ `/content/Colab` (ลบของค้างก่อน), `chdir` เข้ารากรีโป แล้วจึง import `strata.bootstrap`
- เพิ่มช่อง `REPO_SLUG` / `REPO_BRANCH` (Colab form) สำหรับ fork หรือทดสอบ branch/PR
- ถ้า clone แล้วยังไม่เจอ `strata/` (เช่น branch หลักยังไม่ merge) จะได้ error บอกให้ใส่ `REPO_BRANCH`
- เซลล์ 2 มี guard: ถ้ารันข้ามเซลล์จะบอกให้รันเซลล์ 1 ก่อน
- เทสต์: `test_notebook_repo_bootstrap` (มีโค้ดอยู่แล้ว / ต้อง clone / ลำดับ fallback / ของค้างถูกลบ / error บอกทางออก)

### v4: fallback ไป branch ของ PR อัตโนมัติ
อาการจริงที่เจอ: `REPO_BRANCH` ว่าง → clone `main` → `main` ยังไม่มี `strata/` (PR ยังไม่ merge) → `RuntimeError`
- `_pr_branches()` ถาม `https://api.github.com/repos/{slug}/pulls?state=open` (timeout 10s, ล้มเหลว = เงียบ ๆ ใช้ branch หลัก)
- `_ensure_repo()` ไล่ลำดับ: branch ที่ผู้ใช้ระบุ → branch หลัก → branch ต้นทางของ PR ที่เปิดอยู่
  โดยลบโฟลเดอร์ปลายทางก่อน clone ทุกครั้ง และหยุดที่ตัวแรกที่มี `strata/__init__.py`
- ถ้าทุกตัวไม่มีโค้ด → `RuntimeError` บอกให้ใส่ `REPO_BRANCH`; ทดสอบกับ GitHub จริงแล้วไล่ไปเจอ `arena/e076c6f4-colab` เอง
- `NOTEBOOK_REV = "v4-pr-fallback"` พิมพ์เป็นบรรทัดแรก เพื่อแยกเซลล์เก่า/ใหม่ได้ทันที

---

## 🐞 P0 (เจอตอนรันบน Colab จริง): `.gitignore` กว้างเกิน ทำ template หายจาก clone

`FileNotFoundError: strata/templates/flake.nix` ตอน `ensure_environment()` — ต้นตอคือ `.gitignore`
เขียนว่า `flake.nix` (ไม่มี `/` นำหน้า) ซึ่ง match **ทุกไฟล์ชื่อ flake.nix ในทุกโฟลเดอร์**
รวมถึง template ของแพ็กเกจ → ไฟล์ไม่เคยถูก commit แต่เครื่อง dev มีอยู่ เทสต์เดิมจึงผ่านหมด

- เปลี่ยนเป็น pattern ที่ยึดรากรีโป: `/.venv/`, `/flake.nix`, `/flake.lock`, `/default.nix`, `/requirements.txt`
- commit `strata/templates/flake.nix` (ไฟล์ที่หายไปตั้งแต่ตอนแยกแพ็กเกจ)
- เทสต์ใหม่ 2 ตัว:
  - `package files tracked by git`: ไฟล์ทุกตัวใต้ `strata/` ต้องอยู่ใน `git ls-files` และต้องไม่ถูก `check-ignore` บัง
  - `flake.nix from template`: อ่าน template ได้จริง + แทน placeholder ครบ + ยังคง `"$@"` ของ `hf download`
- บทเรียน: เทสต์บนเครื่อง dev ผ่านหมดไม่ได้แปลว่า clone มาจะรันได้ — ต้องมีเทสต์ที่ validated ว่าไฟล์ถูก track

---

## 🩹 P0 (รอบ 2): โค้ดเก่าค้างใน clone เดิม + เทมเพลตหาย

หลังแก้ `.gitignore` แล้ว ผู้ใช้ยังเจอ `FileNotFoundError` เดิม เพราะ `/content/Colab` เป็น clone เก่า
ที่ยังไม่มีไฟล์นั้น และเซลล์ 1 เวอร์ชันก่อน ๆ ไม่ยอมอัปเดตมัน

- **เทมเพลตย้ายเข้าโค้ด**: `strata/flake_template.py` เก็บ flake เป็นสตริง — ลบโฟลเดอร์ `strata/templates/`
  ออกทั้งอัน (ไม่มีไฟล์ภายนอกให้หลุด track หรือหายตอน clone อีก), `bootstrap.FLAKE_TEMPLATE` เป็น `str` แล้ว
- **เซลล์ 1 v6-auto-update**: ถ้ามี `/content/Colab` เป็น git repo → `git fetch --depth 1` + `git reset --hard FETCH_HEAD`
  ก่อนเสมอ (ref: branch ปัจจุบัน → branch หลัก → branch ของ PR), ปิดได้ด้วย `AUTO_UPDATE=False`
  แล้วค่อย fallback ไป clone ใหม่ถ้าอัปเดตไม่ได้
- เทสต์เพิ่ม: ลำดับ ref ของ `_refresh`, กรณี fetch ล้มทุก ref → คืน `False` ให้ไป clone, และ `AUTO_UPDATE=False` ต้องไม่ fetch
- ทดสอบกับ git จริง (จำลอง clone เก่าโดยลบไฟล์): `🔄 อัปเดตโค้ดเดิมเป็น branch ล่าสุด เรียบร้อย` → ไฟล์กลับมา → `write_flake()` ผ่าน

---

## 🐞 P0 (รอบ 3): `download-deepseek: command not found` ตอนโหลดโมเดล

`download_model_files()` เรียก `download-deepseek` ตรง ๆ ผ่าน `/bin/bash -c` ซึ่งเป็นสคริปต์ที่มาจาก
`nix develop` devshell — ถ้าเซลล์ 1 ไม่ได้จบ (หรือ PATH ของ devshell หลุด) จะได้ error ของ bash ที่แก้ไม่ได้ด้วยตัวเอง

- `download_command()` เลือกเครื่องมือตามที่หาเจอ: `download-deepseek` → `hf` CLI → `[]`
- `download_with_module()` ใช้ `huggingface_hub.snapshot_download` (มี `allow_patterns`/`ignore_patterns`)
  เป็นทางสุดท้ายที่ไม่ต้องพึ่ง PATH — ถ้าไม่มีเลยจะได้ `RuntimeError` ที่บอกให้รัน `ensure_environment()`
- `huggingface-hub[hf_transfer]` ถูกเพิ่มใน dependency เพื่อให้ทางสำรองนี้มีเสมอ
- `bootstrap.missing_tools()` ตรวจเครื่องมือ devshell (`uv/hf/download-deepseek/jq/rg/ruff`) แล้ว
  เตือนตอน `ensure_environment()` พร้อมคืนค่าใน `missing_tools` ของ dict สรุป
- เทสต์ `downloader fallback`: ลำดับ wrapper → hf → โมดูล, การส่ง pattern, และข้อความ error เมื่อไม่มีอะไรเลย

---

## ✨ ปรับ UX: เดา repo/branch อัตโนมัติ (เซลล์ 1 v8)

ผู้ใช้ไม่ควรต้องพิมพ์ `arena/e076c6f4-colab` เองทุกครั้ง

- `_parse_github_url()` แยก `(owner/repo, branch)` จาก URL ของ Colab (`.../github/OWNER/REPO/blob/<branch>/x.ipynb`)
  รองรับ branch ที่มี `/`, `%2F` (URL-encode), query string และคืน `("", "")` เมื่อไม่ใช่ลิงก์ GitHub
- `_detect_source()` ลำดับการเดา: URL ของแท็บ (ผ่าน `google.colab.output.eval_js`) → `git rev-parse --abbrev-ref HEAD`
  ของ `/content/Colab` หรือ `./Colab` หรือ cwd → ไม่ได้ก็ปล่อยว่าง (แล้วใช้ fallback เดิม: branch หลัก → PR ที่เปิดอยู่)
- ฟอร์ม: `REPO_SLUG`/`REPO_BRANCH` ปล่อยว่างได้, `DEFAULT_SLUG` เป็นค่าสุดท้าย, พิมพ์ `🌿 ใช้ <slug> @ <branch>` ให้เห็นว่าเลือกอะไร
- เทสต์ `branch auto-detect`: parse 4 รูปแบบ URL + เคสไม่มี URL (ถอยไปอ่าน git ของ clone)
- ทดสอบจริง: วาง repo จำลองที่ `content/Colab` แล้วเรียก `_detect_source()` → ได้ `('', 'arena/e076c6f4-colab')`

---

## 🐞 P0 (รอบ 4): `nix develop` exit 1 — flake อยู่ใน git repo

หลังเซลล์ setup `chdir` เข้า `/content/Colab` (ซึ่งเป็น git repo) `write_flake()` จึงเขียน `flake.nix` ที่รากรีโป
แต่ไฟล์นั้นถูก `.gitignore` ไว้ = **untracked** → Nix มองไม่เห็น (`path ... is not tracked by Git`) และเพราะ
`activate_devshell()` กลืน stderr ทิ้ง ผู้ใช้จึงเห็นแค่ `exit status 1`

- `FLAKE_DIR = $STRATA_HOME/flake` (นอก repo) — `write_flake()` เขียนที่นั่น, เก็บกวาด `flake.nix`/`default.nix`
  ที่เคยสร้างใน cwd (เฉพาะไฟล์ที่มี marker `Strata Studio devshell`, ไม่แตะ flake ของผู้ใช้)
- `activate_devshell()` ใช้ `subprocess.run(capture_output=True)` แล้ว raise ข้อความที่มี **15 บรรทัดท้ายของ stderr**
  + วิธีแก้ 3 ข้อ; สำเร็จแล้วคืน `{"flake_dir", "vars"}`
- `nix_diagnostics()` รวม `nix --version`, `experimental-features`, การมี `/nix/store`, ที่อยู่ flake —
  `ensure_environment()` พิมพ์ให้อัตโนมัติเมื่อ devshell ล้ม และคืน `flake_dir` ใน dict สรุป
- `config._default_home()`: ใช้ `/content` เมื่อมี (Colab) ไม่งั้น `~/.strata` (เครื่อง dev ไม่ต้องมีสิทธิ์เขียน `/`)
- เทสต์: `flake lives outside git repo` (เขียนนอก repo + ลบของเดิม + ไม่แตะไฟล์ผู้ใช้), `nix develop error surfacing`
  (แนบ stderr/hint, merge PATH/env สำเร็จ)

---

## 🐞 P0 (รอบ 5): llama-server ตายเพราะ glibc ของ Nix ชนกับของระบบ

`llama-server` (prebuilt binary) รันด้วย `LD_LIBRARY_PATH` ที่มี `/nix/store/…/glibc-2.44/lib` ต่อท้าย
→ `libdl.so.2`/`libpthread.so.0` ของ Nix ไปขอ symbol `GLIBC_ABI_DT_X86_64_PLT` จาก `libc.so.6` ของ Ubuntu 22.04 (2.35)
ที่ไม่มีให้ → `not found` ทุกโหมด fallback

- `sanitize_ld_library_path()` + `server_env()`: ตัด path ใน `/nix/store` ออก, ถอด `LD_PRELOAD` ที่ชี้เข้า store
  แล้วส่ง `env=server_env()` ให้ `subprocess.Popen`
- wrapper `/usr/local/bin/llama-server` เวอร์ชันใหม่ (`WRAPPER_REV`) กรอง `/nix/store` ใน bash ด้วย
  (ใช้ `printf "%s\n"` ก่อน `tr` เพื่อไม่ให้บรรทัดสุดท้ายหายจาก gotcha ของ `read`)
  และติดตั้งทับ wrapper เก่าอัตโนมัติถ้ายังหาไบนารีเดิมเจอ (ไม่ต้องโหลดซ้ำ)
- `start_error_hint()` แนบคำใบ้ท้าย error: glibc / CUDA / อื่น ๆ
- เทสต์: `prebuilt libc isolation (glibc)` — sanitize, server_env, hint และรัน wrapper จริงกับไบนารีปลอม
  ยืนยันว่า `/nix/store` ไม่อยู่ใน env ของ process ลูกและ `LD_PRELOAD` ถูกถอด

---

## 📌 สรุปสถานะ

P0 (บั๊กที่ทำให้รันไม่ขึ้น), P1 (ความปลอดภัย), P2 (ประสิทธิภาพ/คุณภาพ RAG) และ P3 (โครงสร้าง/dependency/เอกสาร)
ถูกแก้ครบแล้วใน 3 คอมมิต ส่วนที่เหลือเป็นงานต่อยอด (rerank/OCR, format ทั้งโปรเจกต์, รันจริงบน Colab GPU)

## หมายเหตุการทดสอบ
- ทดสอบกับ Python 3.11.2 (sandbox): `tools/smoke_test.py` 16/16 ผ่าน, `tools/ui_smoke_test.py` 5/5 ผ่าน, `ruff check` สะอาด
- ยังไม่ได้รันบน Colab GPU จริง (sandbox ไม่มี GPU) — แนะนำ `Restart & Run All` หนึ่งรอบหลัง merge
