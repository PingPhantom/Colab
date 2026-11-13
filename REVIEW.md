# รีวิว & แผนปรับปรุง `nix.ipynb` (Strata Studio)

เอกสารนี้สรุปสิ่งที่ตรวจพบใน `nix.ipynb` (เซลล์โค้ดเดียว ~1,150 บรรทัด), สิ่งที่ **แก้แล้ว** ในรอบนี้
และสิ่งที่ยัง **เหลือ** ให้พิจารณา

รันทดสอบซ้ำได้ด้วย:

```bash
python3 tools/smoke_test.py
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

## 🔜 ยังเหลือ (แนะนำลำดับถัดไป)

### P2 — ประสิทธิภาพ & ความถูกต้อง
- **`_refresh()` แพงเกินไป**: ทุกครั้งที่ส่งข้อความหรือกดปุ่ม จะ `rglob` + `stat` ไฟล์ GGUF ทั้ง 10 โมเดล,
  เรียก `nvidia-smi` และยิง HTTP health check → ควร cache ขนาดโฟลเดอร์ (คำนวณใหม่เมื่อติดตั้ง/ลบ) และยิง health check แบบ debounce
- **Streaming เรนเดอร์ O(n²)**: ทุก token เรียก `_fmt_reply_html(''.join(buf))` ใหม่ทั้งก้อน → อัปเดตทุก ~100ms หรือเมื่อเจอ paragraph break
- **Arena**: `max_tokens=512` hardcode และใช้ `temp_sl` ของแท็บแชทข้ามแท็บ → แยก state ต่อแท็บ
- **RAG**: chunk ตัดกลางคำ (ควร split ตามย่อหน้า/ประโยคก่อน), `np.vstack` ทุกครั้งที่เพิ่มเอกสาร (O(n²)) ควรใช้ list แล้วค่อย concat,
  `search()` ไม่มี score threshold, `add_pdf_bytes` เรียก `extract_text()` ซ้ำ 2 รอบต่อหน้า, ไม่มี dedup/จำกัดขนาดไฟล์
- **RAG ไม่ persist**: restart แล้ว embeddings หายหมด → dump `chunks` + embeddings ลง `npz`/json หรือ ChromaDB บน Drive
- **`gb_per_layer=0.5`** หยาบเกินสำหรับโมเดล 27B (จริง ~0.24 GB/layer) → คำนวณจากขนาดไฟล์ / จำนวน layer แทนค่าคงที่

### P3 — โครงสร้าง & repo hygiene
- **แยกเซลล์/โมดูล**: 1,150 บรรทัดในเซลล์เดียว debug ยาก → แยกเป็น `strata/core.py` (config, download, server), `strata/rag.py`, `strata/ui.py`
  แล้ว notebook เหลือแค่ `!pip install -e .` + `from strata.ui import show_model_manager_ui`
- **Notebook บวม**: ไฟล์ 611 KB แต่โค้ดจริง ~70 KB — ที่เหลือคือ output + widget state (329 KB) ที่ค้างจากรันครั้งก่อน
  → `jupyter nbconvert --clear-output --inplace nix.ipynb` ก่อน commit (ตั้งใจเว้นไว้ในรอบนี้เพราะยังไม่ได้รัน UI ใหม่)
- **Pin dependency**: `uv add` ไม่ระบุเวอร์ชัน, `nixpkgs` ชี้ `nixos-unstable` → ควรมี `pyproject.toml` ที่ pin และ pin nixpkgs rev
- **`curl | sh` สำหรับ Nix**: ยังไม่ได้ตรวจ checksum (ควรดาวน์โหลดสคริปต์มาแล้ว verify ก่อนรัน)
- **ลบ `/usr/local/bin/bash` / `/usr/local/bin/sh`**: อันตรายกับ runtime ของ Colab — ควรใช้ `ln -sf` ทับแทนการลบ
- **Dead code**: `asyncio`, `TypeVar T`, `list_model_files()`, `CUSTOM_INCLUDE` (เป็น `None` ตลอด)
- **Global state**: `SPEC_STATE`, `LLAMA_SERVER_PROC`, `LAST_MODEL_DIR` ไม่มี lock → ดับเบิลคลิกปุ่มเร็ว ๆ แล้วสตาร์ทซ้อน
- **README** ยังเป็น `# Colab` เฉย ๆ → ควรมีวิธีใช้, ข้อกำหนด GPU/ดิสก์, พอร์ต, และตัวอย่าง API

---

## หมายเหตุการทดสอบ
- ทดสอบกับ Python 3.11.2 (sandbox) — คอมไพล์ผ่านและ smoke test ผ่านทั้งหมด
- ยังไม่ได้รันบน Colab GPU จริงในรอบนี้ (ต้องมี T4 + ดาวน์โหลดโมเดล) — แนะนำรัน `Restart & Run All` หนึ่งรอบหลัง merge
