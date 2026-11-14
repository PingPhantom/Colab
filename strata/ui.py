"""Studio UI (ipywidgets) 4 แท็บ: โมเดล&ดิสก์, แชท+RAG, คลัง PDF, เปรียบเทียบโมเดล."""
from __future__ import annotations

import glob
import html as html_lib
import re
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display
from rich import print as rprint

from .config import (
    ALL_PRESET_KEYS,
    DEFAULT_PORT,
    DEFAULT_RAG_PATH,
    DRAFT_CATALOG,
    DRAFT_DIR,
    LLAMA_LOG_FILE,
    MODEL_CATALOG,
    MODEL_DIR,
    SPEC_STATE,
)
from .download import (
    _find_draft_gguf,
    clean_download_cache,
    install_all_models,
    install_draft_model,
    install_model,
    keep_only_models,
    remove_model,
)
from .rag import GLOBAL_RAG, StrataRAG
from .server import (
    _check_server_online,
    _invalidate_server_cache,
    check_gpu_and_recommend,
    get_openai_client,
    start_strata_server,
    stop_strata_server,
    strata_server_status,
    switch_strata_model,
)
from .utils import (
    NetworkMonitor,
    _dir_stats,
    _fmt_bytes,
    _fmt_rate,
    _invalidate_dir_cache,
    net_bandwidth,
    recent_log,
)


def _fmt_reply_html(raw: str, docs: list[dict] | None = None) -> str:
    out = ""
    if docs:
        items = "".join(f"<div><b style='color:#fbbf24;'>[{html_lib.escape(d['source'])}]</b> ({d['score']}): {html_lib.escape(d['text'][:180])}...</div>" for d in docs)
        out += f"<details style='background:#0f172a;border:1px solid #1e3a8a;border-radius:6px;padding:6px 10px;margin-bottom:6px;font-size:12px;color:#cbd5e1;'><summary style='cursor:pointer;color:#60a5fa;font-weight:600;'>📚 อ้างอิงจาก StrataRAG ({len(docs)} ส่วน)</summary>{items}</details>"
    thinks: list[str] = []
    vis = re.sub(r"<think>(.*?)</think>", lambda m: (thinks.append(m.group(1).strip()) or ""), raw, flags=re.DOTALL)
    if "<think>" in vis and "</think>" not in vis:
        vis, rest = vis.split("<think>", 1); thinks.append(rest.strip() + " ⏳...")
    if thinks:
        out += f"<details style='background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 10px;margin-bottom:6px;font-size:12px;color:#94a3b8;'><summary style='cursor:pointer;color:#38bdf8;'>🧠 กระบวนการคิด</summary>{html_lib.escape(chr(10).join(thinks)).replace(chr(10), '<br>')}</details>"
    return out + f"<div style='line-height:1.6;'>{html_lib.escape(vis.strip()).replace(chr(10), '<br>') or '...'}</div>"

def _L(**kw) -> widgets.Layout:
    return widgets.Layout(**kw)

def _B(desc: str, style: str = "", w: str = "110px", **kw) -> widgets.Button:
    return widgets.Button(description=desc, button_style=style, layout=_L(width=w, **kw))

LIVE_REFS: list = []  # อ้างอิงตัวรีเฟรชล่าสุด (ให้เทสต์/ผู้ใช้เรียก run_once() ได้)


class LiveRefresher:
    """รีเฟรช UI แบบเรียลไทม์ใน thread แยก — เปิด/ปิดได้ และเรียก run_once() เพื่อทดสอบได้"""

    def __init__(self, tick: Callable[[int], None], interval: float = 2.0):
        self.tick = tick
        self.interval = max(0.5, float(interval))
        self.running = False
        self.ticks = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.running = True
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="strata-live")
        self._thread.start()

    def stop(self) -> None:
        self.running = False

    def run_once(self) -> None:
        with self._lock:  # กัน tick ซ้อนกัน (widget อัปเดตทีละครั้ง)
            self.ticks += 1
            self.tick(self.ticks)

    def _loop(self) -> None:
        while True:
            time.sleep(self.interval)
            if not self.running:
                continue
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 — thread ต้องไม่ตายเพราะ widget ตัวใดตัวหนึ่ง
                continue


def show_model_manager_ui(model_dir: str = MODEL_DIR, port: int = DEFAULT_PORT, rag_engine: StrataRAG = GLOBAL_RAG, live: bool = True):
    state = {"active": "ยังไม่ระบุ"}; history: list[dict] = []; btns: list[widgets.Button] = []
    header, rag_stat, spec_hint, c_stat, rag_out, a_stat = (widgets.HTML() for _ in range(6))

    def _header_html() -> str:
        cp = Path(model_dir) if Path(model_dir).exists() else Path("/content")
        u = shutil.disk_usage(cp); tg, ug, fg = u.total / (1024**3), u.used / (1024**3), u.free / (1024**3)
        pct = min(100.0, ug / tg * 100) if tg else 0.0
        bc = "#22c55e" if pct < 70 else ("#f59e0b" if pct < 88 else "#ef4444")
        hw = check_gpu_and_recommend(verbose=False); on, mid = _check_server_online(port)
        if on:
            state["active"] = mid
        sb = f"<span style='background:#14532d;color:#4ade80;padding:3px 9px;border-radius:99px;font-size:12px;'>🟢 Online (:{port} • {mid})</span>" if on else f"<span style='background:#450a0a;color:#fca5a5;padding:3px 9px;border-radius:99px;font-size:12px;'>🔴 Offline (:{port})</span>"
        spb = f"<span style='background:#3b0764;color:#e9d5ff;padding:3px 9px;border-radius:99px;font-size:12px;'>{html_lib.escape(SPEC_STATE['active_label'])}</span>" if on and not SPEC_STATE["active_label"].startswith("⏹️") else ""
        rb = f"<span style='background:#1e3a8a;color:#93c5fd;padding:3px 9px;border-radius:99px;font-size:12px;'>📚 RAG: {len(rag_engine.chunks)}</span>"
        gb = f"🎮 {hw['gpu_name']} ({hw['vram_free_gb']:.1f}/{hw['vram_total_gb']:.1f}GB)" if hw["has_gpu"] else "🖥️ CPU"
        net = net_bandwidth()
        nb = (
            "<span style='background:#0c4a6e;color:#7dd3fc;padding:3px 9px;border-radius:99px;font-size:12px;'"
            f" title='รวมทั้งหมด — ดาวน์โหลด {_fmt_bytes(net['rx_bytes'])} • อัปโหลด {_fmt_bytes(net['tx_bytes'])}'>"
            f"⬇ {_fmt_rate(net['down_bps'])} · ⬆ {_fmt_rate(net['up_bps'])}"
            "</span>"
        )
        return f"<div style='background:#0f172a;color:#f8fafc;padding:12px 16px;border-radius:10px;border:1px solid #334155;font-family:sans-serif;margin-bottom:6px;'><div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;margin-bottom:6px;'><b style='font-size:15px;'>🎛️ Strata Studio (Port {port} • {len(ALL_PRESET_KEYS)} Models + Draft/MTP/N-gram + RAG)</b><div>{sb} {spb} {rb} {nb} <span style='font-size:12px;color:#38bdf8;margin-left:4px;'>{gb}</span></div></div><div style='font-size:12px;color:#cbd5e1;margin-bottom:4px;'>💾 ดิสก์ว่าง <b style='color:{bc};'>{fg:.2f} GB</b> / {tg:.2f} GB ({pct:.1f}%)</div><div style='width:100%;background:#1e293b;height:8px;border-radius:99px;overflow:hidden;'><div style='width:{pct:.1f}%;background:{bc};height:100%;'></div></div></div>"

    def _draft_opts() -> list[tuple[str, str]]:
        opts = []
        for dk, dv in DRAFT_CATALOG.items():
            fp = _find_draft_gguf(Path(DRAFT_DIR) / dk)
            opts.append((f"✅ {dv['desc']} ({_fmt_bytes(fp.stat().st_size)})" if fp else f"⬜ {dv['desc']}", dk))
        opts.append(("⚙️ กำหนดเอง (พิมพ์ org/repo:*Q4_K_M*.gguf ช่องขวา)", "custom"))
        return opts

    spec_tb = widgets.ToggleButtons(options=[("⏹️ ปิด", "off"), ("✨ Auto", "auto"), ("🚀 MTP", "mtp"), ("⚡ Draft", "draft"), ("📚 N-gram", "ngram")], value=SPEC_STATE["mode"], description="⚡ เร่งสปีด:", style={"button_width": "82px", "description_width": "72px"})
    dmax_sl = widgets.IntSlider(value=int(SPEC_STATE["draft_max"]), min=0, max=24, description="Draft-Max:", continuous_update=False, layout=_L(width="205px"))
    chk_kvq8 = widgets.Checkbox(value=bool(SPEC_STATE["kv_q8"]), description="💾 KV Q8_0", indent=False, layout=_L(width="110px"))
    b_apply_spec = _B("🔄 รีสตาร์ทใช้โหมดนี้", "info", "150px")
    draft_dd = widgets.Dropdown(options=_draft_opts(), value="qwen35-0.8b", description="📦 Draft:", layout=_L(width="335px"))
    b_dl_draft, b_clr_draft, b_rm_draft = _B("📥 โหลด & ใช้ Draft", "primary", "145px"), _B("✖ ล้าง", "", "68px"), _B("🗑️ ลบไฟล์", "danger", "88px")
    draft_in = widgets.Text(value=str(SPEC_STATE["draft_model"]), placeholder="พาธ .gguf หรือ org/repo:*Q4_K_M*.gguf", description="ไฟล์ที่ใช้:", layout=_L(flex="1 1 250px"))

    def _upd_spec(_=None):
        SPEC_STATE.update({"mode": spec_tb.value, "draft_max": int(dmax_sl.value), "kv_q8": bool(chk_kvq8.value), "draft_model": draft_in.value.strip()})
        dm_t = "Auto (MTP=2, Draft=6, N-gram=16)" if dmax_sl.value == 0 else str(dmax_sl.value)
        df_t = Path(SPEC_STATE["draft_model"]).name if SPEC_STATE["draft_model"] and Path(SPEC_STATE["draft_model"]).is_file() else "ยังไม่ระบุ"
        mdesc = {"off": "⏹️ ปิด Speculative", "auto": "✨ Auto: Draft > MTP > N-gram", "mtp": "🚀 MTP (--spec-type draft-mtp)", "draft": f"⚡ External Draft ({html_lib.escape(df_t)})", "ngram": "📚 N-gram (--spec-type ngram-mod)"}[spec_tb.value]
        kv_badge = ' | <b style="color:#4ade80;">KV:Q8_0</b>' if chk_kvq8.value else ''
        spec_hint.value = f"<span style='font-size:11.5px;color:#cbd5e1;'>{mdesc} | <b style='color:#38bdf8;'>Max:</b> {dm_t}{kv_badge}</span>"

    def _on_dd_chg(ch):
        fp = _find_draft_gguf(Path(DRAFT_DIR) / ch["new"]) if ch["new"] in DRAFT_CATALOG else None
        if fp:
            draft_in.value = str(fp); spec_tb.value = "mtp" if DRAFT_CATALOG[ch["new"]].get("is_mtp") else "draft"

    draft_dd.observe(_on_dd_chg, names="value")
    for w in (spec_tb, dmax_sl, chk_kvq8, draft_in):
        w.observe(_upd_spec, names="value")
    _upd_spec()

    rows_box = widgets.VBox(layout=_L(gap="6px", margin="6px 0"))
    log_out = widgets.Output(layout=_L(border="1px solid #334155", border_radius="8px", padding="8px", max_height="240px", overflow_y="auto"))

    # --- แผงวัดความเร็วเน็ต (DL/UL) แบบเรียลไทม์ ---
    net_out = widgets.HTML("<div style='font-size:12px;color:#64748b;'>📶 กดปุ่มเพื่อเริ่มวัดความเร็วเน็ตแบบเรียลไทม์ (อัปเดตทุก 1 วิ)</div>")
    b_net = _B("📶 วัดเน็ตเรียลไทม์", "", "165px")
    net_state: dict = {"run": False, "mon": NetworkMonitor()}

    def _net_card(mon: NetworkMonitor) -> str:
        sample = mon.sample()
        bar = min(100.0, sample["down_bps"] / (20 * 1024**2) * 100) if sample["down_bps"] else 0.0
        return (
            "<div style='font-family:sans-serif;background:#0f172a;color:#f8fafc;border:1px solid #334155;"
            "border-radius:8px;padding:8px 10px;font-size:12px;'>"
            f"<b style='color:#7dd3fc;'>⬇ {_fmt_rate(sample['down_bps'])}</b> · "
            f"<b style='color:#fcd34d;'>⬆ {_fmt_rate(sample['up_bps'])}</b>"
            f"<span style='color:#64748b;'> | รวม ⬇ {_fmt_bytes(sample['rx_bytes'])} • ⬆ {_fmt_bytes(sample['tx_bytes'])}</span>"
            "<div style='background:#1e293b;height:6px;border-radius:99px;margin-top:6px;overflow:hidden;'>"
            f"<div style='width:{bar:.1f}%;background:#38bdf8;height:100%;'></div></div></div>"
        )

    def _net_reset_button():
        try:
            b_net.description = "📶 วัดเน็ตเรียลไทม์"
            b_net.button_style = ""
        except Exception:
            pass

    def _net_loop():
        mon = net_state["mon"]
        mon.reset()
        while net_state["run"]:
            try:
                net_out.value = _net_card(mon)
            except Exception:
                break
            time.sleep(1.0)
        net_state["run"] = False
        _net_reset_button()

    def _toggle_net(_):
        if net_state["run"]:
            net_state["run"] = False       # ลูปจะจบเองภายใน ~1 วิ (ปุ่มกลับเป็นสถานะเริ่มทันที)
            _net_reset_button()
            return
        net_state["run"] = True
        b_net.description = "⏸ หยุดวัดเน็ต"
        b_net.button_style = "warning"
        threading.Thread(target=_net_loop, daemon=True).start()

    b_net.on_click(_toggle_net)
    b_all, b_cache, b_stat, b_stop, b_delall, b_ref = _B(f"📦 โหลดครบ {len(ALL_PRESET_KEYS)} ตัว", "primary", "155px"), _B("🧹 ล้างแคช", "info", "105px"), _B("📡 สถานะ", "", "95px"), _B("🛑 หยุด Server", "warning", "115px"), _B("🗑️ ลบทั้งหมด", "danger", "105px"), _B("🔄 รีเฟรช", "", "88px")
    chk_del = widgets.Checkbox(value=False, description="ยืนยันลบทั้งหมด", indent=False, layout=_L(width="125px"))

    busy = {"n": 0}

    def _set_busy(flag: bool) -> None:
        """ปิด/เปิดปุ่มทั้งหมดระหว่างทำงาน (นับซ้อนได้) — ต้องเปิดคืนเสมอ ไม่งั้นปุ่มค้าง disabled"""
        busy["n"] = busy["n"] + 1 if flag else max(0, busy["n"] - 1)
        for b in btns:
            b.disabled = busy["n"] > 0

    def _act(fn: Callable, *a, active: str | None = None, tab: int | None = None, **kw):
        _set_busy(True)
        with log_out:
            try:
                fn(*a, **kw)
                if active:
                    state["active"] = active
                if tab is not None:
                    tabs.selected_index = tab
            except Exception as e:
                rprint(f"[bold red]❌ {e}[/bold red]")
            finally:
                _invalidate_dir_cache(); _invalidate_server_cache(port)
                _set_busy(False)  # เปิดปุ่มคืนก่อน แล้ว _refresh() จะจัดสถานะรายแถวให้ถูกอีกที
                _refresh()

    def _on_dl_draft(_):
        spec = draft_in.value.strip() if draft_dd.value == "custom" else draft_dd.value
        if not spec:
            spec_hint.value = "<span style='color:#fbbf24;'>⚠️ พิมพ์ org/repo:*Q4_K_M*.gguf ในช่อง 'ไฟล์ที่ใช้' ก่อนครับ</span>"; return
        def _job():
            gp, is_m = install_draft_model(spec, draft_base_dir=DRAFT_DIR)
            draft_in.value = gp; spec_tb.value = "mtp" if is_m else "draft"; _upd_spec()
        tabs.selected_index = 0; _act(_job)

    def _on_rm_draft(_):
        def _job():
            if draft_dd.value in DRAFT_CATALOG and (Path(DRAFT_DIR) / draft_dd.value).is_dir():
                shutil.rmtree(Path(DRAFT_DIR) / draft_dd.value, ignore_errors=True)
                rprint(f"[bold yellow]🗑️ ลบ Draft '{draft_dd.value}' แล้ว[/bold yellow]")
            elif draft_in.value.strip() and Path(draft_in.value.strip()).is_file():
                Path(draft_in.value.strip()).unlink(missing_ok=True)
            draft_in.value = ""
            if spec_tb.value == "draft":
                spec_tb.value = "off"
            _upd_spec()
        _act(_job)

    def _on_apply_spec(_):
        _upd_spec(); on, mid = _check_server_online(port, ttl=0.0)
        cdir = SPEC_STATE.get("last_model_dir") or (str(Path(model_dir) / mid) if on and (Path(model_dir) / mid).is_dir() else None)
        if cdir and Path(cdir).is_dir():
            _act(start_strata_server, model_dir=cdir, port=port, restart=True)
        else:
            spec_hint.value = "<span style='color:#fbbf24;'>✅ บันทึกโหมดแล้ว! กด 🚀 รัน & แชท ของโมเดลหลักได้เลย</span>"

    def _on_del_all(_):
        if not chk_del.value:
            with log_out:
                rprint("[yellow]⚠️ ติ๊ก 'ยืนยันลบทั้งหมด' ก่อนกดลบครับ[/yellow]")
            return
        chk_del.value = False; _act(remove_model, "all", model_dir=model_dir, stop_server=True, active="หยุดแล้ว")

    b_dl_draft.on_click(_on_dl_draft)
    b_clr_draft.on_click(lambda _: (setattr(draft_in, "value", ""), setattr(spec_tb, "value", "off" if spec_tb.value == "draft" else spec_tb.value), _upd_spec()))
    b_rm_draft.on_click(_on_rm_draft); b_apply_spec.on_click(_on_apply_spec)
    b_all.on_click(lambda _: _act(install_all_models, model_dir=model_dir, port=port))
    b_cache.on_click(lambda _: _act(clean_download_cache, model_dir=model_dir))
    b_stat.on_click(lambda _: _act(strata_server_status, port=port))
    b_stop.on_click(lambda _: _act(stop_strata_server, active="หยุดแล้ว"))
    b_delall.on_click(_on_del_all)

    def _on_refresh(_):
        # ต้องล้างแคชก่อน ไม่งั้นกดรีเฟรชหลังโหลดเสร็จจะเห็นขนาด/สถานะเก่า
        _invalidate_dir_cache(); _invalidate_server_cache(port)
        _refresh()
        with log_out:
            rprint("[dim]🔄 รีเฟรชสถานะโมเดล/ดิสก์แล้ว[/dim]")

    b_ref.on_click(_on_refresh)

    rows: dict[str, dict] = {}

    def _row_widgets(k: str) -> dict:
        w = {
            "info": widgets.HTML(layout=_L(flex="1 1 250px")),
            "bd": _B("📥 โหลด", "primary", "95px"),
            "br": _B("🚀 รัน & แชท", "success", "110px"),
            "bk": _B("📌 เก็บตัวเดียว", "", "105px"),
            "bx": _B("🗑️ ลบ", "danger", "70px"),
            "installed": False,
        }
        w["bd"].on_click(lambda _, opt=k: _act(install_model, opt, model_dir=model_dir, port=port))
        w["br"].on_click(lambda _, opt=k: _act(switch_strata_model if rows[opt]["installed"] else install_model, opt, model_dir=model_dir, port=port, active=opt, tab=1, **({} if rows[opt]["installed"] else {"auto_start_server": True})))
        w["bk"].on_click(lambda _, opt=k: _act(keep_only_models, opt, model_dir=model_dir))
        w["bx"].on_click(lambda _, opt=k: _act(remove_model, opt, model_dir=model_dir))
        return w

    def _build_rows() -> tuple:
        for k in ALL_PRESET_KEYS:
            rows.setdefault(k, _row_widgets(k))
        return tuple(
            widgets.HBox(
                [rows[k]["info"], widgets.HBox([rows[k]["bd"], rows[k]["br"], rows[k]["bk"], rows[k]["bx"]], layout=_L(gap="5px"))],
                layout=_L(justify_content="space-between", align_items="center", padding="8px 10px", border="1px solid #334155", border_radius="8px", flex_flow="row wrap"),
            )
            for k in ALL_PRESET_KEYS
        )

    ui_lock = threading.RLock()

    def _refresh():
        with ui_lock:
            _refresh_inner()

    def _refresh_inner():
        header.value = _header_html()
        cur_d = draft_dd.value; n_opts = _draft_opts(); draft_dd.options = n_opts
        if any(v == cur_d for _, v in n_opts):
            draft_dd.value = cur_d
        rag_stat.value = f"<b style='color:#38bdf8;'>📊 คลังความรู้ StrataRAG:</b> <b>{len(rag_engine.chunks)}</b> chunks (PDF, TXT, MD)"
        if not rows:
            rows_box.children = _build_rows()
            btns.clear()
            btns.extend([b_all, b_cache, b_stat, b_stop, b_delall, b_ref, b_apply_spec, b_dl_draft, b_clr_draft, b_rm_draft])
            for k in ALL_PRESET_KEYS:
                btns.extend([rows[k]["bd"], rows[k]["br"], rows[k]["bk"], rows[k]["bx"]])
        base = Path(model_dir)
        cnt_dirs = len([d for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]) if base.exists() else 0
        for k in ALL_PRESET_KEYS:
            cfg = MODEL_CATALOG[k]; w = rows[k]; sub = base / k; ex = sub.is_dir()
            stats = _dir_stats(sub) if ex else {"bytes": 0, "gguf": 0}
            gc, sz = int(stats["gguf"]), int(stats["bytes"])
            w["installed"] = gc > 0
            mtag = " <span style='background:#3b0764;color:#e9d5ff;padding:1px 6px;border-radius:6px;font-size:10px;'>🚀 MTP</span>" if cfg.get("has_mtp") else ""
            bdg = f"<span style='color:#4ade80;font-size:11px;'>✅ ติดตั้งแล้ว ({_fmt_bytes(sz)})</span>" if gc else f"<span style='color:#94a3b8;font-size:11px;'>⬜ ยังไม่โหลด (~{cfg['approx_gb']}GB)</span>"
            w["info"].value = f"<div style='font-family:sans-serif;'><b style='color:#38bdf8;'>{k}</b>{mtag} {bdg}<div style='font-size:11px;color:#cbd5e1;'>{cfg['desc']} (<code>{cfg['repo_id']}</code>)</div></div>"
            w["bd"].description = "📥 โหลดซ้ำ" if gc else "📥 โหลด"
            w["bd"].button_style = "" if gc else "primary"
            w["bk"].disabled = not (ex and cnt_dirs > 1)
            w["bx"].disabled = not ex

    spec_bar = widgets.VBox([
        widgets.HBox([spec_tb, dmax_sl, chk_kvq8, b_apply_spec], layout=_L(flex_flow="row wrap", align_items="center", gap="6px")),
        widgets.HBox([draft_dd, b_dl_draft, b_clr_draft, b_rm_draft, draft_in], layout=_L(flex_flow="row wrap", align_items="center", gap="6px", width="100%")),
        spec_hint,
    ], layout=_L(padding="8px 12px", border="1px solid #334155", border_radius="10px", margin="0 0 6px 0", gap="5px"))

    b_live = _B("⏱ เรียลไทม์: เปิด", "success", "150px")
    live_out = widgets.HTML()
    tab1 = widgets.VBox([
        widgets.HBox([b_all, b_cache, b_stat, b_stop, chk_del, b_delall, b_ref], layout=_L(flex_flow="row wrap", gap="5px")),
        rows_box,
        widgets.HBox([b_net, b_live, net_out], layout=_L(flex_flow="row wrap", gap="8px", align_items="center")),
        live_out,
        log_out,
    ])

    # --- TAB 2: Chat + StrataRAG ---
    sys_in = widgets.Text(value="คุณคือผู้ช่วย AI อัจฉริยะที่ตอบชัดเจนและถูกต้อง", description="System:", layout=_L(flex="1 1 300px"))
    q_dd = widgets.Dropdown(options=[("— สลับโมเดลด่วน —", "")] + [(k, k) for k in ALL_PRESET_KEYS], value="", layout=_L(width="220px"))
    b_qsw = _B("⚡ สลับ", "warning", "90px")
    chk_rag = widgets.Checkbox(value=True, description="📚 ใช้ StrataRAG", indent=False, layout=_L(width="145px"))
    topk_sl = widgets.IntSlider(value=3, min=1, max=8, description="Top-K:", layout=_L(width="185px"))
    temp_sl = widgets.FloatSlider(value=0.7, min=0.0, max=2.0, step=0.05, description="Temp:", layout=_L(width="190px"))
    max_sl = widgets.IntSlider(value=1024, min=128, max=4096, step=64, description="MaxTok:", layout=_L(width="215px"))
    feed = widgets.VBox([widgets.HTML("<div style='color:#94a3b8;text-align:center;padding:10px;'>💬 พิมพ์ข้อความด้านล่างเพื่อเริ่มแชท</div>")], layout=_L(border="1px solid #334155", border_radius="8px", padding="10px", min_height="240px", max_height="400px", overflow_y="auto", gap="8px"))
    u_in = widgets.Textarea(placeholder="พิมพ์คำถาม...", rows=3, layout=_L(flex="1 1 auto"))
    b_send, b_clr = _B("📨 ส่งข้อความ", "success", "125px", height="66px"), _B("🗑️ ล้างแชท", "danger", "95px", height="66px")

    b_qsw.on_click(lambda _: q_dd.value and _act(switch_strata_model if (Path(model_dir) / q_dd.value).exists() else install_model, q_dd.value, model_dir=model_dir, port=port, active=q_dd.value, **({} if (Path(model_dir) / q_dd.value).exists() else {"auto_start_server": True})))
    b_clr.on_click(lambda _: (history.clear(), setattr(feed, "children", (widgets.HTML("<div style='color:#94a3b8;text-align:center;'>🧹 ล้างแชทแล้ว</div>"),))))

    def _send(_):
        q = u_in.value.strip()
        if not q:
            return
        is_on, active_mid = _check_server_online(port, ttl=0.5)
        if not is_on:
            c_stat.value = f"<span style='color:#f87171;'>❌ Server พอร์ต {port} ยังไม่เปิด! กด 🚀 รัน & แชท ที่แท็บแรกก่อนครับ</span>"; return
        u_in.value = ""; b_send.disabled = True
        aug_q, docs = rag_engine.build_augmented_prompt(q, top_k=int(topk_sl.value)) if (chk_rag.value and rag_engine.chunks) else (q, [])
        ub = widgets.HTML(f"<div style='text-align:right;'><span style='display:inline-block;background:#1d4ed8;color:#fff;padding:8px 12px;border-radius:10px;max-width:80%;text-align:left;'>👤 {html_lib.escape(q)}</span></div>")
        bb = widgets.HTML("<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;'>⏳ กำลังคิด...</div>")
        feed.children = tuple([*(feed.children if history else []), ub, bb])
        t0, buf, n = time.perf_counter(), [], 0; stag = html_lib.escape(SPEC_STATE["active_label"])
        try:
            _ctx = max(2048, int(SPEC_STATE.get("ctx_size", 4096)))
            _mtok = max(128, min(int(max_sl.value), _ctx - 1024))
            st = get_openai_client(port=port).chat.completions.create(model=active_mid, messages=[{"role": "system", "content": sys_in.value}, *history, {"role": "user", "content": aug_q}], temperature=float(temp_sl.value), max_tokens=_mtok, stream=True)  # type: ignore[arg-type]
            last_draw = 0.0
            for ch in st:
                c0 = next(iter(ch.choices), None); d = c0.delta.content if c0 else None
                if not d:
                    continue
                buf.append(d); n += 1
                now = time.perf_counter()
                if (now - last_draw) < 0.1 and not d.endswith(("\n", "。")):
                    continue      # วาด HTML ทุก ~100ms พอ (เดิมทำทุก token = O(n^2))
                last_draw = now
                spd = n / max(0.01, now - t0)
                bb.value = f"<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;border:1px solid #334155;'><div style='font-size:11px;color:#38bdf8;'>🤖 {active_mid} • <span style='color:#e9d5ff;'>{stag}</span> • <b>{spd:.1f} tok/s</b></div>{_fmt_reply_html(''.join(buf), docs)}</div>"
            el = max(0.01, time.perf_counter() - t0)
            bb.value = f"<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;border:1px solid #334155;'><div style='font-size:11px;color:#38bdf8;'>🤖 {active_mid} • <span style='color:#e9d5ff;'>{stag}</span> • <b>{n / el:.1f} tok/s</b></div>{_fmt_reply_html(''.join(buf), docs)}</div>"
            history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": "".join(buf)}])
            _budget_chars = max(1024, _ctx - _mtok - 512) * 3
            while len(history) > 2 and sum(len(m["content"]) for m in history) > _budget_chars:
                history.pop(0)
                if history and history[0]["role"] == "assistant":
                    history.pop(0)
            c_stat.value = f"<span style='color:#4ade80;font-size:12px;'>✅ เสร็จใน {el:.2f}s ({n / el:.1f} tok/s • {stag})</span>"
        except Exception as e:
            bb.value = f"<div style='background:#450a0a;color:#fecaca;padding:10px;border-radius:8px;'>❌ {html_lib.escape(str(e))}</div>"
        finally:
            b_send.disabled = False; header.value = _header_html()

    b_send.on_click(_send)
    tab2 = widgets.VBox([widgets.HBox([sys_in, q_dd, b_qsw], layout=_L(flex_flow="row wrap", gap="6px")), widgets.HBox([chk_rag, topk_sl, temp_sl, max_sl], layout=_L(flex_flow="row wrap", gap="8px")), feed, widgets.HBox([u_in, b_send, b_clr], layout=_L(gap="6px")), c_stat])

    # --- TAB 3: PDF & Text StrataRAG ---
    uploader = widgets.FileUpload(accept=".pdf,.txt,.md,.csv,.json,.py", multiple=True, description="📤 อัปโหลดไฟล์")
    b_up, b_clrrag = _B("📥 นำไฟล์เข้า RAG", "success", "165px"), _B("🗑️ ล้างคลัง RAG", "danger", "130px")
    chk_sz = widgets.IntSlider(value=500, min=200, max=1500, step=50, description="Chunk:", layout=_L(width="205px"))
    fp_in = widgets.Text(placeholder="/content/doc.pdf หรือ /content/*.pdf", description="พาธไฟล์:", layout=_L(flex="1 1 260px"))
    b_fp, b_txt, b_tq = _B("📂 โหลดจากพาธ", "info", "130px"), _B("➕ เพิ่มข้อความ", "primary", "130px"), _B("🔍 ทดสอบค้นหา", "", "125px")
    b_save_rag, b_load_rag = _B("💾 บันทึกคลัง", "info", "130px"), _B("📂 โหลดคลังเดิม", "info", "140px")
    rag_path_in = widgets.Text(value=DEFAULT_RAG_PATH, description="ไฟล์คลัง:", layout=_L(flex="1 1 260px"))
    src_in = widgets.Text(value="note", description="แหล่งที่มา:", layout=_L(width="210px"))
    txt_in = widgets.Textarea(placeholder="วางข้อความ/บทความที่นี่...", rows=4, layout=_L(width="100%"))
    tq_in = widgets.Text(placeholder="คำค้นหาทดสอบ...", description="ค้นหา:", layout=_L(flex="1 1 240px"))

    def _ingest_up(_):
        items = list(uploader.value.values()) if isinstance(uploader.value, dict) else list(uploader.value)
        if not items:
            rag_out.value = "<span style='color:#fbbf24;'>⚠️ เลือกไฟล์ก่อนครับ</span>"; return
        for it in items:
            nm = it.get("name") or it.get("metadata", {}).get("name", "upload.pdf")
            raw = it.get("content", b""); rb = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
            rag_engine.add_pdf_bytes(rb, filename=nm, chunk_size=int(chk_sz.value)) if nm.lower().endswith(".pdf") else rag_engine.add_documents([{"source": nm, "text": rb.decode("utf-8", errors="ignore")}], chunk_size=int(chk_sz.value))
        _refresh(); rag_out.value = f"<span style='color:#4ade80;'>✅ นำเข้า {len(items)} ไฟล์สำเร็จ! ({len(rag_engine.chunks)} chunks)</span>"

    def _ingest_fp(_):
        pat = fp_in.value.strip()
        files = [Path(p) for p in (glob.glob(pat) if any(c in pat for c in "*?[]") else [pat]) if Path(p).is_file()] if pat else []
        if not files:
            rag_out.value = "<span style='color:#f87171;'>❌ ไม่พบไฟล์</span>"; return
        rag_engine.add_files(*files, chunk_size=int(chk_sz.value)); _refresh()
        rag_out.value = f"<span style='color:#4ade80;'>✅ โหลด {len(files)} ไฟล์เรียบร้อย!</span>"

    def _add_note(_):
        if txt_in.value.strip():
            rag_engine.add_documents([{"source": src_in.value or "note", "text": txt_in.value.strip()}], chunk_size=int(chk_sz.value))
            txt_in.value = ""; _refresh()

    def _save_rag(_):
        try:
            p = rag_engine.save(rag_path_in.value.strip() or DEFAULT_RAG_PATH)
            rag_out.value = f"<span style='color:#4ade80;'>💾 บันทึกแล้ว: {html_lib.escape(p)} ({len(rag_engine.chunks)} chunks)</span>"
        except Exception as e:
            rag_out.value = f"<span style='color:#f87171;'>❌ {html_lib.escape(str(e))}</span>"

    def _load_rag(_):
        try:
            n = rag_engine.load(rag_path_in.value.strip() or DEFAULT_RAG_PATH)
            _refresh()
            rag_out.value = f"<span style='color:#4ade80;'>📂 โหลดคลังเดิม {n} chunks เรียบร้อย</span>"
        except Exception as e:
            rag_out.value = f"<span style='color:#f87171;'>❌ {html_lib.escape(str(e))}</span>"

    b_up.on_click(_ingest_up); b_fp.on_click(_ingest_fp); b_txt.on_click(_add_note)
    b_save_rag.on_click(_save_rag); b_load_rag.on_click(_load_rag)
    b_clrrag.on_click(lambda _: (rag_engine.clear(), _refresh(), setattr(rag_out, "value", "<span style='color:#fbbf24;'>🧹 ล้างคลัง RAG แล้ว</span>")))
    b_tq.on_click(lambda _: setattr(rag_out, "value", "".join(f"<div style='background:#1e293b;color:#f8fafc;padding:6px 10px;border-radius:6px;margin-top:4px;font-size:12px;'><b style='color:#38bdf8;'>#{h['rank']} {html_lib.escape(h['source'])}</b> ({h['score']})<br>{html_lib.escape(h['text'])}</div>" for h in rag_engine.search(tq_in.value.strip(), top_k=int(topk_sl.value))) or "<span style='color:#fbbf24;'>ไม่พบข้อมูลที่เกี่ยวข้อง (ลองคำค้นอื่น)</span>"))

    tab3 = widgets.VBox([rag_stat, widgets.HBox([uploader, b_up, chk_sz, b_clrrag], layout=_L(flex_flow="row wrap", gap="8px", align_items="center")), widgets.HBox([fp_in, b_fp], layout=_L(gap="6px")), widgets.HBox([src_in, b_txt], layout=_L(gap="6px")), txt_in, widgets.HBox([tq_in, b_tq], layout=_L(gap="6px")), widgets.HBox([rag_path_in, b_save_rag, b_load_rag], layout=_L(flex_flow="row wrap", gap="6px", align_items="center")), rag_out], layout=_L(gap="6px"))

    # --- TAB 4: Arena Compare ---
    a_chks = {k: widgets.Checkbox(value=(k in ("qwen35-9b-defiant-fable", "ornith-15-9b")), description=k, indent=False, layout=_L(width="215px")) for k in ALL_PRESET_KEYS}
    a_rag = widgets.Checkbox(value=True, description="📚 แนบ StrataRAG", indent=False, layout=_L(width="155px"))
    a_dl = widgets.Checkbox(value=False, description="โหลดอัตโนมัติถ้ายังไม่มีไฟล์", indent=False, layout=_L(width="210px"))
    a_in = widgets.Textarea(placeholder="พิมพ์คำถามเดียวเพื่อเปรียบเทียบหลายโมเดล...", rows=3, layout=_L(flex="1 1 auto"))
    b_arena = _B("⚔️ เริ่มเปรียบเทียบ", "warning", "150px", height="66px")
    a_temp_sl = widgets.FloatSlider(value=0.7, min=0.0, max=2.0, step=0.05, description="Temp:", layout=_L(width="185px"))
    a_max_sl = widgets.IntSlider(value=512, min=128, max=2048, step=64, description="MaxTok:", layout=_L(width="205px"))
    a_box = widgets.HBox(layout=_L(flex_flow="row wrap", gap="10px"))

    def _run_arena(_):
        _upd_spec(); q = a_in.value.strip(); sel = [k for k, c in a_chks.items() if c.value]
        if not q or not sel:
            a_stat.value = "<span style='color:#fbbf24;'>⚠️ พิมพ์คำถามและเลือกอย่างน้อย 1 โมเดล</span>"; return
        b_arena.disabled = True
        fq, docs = rag_engine.build_augmented_prompt(q, top_k=int(topk_sl.value)) if (a_rag.value and rag_engine.chunks) else (q, [])
        cards = {m: widgets.HTML(f"<div style='background:#0f172a;color:#94a3b8;padding:10px;border-radius:8px;border:1px solid #334155;'><b>🤖 {m}</b><br>⏳ รอคิว...</div>", layout=_L(flex="1 1 300px", min_width="280px")) for m in sel}
        a_box.children = tuple(cards.values()); stats = []
        try:
            for i, m in enumerate(sel, 1):
                _sub = Path(model_dir) / m
                has_f = _sub.is_dir() and _dir_stats(_sub)["gguf"] > 0
                if not has_f and not a_dl.value:
                    cards[m].value = f"<div style='background:#450a0a;color:#fca5a5;padding:10px;border-radius:8px;'>⚠️ <b>{m}</b> ยังไม่ได้โหลดไฟล์</div>"; continue
                a_stat.value = f"<span style='color:#38bdf8;'>🔄 [{i}/{len(sel)}] กำลังเปิด <b>{m}</b>...</span>"
                try:
                    with log_out:
                        install_model(m, model_dir=model_dir, auto_start_server=True, port=port) if not has_f else switch_strata_model(m, model_dir=model_dir, port=port)  # type: ignore[arg-type]
                except Exception as err:
                    cards[m].value = f"<div style='background:#450a0a;color:#fecaca;padding:10px;border-radius:8px;'>❌ {html_lib.escape(str(err))}</div>"; continue
                is_ok, active_mid = _check_server_online(port, ttl=0.0)
                if not is_ok:
                    cards[m].value = "<div style='background:#450a0a;color:#fecaca;padding:10px;border-radius:8px;'>❌ Server ไม่ตอบสนอง</div>"; continue
                state["active"] = active_mid; header.value = _header_html(); stag = html_lib.escape(SPEC_STATE["active_label"])
                t0, buf, n = time.perf_counter(), [], 0
                _actx = max(2048, int(SPEC_STATE.get("ctx_size", 4096)))
                _amtok = max(128, min(int(a_max_sl.value), _actx - 1024))
                last_draw = 0.0
                for ch in get_openai_client(port=port).chat.completions.create(model=active_mid, messages=[{"role": "system", "content": sys_in.value}, {"role": "user", "content": fq}], temperature=float(a_temp_sl.value), max_tokens=_amtok, stream=True):  # type: ignore[arg-type]
                    c0 = next(iter(ch.choices), None); d = c0.delta.content if c0 else None
                    if not d:
                        continue
                    buf.append(d); n += 1
                    now = time.perf_counter()
                    if (now - last_draw) < 0.15 and not d.endswith(("\n", "。")):
                        continue
                    last_draw = now
                    spd = n / max(0.01, now - t0)
                    cards[m].value = f"<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:8px;border:1px solid #38bdf8;'><b style='color:#38bdf8;'>🤖 {m}</b> • <span style='font-size:11px;color:#e9d5ff;'>{stag}</span> ({spd:.1f} tok/s){_fmt_reply_html(''.join(buf), docs if i == 1 else None)}</div>"
                el = max(0.01, time.perf_counter() - t0); stats.append(f"{m}: {el:.1f}s ({n / el:.1f} tok/s)")
            a_stat.value = f"<span style='color:#4ade80;font-weight:600;'>🏆 เสร็จสิ้น: {' | '.join(stats) if stats else 'ไม่มีโมเดลที่พร้อมรัน'}</span>"
        finally:
            b_arena.disabled = False; _refresh()

    b_arena.on_click(_run_arena)
    tab4 = widgets.VBox([widgets.HBox(list(a_chks.values()), layout=_L(flex_flow="row wrap", gap="6px")), widgets.HBox([a_rag, a_dl, a_temp_sl, a_max_sl], layout=_L(gap="12px", flex_flow="row wrap")), widgets.HBox([a_in, b_arena], layout=_L(gap="6px")), a_stat, a_box])

    tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4])
    for idx, ttl in enumerate(["📦 1. โมเดล & ดิสก์ (10 ตัว)", "💬 2. แชท + RAG", "📚 3. คลัง PDF RAG", "⚔️ 4. เปรียบเทียบโมเดล"]):
        tabs.set_title(idx, ttl)
    live_ref = None

    def _tick(n: int) -> None:
        """อัปเดตหัว UI + แผง log ทุกคู่ และเดินดิสก์/แถวโมเดลทุก ~3 คู่"""
        header.value = _header_html()
        lines = recent_log(8)
        tail: list[str] = []
        try:
            if LLAMA_LOG_FILE.exists():
                tail = LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-6:]
        except OSError:
            tail = []
        rows_html = "".join(f"<div>{html_lib.escape(x)}</div>" for x in [*lines, *tail][-10:])
        live_out.value = (
            "<div style='background:#0b1220;color:#94a3b8;font-family:ui-monospace,SFMono-Regular,monospace;"
            "font-size:11px;line-height:1.5;padding:8px 10px;border-radius:8px;border:1px solid #334155;"
            "max-height:150px;overflow:auto;'>"
            f"<b style='color:#38bdf8;'>⏱ เรียลไทม์ (ทุก {live_ref.interval:.0f}s • tick {n})</b>"
            f"{rows_html or '<div>(ยังไม่มี log)</div>'}</div>"
        )
        if n % 3 == 0:  # ทุก ~6 วินาที: คำนวณขนาดโฟลเดอร์/สถานะแถวใหม่
            _refresh()

    live_ref = LiveRefresher(_tick, interval=2.0)

    def _toggle_live(_):
        if live_ref.running:
            live_ref.stop()
            b_live.description = "⏸ เรียลไทม์: ปิด"
            b_live.button_style = ""
        else:
            live_ref.start()
            _tick(live_ref.ticks)
            b_live.description = "⏱ เรียลไทม์: เปิด"
            b_live.button_style = "success"

    b_live.on_click(_toggle_live)

    _refresh()
    _tick(1)  # เติมแผง log ครั้งแรกก่อนแสดงผล (n=1 ไม่ต้องเดินดิสก์ซ้ำ)
    root = widgets.VBox([header, spec_bar, tabs])
    display(root)
    LIVE_REFS.append(live_ref)
    if live:
        live_ref.start()
    return root
