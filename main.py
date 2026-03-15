"""
main.py - Pulse AI v2
5-tab financial sentiment intelligence dashboard.
"""

import plotly.graph_objects as go
import streamlit as st

import time

from logic import (
    analyze_quarters,
    build_pulse_user_prompt,
    call_pulse_llm,
    check_new_filing,
    fetch_earnings_calendar,
    fetch_recent_news,
    fetch_stock_data,
    fetch_transcript,
    generate_outlook,
    generate_pdf_report,
    get_keyword_insights,
    run_sentiment_analysis,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pulse AI · Earnings Sentiment",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background:#0d1117; color:#e6edf3; }
[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #21262d; }

[data-testid="stMetric"] {
    background:#161b22; border:1px solid #21262d;
    border-radius:10px; padding:.9rem 1.1rem;
}
[data-testid="stMetricValue"] { color:#ffffff !important; font-weight:700; }
[data-testid="stMetricLabel"] { color:#8b949e !important; font-size:.75rem; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #21262d; }
.stTabs [data-baseweb="tab"] {
    color:#8b949e; border-radius:6px 6px 0 0;
    padding:.45rem .9rem; font-size:.85rem; font-weight:500;
}
.stTabs [aria-selected="true"] {
    color:#58a6ff !important;
    border-bottom:2px solid #58a6ff; background:transparent !important;
}
[data-testid="stChatMessage"] {
    background:#161b22 !important; border:1px solid #21262d; border-radius:10px;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong { color:#e6edf3 !important; }
[data-testid="stStatusContainer"] {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
}
.stButton > button {
    background:linear-gradient(135deg,#238636,#2ea043); color:#fff;
    border:none; border-radius:6px; font-weight:600; width:100%;
    padding:.55rem 0; transition:opacity .2s;
}
.stButton > button:hover { opacity:.85; }
.stTextInput input {
    background:#0d1117; border:1px solid #30363d; border-radius:6px;
    color:#e6edf3; font-family:'Inter',monospace;
    letter-spacing:.05em; font-size:1rem; text-transform:uppercase;
}
[data-testid="stDownloadButton"] button {
    background:linear-gradient(135deg,#1f6feb,#388bfd) !important;
    color:#fff; border:none; border-radius:6px; font-weight:600; width:100%;
}
hr { border-color:#21262d; }
.score-bar-wrap { background:#21262d; border-radius:8px; height:10px; overflow:hidden; margin-top:4px; }
.score-bar-fill  { height:100%; border-radius:8px; }
.kpi-card {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:1rem 1.2rem; text-align:center;
}
.kpi-label { font-size:.72rem; color:#8b949e; margin-bottom:2px; }
.kpi-value { font-size:1.1rem; font-weight:700; color:#e6edf3; }
.kpi-sub   { font-size:.75rem; color:#484f58; margin-top:1px; }

/* ── Global text brightness fix ────────────────────────────────── */
/* Input / widget labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stTextInput label,
.stSelectbox label,
.stTextArea label { color:#e6edf3 !important; font-weight:500; }

/* Sidebar labels and any plain text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color:#e6edf3 !important; }

/* General markdown / paragraph text in main area */
.stMarkdown p,
.stMarkdown li,
.stMarkdown span { color:#e6edf3 !important; }

/* Captions */
[data-testid="stCaptionContainer"] p { color:#8b949e !important; }

/* Dataframe / table text */
[data-testid="stDataFrame"] { color:#e6edf3 !important; }

/* Status / expander text */
[data-testid="stStatusContainer"] p,
[data-testid="stStatusContainer"] span,
[data-testid="stExpander"] p { color:#e6edf3 !important; }

/* Helper / info text */
[data-testid="stText"] { color:#e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.25:  return "#2ea043"
    if score >= -0.1:  return "#d29922"
    return "#f85149"


def _score_label(score: float) -> str:
    if score >= 0.6:   return "Strongly Positive"
    if score >= 0.25:  return "Moderately Positive"
    if score >= -0.1:  return "Neutral"
    if score >= -0.4:  return "Cautiously Negative"
    return "Bearish"


def _bar_html(value, color, label):
    return f"""
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#8b949e;margin-bottom:2px">
        <span>{label}</span><span>{value:.1f}%</span>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{value}%;background:{color}"></div>
      </div>
    </div>"""


def _kpi(label, value, sub=""):
    return f"""<div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""


def _plotly_dark():
    return dict(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Inter"),
    )



# ── Session state init ────────────────────────────────────────────────────────

defaults = {
    "stock":                None,   # stock data
    "sentiment":            None,   # current quarter sentiment
    "meta":                 None,   # filing metadata
    "transcript":           None,   # full text
    "history":              None,   # 4-quarter list
    "keywords":             None,   # keyword insights dict
    "compare":              None,   # {ticker: {stock, sentiment}} dict
    "pdf_bytes":            None,
    "primary_ticker":       "",     # last successfully analysed primary ticker
    # ── New feature state ──────────────────────────────────────────────────
    "watchlist":            [],     # list of tickers in earnings watchlist
    "calendar_data":        [],     # [{ticker, earnings_date, days_until, ...}]
    "last_acc":             {},     # {ticker: acc_number} for change-detection
    "new_filing_banner":    None,   # {ticker, acc, date} when new filing found
    "outlook":              None,   # forward-looking outlook dict
    "news_items":           [],     # [{title, publisher, link, published_date}]
    "_last_auto_check":     0.0,    # unix timestamp of last auto-check
    "auto_run_ticker":      "",     # set by auto-trigger to invoke pipeline on next rerun
    "chat_history":         [],     # [{role, content, parsed}] for Ask Pulse chatbot
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
      <span style="font-size:1.5rem">📡</span>
      <span style="font-size:1.25rem;font-weight:700;color:#e6edf3">Pulse AI</span>
    </div>
    <p style="color:#8b949e;font-size:.78rem;margin-top:0;margin-bottom:20px;line-height:1.4">
      Earnings call sentiment powered by FinBERT
    </p>""", unsafe_allow_html=True)

    ticker_input   = st.text_input("Primary Ticker", placeholder="e.g. AAPL", max_chars=10).strip().upper()
    run_button     = st.button("⚡ Run Analysis", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    compare_input  = st.text_input(
        "Compare Tickers (comma-separated)",
        placeholder="e.g. MSFT, GOOGL, TSLA", help="Up to 3 extra tickers"
    )
    compare_button = st.button("🔄 Run Comparison", use_container_width=True)

    if st.session_state.pdf_bytes:
        st.download_button(
            label="📥 Download PDF Report",
            data=st.session_state.pdf_bytes,
            file_name=f"pulse_ai_{ticker_input or 'report'}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # ── Earnings Watchlist ────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="font-size:.82rem;font-weight:600;color:#e6edf3;margin-bottom:8px">
      📅 Earnings Watchlist
    </div>""", unsafe_allow_html=True)

    wl_col1, wl_col2 = st.columns([3, 1])
    wl_input_val = wl_col1.text_input(
        "Add ticker", placeholder="e.g. NVDA",
        label_visibility="collapsed", key="wl_add_input", max_chars=10
    ).strip().upper()
    if wl_col2.button("➕", help="Add to watchlist", key="wl_add_btn"):
        if wl_input_val and wl_input_val not in st.session_state.watchlist:
            st.session_state.watchlist.append(wl_input_val)
            st.rerun()

    for wl_t in list(st.session_state.watchlist):
        rc1, rc2 = st.columns([4, 1])
        rc1.markdown(
            f"<span style='color:#e6edf3;font-size:.8rem'>{wl_t}</span>",
            unsafe_allow_html=True,
        )
        if rc2.button("✕", key=f"rm_{wl_t}", help=f"Remove {wl_t}"):
            st.session_state.watchlist.remove(wl_t)
            st.session_state.calendar_data = [
                c for c in st.session_state.calendar_data if c["ticker"] != wl_t
            ]
            st.rerun()

    if st.session_state.watchlist:
        if st.button("📅 Refresh Calendar", use_container_width=True, key="refresh_cal"):
            st.session_state.calendar_data = fetch_earnings_calendar(st.session_state.watchlist)
            st.rerun()

    # ── Check for New Filings ─────────────────────────────────────────────────
    st.divider()
    if st.button("🔔 Check for New Filings", use_container_width=True, key="check_filings_btn"):
        tickers_to_check = list(st.session_state.last_acc.keys())
        found = None
        for chk_t in tickers_to_check:
            result = check_new_filing(chk_t, st.session_state.last_acc.get(chk_t))
            if result["is_new"]:
                found = {"ticker": chk_t, "acc": result["acc"], "date": result["date"]}
                break
        if found:
            st.session_state.new_filing_banner = found
            st.rerun()
        else:
            st.toast("No new filings detected.", icon="✅")

    st.divider()
    st.markdown("""
    <p style="color:#8b949e;font-size:.72rem;line-height:1.6">
    <strong style="color:#e6edf3">Market data:</strong> Yahoo Finance<br>
    <strong style="color:#e6edf3">Filings:</strong> SEC EDGAR<br>
    <strong style="color:#e6edf3">Model:</strong> ProsusAI/FinBERT<br>
    <strong style="color:#e6edf3">Inference:</strong> Hugging Face API
    </p>""", unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="font-size:1.7rem;font-weight:700;color:#e6edf3;margin-bottom:2px">
  Earnings Call Sentiment Monitor
</h1>
<p style="color:#8b949e;font-size:.88rem;margin-top:0;margin-bottom:20px">
  Live stock data · FinBERT sentiment · 4-quarter history · Multi-ticker comparison
</p>""", unsafe_allow_html=True)


# ── Primary analysis pipeline ─────────────────────────────────────────────────

_auto_ticker = st.session_state.get("auto_run_ticker", "")
_is_auto_run = bool(_auto_ticker) and not run_button

if run_button or _auto_ticker:
    # Determine which ticker to analyse (manual button vs auto-trigger)
    _run_ticker = ticker_input if run_button else _auto_ticker
    # Consume the auto-run flag immediately so it doesn't loop
    if _is_auto_run:
        st.session_state.auto_run_ticker = ""

    if not _run_ticker:
        st.warning("Enter a ticker symbol in the sidebar before running analysis.")
    else:
        for k in ("stock", "sentiment", "meta", "transcript", "history", "keywords", "pdf_bytes"):
            st.session_state[k] = None

        _status_label = (
            f"🤖 Auto-analysing **{_run_ticker}** — new filing detected…"
            if _is_auto_run
            else f"🔍 Analysing **{_run_ticker}**…"
        )
        with st.status(_status_label, expanded=True, state="running") as status:
            try:
                st.write("📈 Fetching stock market data…")
                stock_data = fetch_stock_data(_run_ticker)
                st.write(f"✅ Stock data loaded — ${stock_data['price']:,.2f} ({stock_data['change_pct']:+.2f}%)")

                st.write("📡 Connecting to SEC EDGAR…")
                filing = fetch_transcript(_run_ticker)
                st.write(f"✅ Filing retrieved: **{filing['title']}**")

                st.write("🤖 Running FinBERT sentiment model on current quarter…")
                sentiment = run_sentiment_analysis(filing["text"])
                st.write(f"✅ Sentiment Score: **{sentiment['vibe_score']:+.2f}** ({sentiment['tone'].title()})")

                st.write("🔍 Extracting keyword insights…")
                keywords = get_keyword_insights(filing["text"])

                st.write("📊 Fetching and analysing last 4 quarters…")
                history = analyze_quarters(_run_ticker, n=4)
                st.write(f"✅ Historical analysis complete — {len(history)} quarters processed")

                st.write("📰 Fetching recent news…")
                news_items = fetch_recent_news(_run_ticker)
                st.write(f"✅ {len(news_items)} recent articles found")

                st.write("🔮 Generating forward-looking outlook…")
                outlook = generate_outlook(_run_ticker, sentiment, history, news_items)
                st.write(
                    f"✅ Outlook: **{outlook['signal'].title()}** "
                    f"(composite {outlook['composite_score']:+.2f})"
                )

                # Save core results BEFORE PDF (so they're available even if PDF fails)
                st.session_state.stock          = stock_data
                st.session_state.sentiment      = sentiment
                st.session_state.meta           = filing
                st.session_state.transcript     = filing["text"]
                st.session_state.history        = history
                st.session_state.keywords       = keywords
                st.session_state.news_items     = news_items
                st.session_state.outlook        = outlook
                st.session_state.primary_ticker = _run_ticker

                # Store accession number for new-filing change-detection
                acc = filing.get("acc", "")
                if acc:
                    st.session_state.last_acc = {
                        **st.session_state.last_acc,
                        _run_ticker: acc,
                    }

                st.write("📄 Generating PDF report…")
                try:
                    pdf_bytes = generate_pdf_report(
                        _run_ticker, stock_data, sentiment, history,
                        compare_data=st.session_state.compare or None,
                    )
                    st.session_state.pdf_bytes = pdf_bytes
                    st.write("✅ PDF report ready — download from sidebar")
                except Exception as pdf_err:
                    st.session_state.pdf_bytes = None
                    st.write(f"⚠️ PDF generation skipped: {str(pdf_err)[:120]}")

                _complete_label = (
                    f"✅ Auto-analysis complete for **{_run_ticker}** — new filing processed"
                    if _is_auto_run
                    else f"✅ Analysis complete for **{_run_ticker}**"
                )
                status.update(label=_complete_label, state="complete", expanded=False)
            except Exception as err:
                status.update(label="❌ Analysis failed", state="error", expanded=True)
                st.error(str(err))


# ── Comparison pipeline ───────────────────────────────────────────────────────

if compare_button:
    raw_tickers = [t.strip().upper() for t in compare_input.split(",") if t.strip()]
    raw_tickers = raw_tickers[:3]

    if not raw_tickers:
        st.warning("Enter at least one ticker in the Compare Tickers box.")
    else:
        compare_results = {}
        with st.status("🔄 Running comparison analysis…", expanded=True, state="running") as cstatus:
            for cticker in raw_tickers:
                try:
                    st.write(f"📡 Fetching {cticker}…")
                    cstock    = fetch_stock_data(cticker)
                    cfiling   = fetch_transcript(cticker)
                    csent     = run_sentiment_analysis(cfiling["text"])
                    compare_results[cticker] = {"stock": cstock, "sentiment": csent}
                    st.write(f"✅ {cticker}: Sentiment Score {csent['vibe_score']:+.2f}")
                except Exception as e:
                    st.write(f"⚠️ {cticker}: {str(e)[:80]}")
            st.session_state.compare = compare_results
            cstatus.update(label="✅ Comparison complete", state="complete", expanded=False)

    # Regenerate PDF to include the newly compared tickers (if primary analysis exists)
    _pticker = st.session_state.primary_ticker
    if st.session_state.stock and st.session_state.sentiment and _pticker:
        try:
            st.session_state.pdf_bytes = generate_pdf_report(
                _pticker,
                st.session_state.stock,
                st.session_state.sentiment,
                st.session_state.history or [],
                compare_data=st.session_state.compare or None,
            )
        except Exception as _pdf_err:
            st.warning(f"⚠️ PDF update failed: {str(_pdf_err)[:200]}")


# ── Auto-trigger: near-earnings new-filing check ──────────────────────────────
# Runs at most once every 30 min per session; only when a ticker is near its
# earnings date AND we have a previous accession number to compare against.

_auto_check_interval = 30 * 60  # seconds
if (
    ticker_input
    and ticker_input in st.session_state.last_acc
    and time.time() - st.session_state["_last_auto_check"] > _auto_check_interval
):
    cal_entry = next(
        (c for c in st.session_state.calendar_data if c["ticker"] == ticker_input),
        None,
    )
    near_earnings = (
        cal_entry is not None
        and cal_entry["days_until"] is not None
        and abs(cal_entry["days_until"]) <= 2
    )
    if near_earnings:
        st.session_state["_last_auto_check"] = time.time()
        result = check_new_filing(ticker_input, st.session_state.last_acc[ticker_input])
        if result["is_new"]:
            st.session_state.new_filing_banner = {
                "ticker": ticker_input,
                "acc":    result["acc"],
                "date":   result["date"],
                "auto":   True,
            }
            # Automatically re-run the full analysis pipeline for the new filing
            st.session_state.auto_run_ticker = ticker_input
            st.rerun()

# ── New-filing notification banner ────────────────────────────────────────────

if st.session_state.new_filing_banner:
    b = st.session_state.new_filing_banner
    col_banner, col_dismiss = st.columns([5, 1])
    with col_banner:
        _banner_msg = (
            f"🤖 **New 8-K filing detected for {b['ticker']}** (filed {b['date']}) — "
            "analysis has been automatically triggered and is running below."
            if b.get("auto")
            else f"📣 **New 8-K filing detected for {b['ticker']}** (filed {b['date']}) — "
            "a fresh earnings release is available. Run analysis to update your dashboard."
        )
        st.info(_banner_msg)
    with col_dismiss:
        if st.button("Dismiss", key="dismiss_banner"):
            st.session_state.new_filing_banner = None
            st.rerun()

# ── Earnings Calendar section (always visible when watchlist is non-empty) ────

if st.session_state.watchlist:
    # Refresh calendar data if stale (empty but watchlist has tickers)
    if not st.session_state.calendar_data:
        st.session_state.calendar_data = fetch_earnings_calendar(st.session_state.watchlist)

    with st.expander("📅 Earnings Calendar", expanded=True):
        cal_data = st.session_state.calendar_data
        if not cal_data:
            st.info("No calendar data available. Click **Refresh Calendar** in the sidebar.")
        else:
            today_dt = __import__("datetime").date.today()

            def _days_badge(days):
                if days is None:
                    return "<span style='color:#484f58'>—</span>"
                if days == 0:
                    return "<span style='color:#f85149;font-weight:700'>TODAY</span>"
                if days == 1:
                    return "<span style='color:#f85149;font-weight:600'>Tomorrow</span>"
                if days <= 7:
                    return f"<span style='color:#d29922;font-weight:600'>In {days}d</span>"
                if days < 0:
                    return f"<span style='color:#484f58'>{abs(days)}d ago</span>"
                return f"<span style='color:#2ea043'>In {days}d</span>"

            header_html = """
            <table style="width:100%;border-collapse:collapse;font-size:.8rem">
              <thead>
                <tr style="color:#8b949e;border-bottom:1px solid #21262d">
                  <th style="text-align:left;padding:6px 8px">Ticker</th>
                  <th style="text-align:left;padding:6px 8px">Company</th>
                  <th style="text-align:left;padding:6px 8px">Earnings Date</th>
                  <th style="text-align:left;padding:6px 8px">Countdown</th>
                  <th style="text-align:left;padding:6px 8px">EPS Est.</th>
                  <th style="text-align:left;padding:6px 8px">Rev Est.</th>
                </tr>
              </thead><tbody>"""
            rows_html = ""
            for c in cal_data:
                bg = "#161b22" if cal_data.index(c) % 2 == 0 else "#0d1117"
                rows_html += f"""
                <tr style="background:{bg};border-bottom:1px solid #21262d21">
                  <td style="padding:7px 8px;color:#58a6ff;font-weight:600">{c['ticker']}</td>
                  <td style="padding:7px 8px;color:#e6edf3">{c['company_name']}</td>
                  <td style="padding:7px 8px;color:#e6edf3">{c['earnings_date']}</td>
                  <td style="padding:7px 8px">{_days_badge(c['days_until'])}</td>
                  <td style="padding:7px 8px;color:#8b949e">{c['eps_estimate']}</td>
                  <td style="padding:7px 8px;color:#8b949e">{c['rev_estimate']}</td>
                </tr>"""
            st.markdown(
                header_html + rows_html + "</tbody></table>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Calendar data from Yahoo Finance · "
                "Dates may shift — always verify with official company IR."
            )

    st.markdown("<br>", unsafe_allow_html=True)

# ── Results layout ────────────────────────────────────────────────────────────

if st.session_state.sentiment and st.session_state.stock:
    stock     = st.session_state.stock
    sentiment = st.session_state.sentiment
    meta      = st.session_state.meta
    history   = st.session_state.history or []
    keywords  = st.session_state.keywords or {}
    compare   = st.session_state.compare or {}

    score = sentiment["vibe_score"]
    color = _score_color(score)
    label = _score_label(score)

    st.divider()

    # ── Top KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        sign = "+" if stock["change"] >= 0 else ""
        st.metric("💰 Stock Price", f"${stock['price']:,.2f}",
                  f"{sign}{stock['change']:.2f} ({stock['change_pct']:+.2f}%)")
    with c2:
        st.metric("📊 Sentiment Score", f"{score:+.2f}", label)
    with c3:
        st.metric("🟢 Positive", f"{sentiment['positive_pct']:.1f}%")
    with c4:
        st.metric("🔴 Negative", f"{sentiment['negative_pct']:.1f}%")
    with c5:
        st.metric("⚪ Neutral", f"{sentiment['neutral_pct']:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PDF download (prominent, in main area) ────────────────────────────────
    if st.session_state.pdf_bytes:
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state.pdf_bytes,
                file_name=f"pulse_ai_{stock['ticker']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ── 7 Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Market Overview",
        "💬 Earnings Sentiment",
        "📈 Sentiment History",
        "🔄 Ticker Compare",
        "📄 Raw Filing",
        "🔮 What's Next",
        "🤖 Ask Pulse",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — Market Overview
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f"""
        <h3 style="font-size:1.1rem;font-weight:600;color:#e6edf3;margin-bottom:4px">
          {stock['company_name']} &nbsp;<span style="color:#8b949e;font-size:.9rem">({stock['ticker']})</span>
        </h3>
        <p style="color:#8b949e;font-size:.78rem;margin-top:0">
          {stock['sector']}
        </p>""", unsafe_allow_html=True)

        kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
        with kc1: st.markdown(_kpi("Market Cap",     stock["market_cap"]),    unsafe_allow_html=True)
        with kc2: st.markdown(_kpi("P/E Ratio",      stock["pe_ratio"]),      unsafe_allow_html=True)
        with kc3: st.markdown(_kpi("52-Week High",   f"${stock['week_52_high']}"), unsafe_allow_html=True)
        with kc4: st.markdown(_kpi("52-Week Low",    f"${stock['week_52_low']}"),  unsafe_allow_html=True)
        with kc5: st.markdown(_kpi("Volume",         stock["volume"]),        unsafe_allow_html=True)
        with kc6: st.markdown(_kpi("Avg Volume",     stock["avg_volume"]),    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3-month price chart
        hist_df = stock.get("hist_df")
        if hist_df is not None and not hist_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["Close"],
                mode="lines", name="Close Price",
                line=dict(color="#58a6ff", width=2),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.07)",
                hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.2f}<extra></extra>",
            ))
            # Add volume as secondary bar
            fig.add_trace(go.Bar(
                x=hist_df.index, y=hist_df["Volume"],
                name="Volume", yaxis="y2",
                marker_color="rgba(88,166,255,0.15)",
            ))
            fig.update_layout(
                **_plotly_dark(),
                title=dict(text=f"{stock['ticker']} — 3-Month Price Chart", font=dict(color="#e6edf3", size=13)),
                yaxis=dict(title="Price (USD)", gridcolor="#21262d", tickprefix="$"),
                yaxis2=dict(
                    title="Volume", overlaying="y", side="right",
                    showgrid=False, tickformat=".2s",
                    title_font=dict(color="#484f58"), tickfont=dict(color="#484f58"),
                ),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e")),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Candlestick chart
            fig2 = go.Figure(go.Candlestick(
                x=hist_df.index,
                open=hist_df["Open"], high=hist_df["High"],
                low=hist_df["Low"], close=hist_df["Close"],
                increasing_line_color="#2ea043",
                decreasing_line_color="#f85149",
                name="OHLC",
            ))
            fig2.update_layout(
                **_plotly_dark(),
                title=dict(text=f"{stock['ticker']} — Candlestick (3 months)", font=dict(color="#e6edf3", size=13)),
                yaxis=dict(title="Price (USD)", gridcolor="#21262d", tickprefix="$"),
                xaxis=dict(rangeslider=dict(visible=False)),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Price chart data not available for this ticker.")


    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — Earnings Sentiment
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown(f"""
        <p style="color:#8b949e;font-size:.78rem;margin-bottom:12px">
          {meta['title']} · {meta['date']}
        </p>""", unsafe_allow_html=True)

        vib_col, don_col = st.columns([1, 1.2])

        with vib_col:
            st.metric("📊 Sentiment Score", f"{score:+.2f}", f"{label}")

            gauge_pct = round((score + 1) / 2 * 100, 1)
            st.markdown(f"""
            <div style="margin:12px 0 20px 0">
              <div style="font-size:.72rem;color:#8b949e;margin-bottom:4px">
                Sentiment Spectrum · <span style="color:{color};font-weight:600">{label}</span>
              </div>
              <div class="score-bar-wrap" style="height:14px">
                <div class="score-bar-fill"
                     style="width:{gauge_pct}%;background:linear-gradient(90deg,#f85149,{color})">
                </div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:.68rem;color:#484f58;margin-top:2px">
                <span>-1.0 Bearish</span><span>0</span><span>+1.0 Bullish</span>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(
                _bar_html(sentiment["positive_pct"], "#2ea043", "🟢 Positive")
                + _bar_html(sentiment["negative_pct"], "#f85149", "🔴 Negative")
                + _bar_html(sentiment["neutral_pct"],  "#8b949e", "⚪ Neutral"),
                unsafe_allow_html=True
            )
            st.caption(f"Analysed {sentiment['chunk_count']} text segments")

        with don_col:
            fig_pie = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[sentiment["positive_pct"], sentiment["negative_pct"], sentiment["neutral_pct"]],
                marker=dict(colors=["#2ea043", "#f85149", "#8b949e"]),
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(color="#e6edf3", size=12),
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_pie.add_annotation(
                text=f"{score:+.2f}", x=0.5, y=0.5,
                font=dict(size=22, color=color, family="Inter"),
                showarrow=False,
            )
            fig_pie.update_layout(
                **_plotly_dark(),
                title=dict(text="Sentiment Distribution", font=dict(color="#e6edf3", size=13)),
                showlegend=True,
                legend=dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="📡"):
            st.markdown(sentiment["summary"])

        # ── Keyword Insights ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.95rem;font-weight:600;color:#e6edf3;margin-bottom:4px">
          🔎 What drove the score?
        </div>
        <p style="color:#8b949e;font-size:.78rem;margin-top:0;margin-bottom:14px">
          Words and phrases most frequently used in this earnings release.
        </p>""", unsafe_allow_html=True)

        kw_left, kw_right = st.columns(2)

        # Treemap of top 25 most frequent words
        with kw_left:
            top_words = keywords.get("top_words", [])
            if top_words:
                tw_labels = [w for w, _ in top_words]
                tw_vals   = [c for _, c in top_words]
                # colour by frequency intensity
                tw_colors = [
                    f"rgba(88,166,255,{min(0.3 + c/max(tw_vals)*0.7, 1.0):.2f})"
                    for c in tw_vals
                ]
                fig_tm = go.Figure(go.Treemap(
                    labels=tw_labels,
                    parents=[""] * len(tw_labels),
                    values=tw_vals,
                    marker=dict(
                        colors=tw_vals,
                        colorscale=[[0, "#1f3b55"], [1, "#58a6ff"]],
                        showscale=False,
                    ),
                    textfont=dict(color="#e6edf3", size=13),
                    hovertemplate="<b>%{label}</b><br>Mentions: %{value}<extra></extra>",
                ))
                fig_tm.update_layout(
                    **_plotly_dark(),
                    title=dict(text="Top Keywords (by frequency)", font=dict(color="#e6edf3", size=13)),
                )
                fig_tm.layout.margin.update(l=0, r=0, t=35, b=0)
                st.plotly_chart(fig_tm, use_container_width=True)

        # Positive vs Negative keyword comparison bar
        with kw_right:
            pos_kw = keywords.get("positive", [])
            neg_kw = keywords.get("negative", [])
            if pos_kw or neg_kw:
                all_kw   = [(w, c, "#2ea043") for w, c in pos_kw] + \
                           [(w, -c, "#f85149") for w, c in neg_kw]
                all_kw.sort(key=lambda x: x[1])
                kw_words  = [x[0] for x in all_kw]
                kw_vals   = [x[1] for x in all_kw]
                kw_colors = [x[2] for x in all_kw]

                fig_kw = go.Figure(go.Bar(
                    x=kw_vals,
                    y=kw_words,
                    orientation="h",
                    marker_color=kw_colors,
                    hovertemplate="<b>%{y}</b><br>Mentions: %{x}<extra></extra>",
                ))
                fig_kw.add_vline(x=0, line_color="#484f58", line_width=1)
                fig_kw.update_layout(
                    **_plotly_dark(),
                    title=dict(text="Positive vs Negative Keywords", font=dict(color="#e6edf3", size=13)),
                    xaxis=dict(title="Mentions (negative = bearish signal)", gridcolor="#21262d"),
                    yaxis=dict(title=""),
                    margin=dict(l=10, r=10, t=35, b=30),
                )
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No sentiment keywords detected in this filing.")


    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 — Sentiment History
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:
        if not history:
            st.info("Historical data not available. Run the analysis to load 4-quarter history.")
        else:
            dates   = [r["date"] for r in history]
            scores  = [r["vibe_score"] for r in history]
            tones   = [r["tone"].title() for r in history]
            colors  = [_score_color(s) for s in scores]

            # Make quarter labels unique: if two filings fall in the same quarter,
            # append the filing date so each bar has a distinct label
            raw_labels = [r.get("quarter_label", r["date"]) for r in history]
            from collections import Counter
            label_counts = Counter(raw_labels)
            seen = {}
            qlabels = []
            for ql, d in zip(raw_labels, dates):
                if label_counts[ql] > 1:
                    seen[ql] = seen.get(ql, 0) + 1
                    qlabels.append(f"{ql} ({d})")
                else:
                    qlabels.append(ql)

            # Sentiment Score bar chart — dates as X, quarter labels as tick text
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=dates, y=scores,
                marker_color=colors,
                text=[f"{s:+.3f}" for s in scores],
                textposition="outside",
                textfont=dict(color="#e6edf3", size=11),
                hovertemplate="<b>%{text}</b><br>Sentiment Score: %{y:+.3f}<extra></extra>",
                name="Sentiment Score",
            ))
            fig_hist.add_hline(y=0,    line_color="#484f58", line_dash="dash", line_width=1)
            fig_hist.add_hline(y=0.25, line_color="#2ea043", line_dash="dot",  line_width=0.8, opacity=0.5)
            fig_hist.add_hline(y=-0.1, line_color="#f85149", line_dash="dot",  line_width=0.8, opacity=0.5)
            fig_hist.update_layout(
                **_plotly_dark(),
                title=dict(text="Sentiment Score - Last 4 Quarters", font=dict(color="#e6edf3", size=14)),
                yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1], zeroline=False, gridcolor="#21262d"),
                xaxis=dict(title="Quarter", tickvals=dates, ticktext=qlabels),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Line trend — dates as X, quarter labels as tick text
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=dates, y=scores, mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=9, color=colors, line=dict(color="#0d1117", width=2)),
                hovertemplate="<b>%{text}</b><br>Sentiment Score: %{y:+.3f}<extra></extra>",
                text=qlabels,
                name="Sentiment Score",
            ))
            fig_line.add_hline(y=0, line_color="#484f58", line_dash="dash", line_width=1)
            fig_line.update_layout(
                **_plotly_dark(),
                title=dict(text="Sentiment Trend Over Time", font=dict(color="#e6edf3", size=14)),
                yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1], gridcolor="#21262d"),
                xaxis=dict(title="Quarter", tickvals=dates, ticktext=qlabels),
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # Stacked sentiment % chart w/ quarter labels
            pos_vals = [r["positive_pct"] for r in history]
            neu_vals = [r["neutral_pct"]  for r in history]
            neg_vals = [r["negative_pct"] for r in history]

            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(name="Positive", x=dates, y=pos_vals, marker_color="#2ea043"))
            fig_stack.add_trace(go.Bar(name="Neutral",  x=dates, y=neu_vals, marker_color="#8b949e"))
            fig_stack.add_trace(go.Bar(name="Negative", x=dates, y=neg_vals, marker_color="#f85149"))
            fig_stack.update_layout(
                **_plotly_dark(),
                barmode="stack",
                title=dict(text="Sentiment Breakdown per Quarter (%)", font=dict(color="#e6edf3", size=14)),
                yaxis=dict(title="Percentage", gridcolor="#21262d"),
                xaxis=dict(title="Quarter", tickvals=dates, ticktext=qlabels),
                legend=dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

            # Per-quarter insight cards
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:.9rem;font-weight:600;color:#e6edf3;margin-bottom:10px">
              📋 Quarter-by-Quarter Insights
            </div>""", unsafe_allow_html=True)
            insight_cols = st.columns(len(history))
            for col, r in zip(insight_cols, history):
                vs  = r["vibe_score"]
                clr = _score_color(vs)
                lbl = _score_label(vs)
                ql  = r.get("quarter_label", r["date"])
                summ = r.get("summary", "")
                # First sentence only
                first_sentence = summ.split(".")[0] + "." if summ else ""
                with col:
                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid {clr}40;
                                border-top:3px solid {clr};border-radius:10px;
                                padding:14px 16px;height:100%">
                      <div style="font-size:.72rem;color:#8b949e;margin-bottom:2px">{ql}</div>
                      <div style="font-size:1.2rem;font-weight:700;color:{clr}">{vs:+.3f}</div>
                      <div style="font-size:.72rem;color:{clr};margin-bottom:8px">{lbl}</div>
                      <div style="font-size:.75rem;color:#e6edf3;line-height:1.5">{first_sentence}</div>
                      <div style="margin-top:8px;font-size:.7rem;color:#8b949e">
                        ✅ {r['positive_pct']:.0f}%&nbsp;&nbsp;
                        ❌ {r['negative_pct']:.0f}%&nbsp;&nbsp;
                        ⚪ {r['neutral_pct']:.0f}%
                      </div>
                    </div>""", unsafe_allow_html=True)

            # Summary table
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("📋 Historical Summary Table")
            rows = []
            for r in history:
                vs = r["vibe_score"]
                rows.append({
                    "Quarter":          r.get("quarter_label", r["date"]),
                    "Filing Date":      r["date"],
                    "Sentiment Score":  f"{vs:+.3f}",
                    "Tone":             r["tone"].title(),
                    "🟢 Positive":      f"{r['positive_pct']}%",
                    "🔴 Negative":      f"{r['negative_pct']}%",
                    "⚪ Neutral":       f"{r['neutral_pct']}%",
                    "Segments":         r["chunk_count"],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)


    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — Ticker Compare
    # ──────────────────────────────────────────────────────────────────────────
    with tab4:
        # Always include primary ticker in comparison
        all_compare = {}
        if st.session_state.sentiment and st.session_state.stock:
            all_compare[ticker_input] = {
                "stock":     st.session_state.stock,
                "sentiment": st.session_state.sentiment,
            }
        all_compare.update(compare)

        if len(all_compare) < 2:
            st.info(
                "Enter one or more tickers in the **Compare Tickers** sidebar box "
                "and click **🔄 Run Comparison** to see results here."
            )
        else:
            ctickers = list(all_compare.keys())
            cscores  = [all_compare[t]["sentiment"]["vibe_score"] for t in ctickers]
            ccolors  = [_score_color(s) for s in cscores]
            cprices  = [all_compare[t]["stock"]["price"] for t in ctickers]
            cchg     = [all_compare[t]["stock"]["change_pct"] for t in ctickers]

            # Vibe Score comparison bar chart
            fig_cmp = go.Figure(go.Bar(
                x=ctickers, y=cscores,
                marker_color=ccolors,
                text=[f"{s:+.3f}" for s in cscores],
                textposition="outside",
                textfont=dict(color="#e6edf3", size=13),
                width=0.5,
                hovertemplate="<b>%{x}</b><br>Vibe Score: %{y:+.3f}<extra></extra>",
            ))
            fig_cmp.add_hline(y=0, line_color="#484f58", line_dash="dash", line_width=1)
            fig_cmp.update_layout(
                **_plotly_dark(),
                title=dict(text="Vibe Score Comparison", font=dict(color="#e6edf3", size=14)),
                yaxis=dict(title="Vibe Score", range=[-1.1, 1.2], gridcolor="#21262d"),
                showlegend=False,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Radar chart: positive / negative / neutral breakdown
            fig_radar = go.Figure()
            radar_cats = ["Positive %", "Neutral %", "Negative %"]
            for t in ctickers:
                s = all_compare[t]["sentiment"]
                fig_radar.add_trace(go.Scatterpolar(
                    r=[s["positive_pct"], s["neutral_pct"], s["negative_pct"]],
                    theta=radar_cats,
                    fill="toself",
                    name=t,
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                **_plotly_dark(),
                polar=dict(
                    bgcolor="#161b22",
                    radialaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickcolor="#8b949e"),
                    angularaxis=dict(linecolor="#21262d", tickcolor="#e6edf3"),
                ),
                title=dict(text="Sentiment Radar", font=dict(color="#e6edf3", size=14)),
                showlegend=True,
                legend=dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Mini metric cards per ticker
            cols = st.columns(len(ctickers))
            for col, t in zip(cols, ctickers):
                s    = all_compare[t]["sentiment"]
                stk  = all_compare[t]["stock"]
                clr  = _score_color(s["vibe_score"])
                p_clr = "#2ea043" if stk["change_pct"] >= 0 else "#f85149"
                arrow = "▲" if stk["change_pct"] >= 0 else "▼"
                with col:
                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:14px 16px">
                      <div style="font-size:.8rem;color:#8b949e;margin-bottom:2px">{stk.get('company_name', t)}</div>
                      <div style="font-size:1.4rem;font-weight:700;color:#e6edf3">{t}</div>
                      <div style="font-size:1rem;color:#e6edf3;margin-top:4px">${stk['price']:,.2f}
                        <span style="color:{p_clr};font-size:.82rem">{arrow} {stk['change_pct']:+.2f}%</span>
                      </div>
                      <div style="margin-top:8px;padding:4px 10px;background:{clr}20;border:1px solid {clr};
                                  border-radius:6px;text-align:center">
                        <span style="color:{clr};font-weight:700;font-size:1.05rem">
                          {s['vibe_score']:+.2f}
                        </span>
                        <span style="color:#8b949e;font-size:.75rem;margin-left:6px">
                          {_score_label(s['vibe_score'])}
                        </span>
                      </div>
                    </div>""", unsafe_allow_html=True)

            # Price change comparison
            st.markdown("<br>", unsafe_allow_html=True)
            p_colors = ["#2ea043" if c >= 0 else "#f85149" for c in cchg]
            fig_price = go.Figure(go.Bar(
                x=ctickers, y=cchg,
                marker_color=p_colors,
                text=[f"{c:+.2f}%" for c in cchg],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
                width=0.5,
                hovertemplate="<b>%{x}</b><br>Change: %{y:+.2f}%<extra></extra>",
            ))
            fig_price.add_hline(y=0, line_color="#484f58", line_dash="dash", line_width=1)
            fig_price.update_layout(
                **_plotly_dark(),
                title=dict(text="Daily Price Change (%)", font=dict(color="#e6edf3", size=14)),
                yaxis=dict(title="Change %", gridcolor="#21262d"),
                showlegend=False,
            )
            st.plotly_chart(fig_price, use_container_width=True)


    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5 — Raw Filing
    # ──────────────────────────────────────────────────────────────────────────
    with tab5:
        transcript = st.session_state.transcript or ""
        word_count = len(transcript.split())
        st.caption(
            f"Full EX-99.1 text · {word_count:,} words · "
            f"Source: SEC EDGAR · Filing: {meta['date']}"
        )
        st.text_area("Transcript", value=transcript, height=520, label_visibility="collapsed")


    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6 — What's Next (Forward-Looking Outlook)
    # ──────────────────────────────────────────────────────────────────────────
    with tab6:
        outlook   = st.session_state.outlook
        news_data = st.session_state.news_items or []

        # Auto-fetch news if session state is empty but a ticker has been analysed
        if not news_data and st.session_state.stock:
            _loaded_ticker = st.session_state.stock["ticker"]
            with st.spinner(f"Fetching latest news for {_loaded_ticker}…"):
                try:
                    _fresh = fetch_recent_news(_loaded_ticker)
                    if _fresh:
                        st.session_state.news_items = _fresh
                        news_data = _fresh
                except Exception:
                    pass

        if not outlook:
            st.info("Run the analysis to generate a forward-looking outlook.")
        else:
            sig = outlook["signal"]
            sig_color = {"bullish": "#2ea043", "bearish": "#f85149", "neutral": "#d29922"}[sig]
            confidence_pct = round(outlook["confidence"] * 100)
            composite      = outlook["composite_score"]

            # ── Pulse Prediction hero card ────────────────────────────────────
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid {sig_color};border-top:4px solid {sig_color};
                        border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:20px">
              <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
                <div style="flex:1;min-width:180px">
                  <div style="font-size:.72rem;color:#8b949e;margin-bottom:2px">PULSE PREDICTION</div>
                  <div style="font-size:1.9rem;font-weight:800;color:{sig_color};letter-spacing:.02em">
                    {sig.upper()}
                  </div>
                  <div style="font-size:.8rem;color:#8b949e;margin-top:2px">
                    Composite signal · {confidence_pct}% model confidence
                  </div>
                </div>
                <div style="text-align:right;min-width:120px">
                  <div style="font-size:.72rem;color:#8b949e">Composite Score</div>
                  <div style="font-size:2rem;font-weight:700;color:{sig_color}">{composite:+.2f}</div>
                  <div style="font-size:.7rem;color:#484f58">range −1.0 to +1.0</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Contributing signals ──────────────────────────────────────────
            st.markdown("""
            <div style="font-size:.9rem;font-weight:600;color:#e6edf3;margin-bottom:10px">
              📐 Signal Breakdown
            </div>""", unsafe_allow_html=True)

            def _dir_badge(d):
                c = {"bullish": "#2ea043", "bearish": "#f85149", "neutral": "#d29922"}[d]
                icon = {"bullish": "▲", "bearish": "▼", "neutral": "●"}[d]
                return f"<span style='color:{c};font-weight:700'>{icon} {d.title()}</span>"

            sig_cols = st.columns(3)
            for sc, sig_item in zip(sig_cols, outlook["signals"]):
                d_color = {"bullish": "#2ea04320", "bearish": "#f8514920", "neutral": "#d2992220"}[sig_item["direction"]]
                d_border = {"bullish": "#2ea043", "bearish": "#f85149", "neutral": "#d29922"}[sig_item["direction"]]
                with sc:
                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid {d_border}40;
                                border-left:3px solid {d_border};border-radius:8px;
                                padding:12px 14px;height:100%">
                      <div style="font-size:.7rem;color:#8b949e;margin-bottom:2px">
                        {sig_item['name']} · <span style="color:#484f58">{sig_item['weight']}</span>
                      </div>
                      <div style="font-size:.95rem;font-weight:600;color:#e6edf3;margin-bottom:4px">
                        {sig_item['value']}
                      </div>
                      {_dir_badge(sig_item['direction'])}
                      <div style="font-size:.7rem;color:#484f58;margin-top:4px">{sig_item['detail']}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Narrative ─────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with st.chat_message("assistant", avatar="🔮"):
                st.markdown(outlook["summary"])
            st.caption("⚠️ This is an AI-generated signal for educational purposes only — not financial advice.")

            # ── News sentiment overview ───────────────────────────────────────
            ns = outlook.get("news_sentiment", {})
            if ns.get("total", 0) > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div style="font-size:.9rem;font-weight:600;color:#e6edf3;margin-bottom:10px">
                  📰 News Sentiment Snapshot
                </div>""", unsafe_allow_html=True)

                ns_cols = st.columns(4)
                ns_cols[0].metric("Overall Score",  f"{ns['score']:+.2f}")
                ns_cols[1].metric("🟢 Positive",    ns["positive"])
                ns_cols[2].metric("🔴 Negative",    ns["negative"])
                ns_cols[3].metric("⚪ Neutral",     ns["neutral"])

                # News sentiment mini bar
                if ns["total"] > 0:
                    p_pct = ns["positive"] / ns["total"] * 100
                    n_pct = ns["negative"] / ns["total"] * 100
                    e_pct = ns["neutral"]  / ns["total"] * 100
                    st.markdown(
                        _bar_html(p_pct, "#2ea043", "🟢 Positive News")
                        + _bar_html(n_pct, "#f85149", "🔴 Negative News")
                        + _bar_html(e_pct, "#8b949e", "⚪ Neutral News"),
                        unsafe_allow_html=True,
                    )

            # ── Recent news feed ──────────────────────────────────────────────
            if news_data:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-size:.9rem;font-weight:600;color:#e6edf3;margin-bottom:10px">
                  🗞️ Recent News Feed &nbsp;
                  <span style="font-size:.75rem;color:#8b949e;font-weight:400">
                    {len(news_data)} articles · Source: Yahoo Finance
                  </span>
                </div>""", unsafe_allow_html=True)

                for art in news_data:
                    title  = art.get("title", "")
                    pub    = art.get("publisher", "")
                    date_s = art.get("published_date", "")
                    link   = art.get("link", "")
                    link_html = (
                        f'<a href="{link}" target="_blank" '
                        f'style="color:#58a6ff;text-decoration:none;font-size:.85rem;font-weight:500">'
                        f'{title}</a>'
                        if link else
                        f'<span style="color:#e6edf3;font-size:.85rem;font-weight:500">{title}</span>'
                    )
                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                                padding:10px 14px;margin-bottom:8px">
                      {link_html}
                      <div style="font-size:.72rem;color:#484f58;margin-top:4px">
                        {pub}&nbsp;·&nbsp;{date_s}
                      </div>
                    </div>""", unsafe_allow_html=True)


    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7 — Ask Pulse (Gemini chatbot)
    # ── Fragment keeps the chat UI isolated so button/input events never
    #    trigger a full-page rerun that resets the active tab. ──────────────
    @st.fragment
    def _pulse_chat():
        # ── Resolve context ────────────────────────────────────────────────
        _ticker   = st.session_state.primary_ticker or ""
        _compare  = st.session_state.compare or {}   # {ticker: {stock, sentiment}}
        _all_tickers = [_ticker] + list(_compare.keys()) if _ticker else list(_compare.keys())

        # ── Dynamic suggested questions (cross-ticker when applicable) ─────
        if _compare:
            compare_label = " vs ".join([_ticker.upper()] + [t.upper() for t in _compare.keys()])
            suggested = [
                f"Compare sentiment of {compare_label}",
                f"Which ticker has the strongest outlook?",
                "What are the key revenue drivers mentioned?",
                "What risks did management highlight?",
            ]
        else:
            suggested = [
                "Why is sentiment lower this quarter?",
                "How does this quarter compare to the last 4?",
                "What are the key revenue drivers mentioned?",
                "What risks did management highlight?",
            ]

        # ── Suggested questions ────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:.78rem;color:#484f58;margin-bottom:6px'>"
            "💡 Suggested questions</p>",
            unsafe_allow_html=True,
        )
        cols_s = st.columns(len(suggested))
        for col, q in zip(cols_s, suggested):
            if col.button(q, key=f"suggest_{q[:20]}", use_container_width=True):
                st.session_state["_pulse_prefill"] = q
                st.rerun(scope="fragment")   # only reruns the fragment → tab stays

        # ── Active tickers badge ───────────────────────────────────────────
        if _all_tickers:
            badge_html = " &nbsp;".join(
                f"<span style='background:#21262d;border:1px solid #30363d;"
                f"border-radius:4px;padding:1px 6px;font-size:.75rem;"
                f"color:#58a6ff;font-family:monospace'>{t.upper()}</span>"
                for t in _all_tickers if t
            )
            st.markdown(
                f"<p style='margin:4px 0 12px;font-size:.78rem;color:#8b949e'>"
                f"Context: {badge_html}</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chat history display ───────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    parsed  = msg.get("parsed", {})
                    summary = parsed.get("short_summary", msg["content"])
                    drivers = parsed.get("drivers", [])
                    refs    = parsed.get("data_references", [])
                    st.markdown(
                        f"<div style='color:#e6edf3;font-size:.9rem'>{summary}</div>",
                        unsafe_allow_html=True,
                    )
                    if drivers:
                        st.markdown(
                            "<p style='color:#8b949e;font-size:.78rem;"
                            "font-weight:600;margin:10px 0 4px'>Key Drivers</p>",
                            unsafe_allow_html=True,
                        )
                        for d in drivers:
                            st.markdown(
                                f"<div style='font-size:.82rem;color:#c9d1d9;"
                                f"margin-bottom:3px'>• {d}</div>",
                                unsafe_allow_html=True,
                            )
                    if refs:
                        st.markdown(
                            "<p style='color:#8b949e;font-size:.78rem;"
                            "font-weight:600;margin:10px 0 4px'>Data References</p>",
                            unsafe_allow_html=True,
                        )
                        for r in refs:
                            st.markdown(
                                f"<div style='font-size:.78rem;color:#484f58;"
                                f"font-family:monospace;margin-bottom:2px'>{r}</div>",
                                unsafe_allow_html=True,
                            )

        # ── Chat input ─────────────────────────────────────────────────────
        prefill = st.session_state.pop("_pulse_prefill", "")
        _input_placeholder = (
            f"Ask about {' / '.join(t.upper() for t in _all_tickers if t)}…"
            if _all_tickers else "Load a ticker first to start chatting…"
        )
        user_q  = st.chat_input(_input_placeholder, key="pulse_chat_input")
        active_q = user_q or prefill

        if active_q:
            st.session_state.chat_history.append(
                {"role": "user", "content": active_q}
            )
            with st.chat_message("user"):
                st.markdown(active_q)

            _stock   = st.session_state.stock      or {}
            _sent    = st.session_state.sentiment  or {}
            _hist    = st.session_state.history    or []
            _outlook = st.session_state.outlook
            _filing  = st.session_state.transcript or ""

            # ── Fetch recent news for all loaded tickers ───────────────────
            _news: dict = {}
            try:
                if _ticker:
                    _news[_ticker] = fetch_recent_news(_ticker, n=5)
                for _ct in _compare:
                    _news[_ct] = fetch_recent_news(_ct, n=5)
            except Exception:
                pass   # news is optional — never block the chat on a fetch error

            with st.spinner("Pulse is thinking…"):
                try:
                    prompt = build_pulse_user_prompt(
                        ticker        = _ticker,
                        stock         = _stock,
                        sentiment     = _sent,
                        history       = _hist,
                        outlook       = _outlook,
                        filing_text   = _filing,
                        user_question = active_q,
                        compare_data  = _compare or None,
                        news_items    = _news or None,
                    )
                    result = call_pulse_llm(prompt)
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": result.get("short_summary", ""),
                        "parsed":  result,
                    })
                    with st.chat_message("assistant"):
                        _summary = result.get("short_summary", "")
                        _drivers = result.get("drivers", [])
                        _refs    = result.get("data_references", [])
                        st.markdown(
                            f"<div style='color:#e6edf3;font-size:.9rem'>{_summary}</div>",
                            unsafe_allow_html=True,
                        )
                        if _drivers:
                            st.markdown(
                                "<p style='color:#8b949e;font-size:.78rem;"
                                "font-weight:600;margin:10px 0 4px'>Key Drivers</p>",
                                unsafe_allow_html=True,
                            )
                            for _d in _drivers:
                                st.markdown(
                                    f"<div style='font-size:.82rem;color:#c9d1d9;"
                                    f"margin-bottom:3px'>• {_d}</div>",
                                    unsafe_allow_html=True,
                                )
                        if _refs:
                            st.markdown(
                                "<p style='color:#8b949e;font-size:.78rem;"
                                "font-weight:600;margin:10px 0 4px'>Data References</p>",
                                unsafe_allow_html=True,
                            )
                            for _r in _refs:
                                st.markdown(
                                    f"<div style='font-size:.78rem;color:#484f58;"
                                    f"font-family:monospace;margin-bottom:2px'>{_r}</div>",
                                    unsafe_allow_html=True,
                                )
                except Exception as e:
                    _err = str(e)
                    if "429" in _err or "RESOURCE_EXHAUSTED" in _err:
                        _msg = "⚠️ Gemini rate limit reached. Please wait ~1 minute and try again."
                    elif "404" in _err or "NOT_FOUND" in _err:
                        _msg = "⚠️ Gemini model not found. Check your API key project settings."
                    elif "API_KEY" in _err or "api_key" in _err.lower():
                        _msg = "⚠️ Gemini API key is missing or invalid. Check your .env file."
                    elif "JSON" in _err or "non-JSON" in _err:
                        _msg = "⚠️ Pulse received an unexpected response format. Please try again."
                    else:
                        _msg = f"⚠️ Pulse encountered an error: {_err[:120]}"
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": _msg, "parsed": {}}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(_msg)

        # ── Clear chat button ──────────────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", key="clear_pulse_chat"):
                st.session_state.chat_history = []
                st.rerun(scope="fragment")

    with tab7:
        _loaded_compare = st.session_state.compare or {}
        _multi_note = (
            f" Comparing: **{', '.join(k.upper() for k in _loaded_compare)}**."
            if _loaded_compare else ""
        )
        st.markdown(f"""
        <h3 style="font-size:1.1rem;font-weight:600;color:#e6edf3;margin-bottom:4px">
          🤖 Ask Pulse
        </h3>
        <p style="color:#8b949e;font-size:.82rem;margin-top:0;margin-bottom:16px">
          Ask any question about earnings, sentiment, market data, or recent news for
          all loaded tickers. Pulse draws on filing excerpts, FinBERT scores, and
          live Yahoo Finance headlines.{_multi_note}
        </p>""", unsafe_allow_html=True)
        _pulse_chat()


# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="border:1px dashed #21262d;border-radius:12px;padding:3rem 2rem;
                text-align:center;color:#484f58;margin-top:2rem">
      <div style="font-size:2.5rem;margin-bottom:12px">📡</div>
      <p style="font-size:1rem;font-weight:500;color:#8b949e;margin:0 0 6px 0">
        No analysis loaded
      </p>
      <p style="font-size:.85rem;margin:0">
        Enter a ticker in the sidebar and click
        <strong style="color:#e6edf3">⚡ Run Analysis</strong> to begin.
      </p>
    </div>""", unsafe_allow_html=True)
