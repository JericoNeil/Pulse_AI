"""
main.py - Pulse AI v2
5-tab financial sentiment intelligence dashboard.
"""

import plotly.graph_objects as go
import streamlit as st

from logic import (
    analyze_quarters,
    fetch_stock_data,
    fetch_transcript,
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
    "stock":      None,  # stock data
    "sentiment":  None,  # current quarter sentiment
    "meta":       None,  # filing metadata
    "transcript": None,  # full text
    "history":    None,  # 4-quarter list
    "keywords":   None,  # keyword insights dict
    "compare":    None,  # {ticker: {stock, sentiment}} dict
    "pdf_bytes":  None,
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

if run_button:
    if not ticker_input:
        st.warning("Enter a ticker symbol in the sidebar before running analysis.")
    else:
        for k in ("stock", "sentiment", "meta", "transcript", "history", "keywords", "pdf_bytes"):
            st.session_state[k] = None

        with st.status(f"🔍 Analysing **{ticker_input}**…", expanded=True, state="running") as status:
            try:
                st.write("📈 Fetching stock market data…")
                stock_data = fetch_stock_data(ticker_input)
                st.write(f"✅ Stock data loaded — ${stock_data['price']:,.2f} ({stock_data['change_pct']:+.2f}%)")

                st.write("📡 Connecting to SEC EDGAR…")
                filing = fetch_transcript(ticker_input)
                st.write(f"✅ Filing retrieved: **{filing['title']}**")

                st.write("🤖 Running FinBERT sentiment model on current quarter…")
                sentiment = run_sentiment_analysis(filing["text"])
                st.write(f"✅ Sentiment Score: **{sentiment['vibe_score']:+.2f}** ({sentiment['tone'].title()})")

                st.write("🔍 Extracting keyword insights…")
                keywords = get_keyword_insights(filing["text"])

                st.write("📊 Fetching and analysing last 4 quarters…")
                history = analyze_quarters(ticker_input, n=4)
                st.write(f"✅ Historical analysis complete — {len(history)} quarters processed")

                # Save core results BEFORE PDF (so they're available even if PDF fails)
                st.session_state.stock      = stock_data
                st.session_state.sentiment  = sentiment
                st.session_state.meta       = filing
                st.session_state.transcript = filing["text"]
                st.session_state.history    = history
                st.session_state.keywords   = keywords

                st.write("📄 Generating PDF report…")
                try:
                    pdf_bytes = generate_pdf_report(ticker_input, stock_data, sentiment, history)
                    st.session_state.pdf_bytes = pdf_bytes
                    st.write("✅ PDF report ready — download from sidebar")
                except Exception as pdf_err:
                    st.session_state.pdf_bytes = None
                    st.write(f"⚠️ PDF generation skipped: {str(pdf_err)[:120]}")

                status.update(
                    label=f"✅ Analysis complete for **{ticker_input}**",
                    state="complete", expanded=False
                )
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

    # ── 5 Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview",
        "💬 Earnings Sentiment",
        "📈 Sentiment History",
        "🔄 Ticker Compare",
        "📄 Raw Filing",
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
                    titlefont=dict(color="#484f58"), tickfont=dict(color="#484f58"),
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
