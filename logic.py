"""
logic.py - Pulse AI v2
Data acquisition (SEC EDGAR + yfinance) and FinBERT sentiment pipeline.
"""

import io
import os
import re
import datetime
from typing import Optional

import requests
import yfinance as yf
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

HF_TOKEN      = os.getenv("HF_TOKEN")
FINBERT_MODEL = "ProsusAI/finbert"
HF_API_URL    = f"https://router.huggingface.co/hf-inference/models/{FINBERT_MODEL}"

EDGAR_HEADERS = {
    "User-Agent": "PulseAI contact@pulseai.com",
    "Accept-Encoding": "gzip, deflate",
}


# ════════════════════════════════════════════════════════════════════════════
# yfinance — Stock Market Data
# ════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker: str) -> dict:
    """
    Fetch live stock data via yfinance.

    Returns dict with keys:
        price, change, change_pct, prev_close, market_cap, pe_ratio,
        week_52_high, week_52_low, volume, avg_volume,
        company_name, sector, currency,
        hist_df (DataFrame: 3-month OHLCV)
    """
    ticker = ticker.upper()
    tk = yf.Ticker(ticker)

    try:
        info = tk.info
    except Exception:
        info = {}

    try:
        hist = tk.history(period="3mo")
    except Exception:
        hist = None

    price      = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    prev_close = info.get("previousClose", 0)
    change     = round(price - prev_close, 2) if price and prev_close else 0
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0

    def _fmt_large(n):
        if not n:
            return "N/A"
        if n >= 1e12:
            return f"${n/1e12:.2f}T"
        if n >= 1e9:
            return f"${n/1e9:.2f}B"
        if n >= 1e6:
            return f"${n/1e6:.2f}M"
        return f"${n:,.0f}"

    return {
        "ticker"       : ticker,
        "company_name" : info.get("longName", ticker),
        "sector"       : info.get("sector", "N/A"),
        "currency"     : info.get("currency", "USD"),
        "price"        : price,
        "change"       : change,
        "change_pct"   : change_pct,
        "prev_close"   : prev_close,
        "market_cap"   : _fmt_large(info.get("marketCap")),
        "pe_ratio"     : round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
        "week_52_high" : info.get("fiftyTwoWeekHigh", "N/A"),
        "week_52_low"  : info.get("fiftyTwoWeekLow", "N/A"),
        "volume"       : f"{info.get('volume', 0):,}" if info.get("volume") else "N/A",
        "avg_volume"   : f"{info.get('averageVolume', 0):,}" if info.get("averageVolume") else "N/A",
        "hist_df"      : hist,
    }


# ════════════════════════════════════════════════════════════════════════════
# SEC EDGAR — Transcript / Earnings Release Fetching
# ════════════════════════════════════════════════════════════════════════════

def _get_cik(ticker: str) -> str:
    url  = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise RuntimeError(
        f"Ticker '{ticker}' not found in SEC EDGAR. Please verify the symbol."
    )


def _find_exhibit_filename(sgml_text: str, exhibit_type: str = "EX-99.1") -> Optional[str]:
    pattern = re.compile(
        rf"<TYPE>{re.escape(exhibit_type)}\s*\n<SEQUENCE>\d+\s*\n<FILENAME>(.+)",
        re.MULTILINE,
    )
    m = pattern.search(sgml_text)
    return m.group(1).strip() if m else None


def _parse_exhibit_text(sgml_text: str, exhibit_filename: str) -> str:
    """Extract and strip HTML from an exhibit embedded in the SGML bundle."""
    raw_html = sgml_text
    # Try narrowing to just the exhibit section
    fname_re = re.escape(exhibit_filename)
    m = re.search(rf"<FILENAME>{fname_re}.+?<TEXT>(.*?)</DOCUMENT>", sgml_text, re.DOTALL)
    if m:
        raw_html = m.group(1)
    text = re.sub(r"<[^>]+>",  " ", raw_html)
    text = re.sub(r"&nbsp;",   " ", text)
    text = re.sub(r"&amp;",    "&", text)
    text = re.sub(r"&lt;",     "<", text)
    text = re.sub(r"&gt;",     ">", text)
    text = re.sub(r"&#\d+;",   " ", text)
    text = re.sub(r"\s{3,}",   " ", text).strip()
    return text


def _download_exhibit(cik_int: int, acc_nodash: str, filename: str) -> str:
    """Download an exhibit HTML file directly and strip tags."""
    url  = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{filename}"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
    resp.raise_for_status()
    return _parse_exhibit_text(resp.text, filename)


def fetch_last_n_filings(ticker: str, n: int = 4) -> list:
    """
    Fetch the last *n* 8-K earnings releases for *ticker* from SEC EDGAR.

    Returns a list of dicts: [{title, text, date, quarter}, ...]
    ordered most-recent first.
    """
    ticker  = ticker.upper()
    cik     = _get_cik(ticker)
    cik_int = int(cik)

    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp    = requests.get(sub_url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    recent  = resp.json()["filings"]["recent"]

    forms = recent["form"]
    dates = recent["filingDate"]
    accs  = recent["accessionNumber"]

    filings = []
    for form, date, acc in zip(forms, dates, accs):
        if form == "8-K":
            filings.append({"date": date, "acc": acc})
        if len(filings) >= n:
            break

    if not filings:
        raise RuntimeError(f"No 8-K filings found for '{ticker}' on SEC EDGAR.")

    results = []
    for i, filing in enumerate(filings):
        acc_dashes = filing["acc"]
        acc_nodash = acc_dashes.replace("-", "")
        date       = filing["date"]

        try:
            sgml_url  = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{acc_nodash}/{acc_dashes}.txt"
            )
            sgml_resp = requests.get(sgml_url, headers=EDGAR_HEADERS, timeout=20)
            sgml_resp.raise_for_status()
            sgml_text = sgml_resp.text

            exhibit_filename = _find_exhibit_filename(sgml_text, "EX-99.1")

            if exhibit_filename:
                text = _download_exhibit(cik_int, acc_nodash, exhibit_filename)
            else:
                # Fallback: try to extract inline from SGML
                fnames = re.findall(r"<FILENAME>(.+\.htm)", sgml_text)
                if fnames:
                    text = _download_exhibit(cik_int, acc_nodash, fnames[0].strip())
                else:
                    text = _parse_exhibit_text(sgml_text, "")

            if len(text.split()) < 80:
                continue

            quarter_label = f"Q{i+1} Filing" if i > 0 else "Most Recent"
            results.append({
                "title"  : f"{ticker} Earnings Release · {date}",
                "text"   : text,
                "date"   : date,
                "quarter": quarter_label,
            })
        except Exception:
            continue   # skip filings we can't parse

    if not results:
        raise RuntimeError(
            f"Could not extract readable text from any of the last {n} 8-K filings for '{ticker}'."
        )
    return results


def fetch_transcript(ticker: str) -> dict:
    """Fetch the single most-recent earnings release. Wrapper for single-ticker flow."""
    filings = fetch_last_n_filings(ticker, n=1)
    return filings[0]


# ════════════════════════════════════════════════════════════════════════════
# FinBERT Sentiment Analysis
# ════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, max_words: int = 300) -> list:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i: i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _label_to_score(label: str, score: float) -> float:
    label = label.lower()
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def run_sentiment_analysis(transcript_text: str) -> dict:
    """
    Run ProsusAI/FinBERT on *transcript_text* in chunks.
    Returns: vibe_score, positive_pct, negative_pct, neutral_pct, summary, chunk_count.
    """
    if not transcript_text or not transcript_text.strip():
        raise RuntimeError("Transcript text is empty — cannot run sentiment analysis.")

    chunks = _chunk_text(transcript_text)
    if not chunks:
        raise RuntimeError("Transcript produced no text chunks after splitting.")

    chunks = chunks[:30]
    label_counts   = {"positive": 0, "negative": 0, "neutral": 0}
    weighted_score = 0.0
    hf_headers     = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    for chunk in chunks:
        try:
            payload = {"inputs": chunk, "parameters": {"truncation": True}}
            resp    = requests.post(HF_API_URL, headers=hf_headers, json=payload, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            raw     = resp.json()
            results = raw[0] if isinstance(raw, list) and isinstance(raw[0], list) else raw
            top     = max(results, key=lambda r: r["score"])
            label   = top["label"].lower()
            conf    = top["score"]
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Hugging Face Inference API error: {exc}") from exc

        label_counts[label] = label_counts.get(label, 0) + 1
        weighted_score += _label_to_score(label, conf)

    n          = len(chunks)
    vibe_score = round(weighted_score / n, 4)
    total      = sum(label_counts.values())
    pos_pct    = round(label_counts["positive"] / total * 100, 1)
    neg_pct    = round(label_counts["negative"] / total * 100, 1)
    neu_pct    = round(label_counts["neutral"]  / total * 100, 1)

    if vibe_score >= 0.6:
        tone, ceo_note = "strongly optimistic", (
            "The CEO conveyed high confidence, using forward-looking language "
            "and growth-oriented phrasing throughout most of the release."
        )
    elif vibe_score >= 0.25:
        tone, ceo_note = "moderately positive", (
            "Management struck a constructive tone, acknowledging challenges "
            "while emphasising strategic progress and near-term catalysts."
        )
    elif vibe_score >= -0.1:
        tone, ceo_note = "broadly neutral", (
            "The language was measured and factual; leadership balanced risks "
            "and opportunities without leaning strongly in either direction."
        )
    elif vibe_score >= -0.4:
        tone, ceo_note = "cautiously negative", (
            "Leadership signalled concern in several segments, hedging on guidance "
            "and highlighting macro headwinds as a recurring theme."
        )
    else:
        tone, ceo_note = "distinctly bearish", (
            "The release shows markedly negative sentiment — executives repeatedly "
            "flagged risk factors, declining metrics, or restructuring pressures."
        )

    summary = (
        f"**Overall tone:** {tone} (Vibe Score: {vibe_score:+.2f}).\n\n"
        f"{ceo_note}\n\n"
        f"**Sentiment breakdown across {n} analysed segments:** "
        f"🟢 Positive {pos_pct}% · 🔴 Negative {neg_pct}% · ⚪ Neutral {neu_pct}%."
    )

    return {
        "vibe_score"  : vibe_score,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct" : neu_pct,
        "summary"     : summary,
        "chunk_count" : n,
        "tone"        : tone,
    }


# ─── Quarter label helper ────────────────────────────────────────────────────

def _date_to_quarter(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to 'Q1 2026' style label."""
    try:
        d = datetime.date.fromisoformat(date_str[:10])
        q = (d.month - 1) // 3 + 1
        return f"Q{q} {d.year}"
    except Exception:
        return date_str


# ─── Keyword insight extractor ───────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "this", "these", "those", "it", "its", "we", "our", "us", "they",
    "their", "them", "he", "she", "his", "her", "i", "my", "you", "your",
    "not", "no", "so", "if", "than", "then", "also", "into", "which",
    "about", "up", "out", "more", "all", "per", "over", "time", "year",
    "quarter", "million", "billion", "percent", "q", "fy", "vs",
    "compared", "including", "during", "through", "well", "new", "total",
    "based", "net", "non", "approximately", "respectively", "increased",
    "decreased", "both", "each", "second", "first", "third", "fourth",
    "due", "continued", "primarily", "within", "current", "prior",
    "fiscal", "ended", "results", "following", "certain", "related",
}

# Financial‐sentiment keyword lists (lightweight, no extra deps)
_POSITIVE_FINANCE_WORDS = {
    "growth", "record", "strong", "exceeded", "beat", "raised", "profit",
    "revenue", "increase", "expansion", "gains", "outperform", "robust",
    "momentum", "demand", "innovation", "leadership", "efficiency",
    "margin", "cash", "dividend", "accelerated", "opportunity",
    "positive", "improved", "milestone", "achieved", "launch", "delivered",
}
_NEGATIVE_FINANCE_WORDS = {
    "decline", "loss", "risk", "challenging", "headwinds", "uncertainty",
    "weakness", "pressure", "lower", "reduced", "miss", "impairment",
    "restructuring", "layoffs", "debt", "deficit", "slowdown", "concern",
    "volatile", "difficult", "downturn", "shortage", "inflation",
    "competition", "lawsuit", "penalty", "warned", "delay",
}


def get_keyword_insights(text: str, top_n: int = 25) -> dict:
    """
    Extract top keywords from filing text.

    Returns:
        top_words : list of (word, count) — all types, sorted by freq
        positive  : list of (word, count) — positive-leaning words
        negative  : list of (word, count) — negative-leaning words
    """
    from collections import Counter

    # Clean and tokenise
    words = re.sub(r"[^a-z\s]", "", text.lower()).split()
    filtered = [
        w for w in words
        if len(w) > 3 and w not in _STOP_WORDS
    ]

    counts = Counter(filtered)
    top_words = counts.most_common(top_n)

    pos_words = [(w, c) for w, c in counts.most_common(200)
                 if w in _POSITIVE_FINANCE_WORDS][:15]
    neg_words = [(w, c) for w, c in counts.most_common(200)
                 if w in _NEGATIVE_FINANCE_WORDS][:15]

    return {
        "top_words": top_words,   # [(word, count), ...]
        "positive":  pos_words,
        "negative":  neg_words,
    }


def analyze_quarters(ticker: str, n: int = 4, progress_callback=None) -> list:
    """
    Fetch last *n* 8-K filings and run sentiment on each.
    Calls progress_callback(i, total, label) between steps if provided.

    Returns list of dicts: [{date, quarter_label, vibe_score, tone, ...}, ...]
    """
    filings = fetch_last_n_filings(ticker, n=n)
    results = []

    for i, filing in enumerate(filings):
        if progress_callback:
            progress_callback(i, len(filings), f"Analysing filing {i+1}/{len(filings)} ({filing['date']})")
        try:
            sentiment = run_sentiment_analysis(filing["text"])
            results.append({
                "date"         : filing["date"],
                "quarter_label": _date_to_quarter(filing["date"]),
                "title"        : filing["title"],
                "vibe_score"   : sentiment["vibe_score"],
                "tone"         : sentiment["tone"],
                "positive_pct" : sentiment["positive_pct"],
                "negative_pct" : sentiment["negative_pct"],
                "neutral_pct"  : sentiment["neutral_pct"],
                "chunk_count"  : sentiment["chunk_count"],
                "summary"      : sentiment["summary"],
            })
        except Exception:
            continue

    if progress_callback:
        progress_callback(len(filings), len(filings), "Complete")
    return results


# ════════════════════════════════════════════════════════════════════════════
# PDF Report Generator
# ════════════════════════════════════════════════════════════════════════════

def _score_to_rgb(score: float):
    if score >= 0.25:
        return (46, 160, 67)    # green
    if score >= -0.1:
        return (210, 153, 34)   # amber
    return (248, 81, 73)        # red


def generate_pdf_report(
    ticker: str,
    stock_data: dict,
    current_sentiment: dict,
    history: list,
) -> bytes:
    """
    Generate a branded PDF report and return as bytes.
    Compatible with Streamlit's st.download_button.
    Uses fpdf2 v2 API (XPos/YPos enums).
    """
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(13, 17, 23)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(230, 237, 243)
    pdf.set_y(8)
    pdf.cell(0, 12, "PULSE AI", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(0, 6, "Earnings Call Sentiment Intelligence Report", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Meta ──────────────────────────────────────────────────────────────────
    pdf.set_y(36)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"{stock_data.get('company_name', ticker)} ({ticker})",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5,
             f"Sector: {stock_data.get('sector', 'N/A')}   |   "
             f"Generated: {datetime.date.today().strftime('%B %d, %Y')}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Stock Snapshot ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(13, 17, 23)
    pdf.cell(0, 7, "MARKET SNAPSHOT",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(88, 166, 255)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    price   = stock_data.get("price", 0)
    chg     = stock_data.get("change", 0)
    chg_pct = stock_data.get("change_pct", 0)
    arrow   = "+" if chg >= 0 else ""
    col_r   = (46, 160, 67) if chg >= 0 else (248, 81, 73)

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(13, 17, 23)
    pdf.cell(0, 10, f"${price:,.2f}  ({arrow}{chg_pct:.2f}%)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    metrics = [
        ("Market Cap",  stock_data.get("market_cap", "N/A")),
        ("P/E Ratio",   str(stock_data.get("pe_ratio", "N/A"))),
        ("52W High",    f"${stock_data.get('week_52_high', 'N/A')}"),
        ("52W Low",     f"${stock_data.get('week_52_low', 'N/A')}"),
        ("Volume",      stock_data.get("volume", "N/A")),
        ("Avg Volume",  stock_data.get("avg_volume", "N/A")),
    ]
    for i, (lbl, val) in enumerate(metrics):
        pdf.set_text_color(100, 100, 100)
        pdf.cell(32, 5, f"{lbl}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(13, 17, 23)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(40, 5, str(val), new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 9)
        if i % 2 == 1 or i == len(metrics) - 1:
            pdf.ln(6)
    pdf.ln(6)

    # ── Current Vibe Score ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(13, 17, 23)
    pdf.cell(0, 7, "SENTIMENT VIBE SCORE - CURRENT QUARTER",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(88, 166, 255)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    score = current_sentiment.get("vibe_score", 0)
    tone  = current_sentiment.get("tone", "N/A")
    r, g, b = _score_to_rgb(score)

    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12,
             f"  Vibe Score: {score:+.2f}  |  {tone.title()}",
             fill=True, align="L",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5,
             f"Positive: {current_sentiment.get('positive_pct', 0)}%   "
             f"Negative: {current_sentiment.get('negative_pct', 0)}%   "
             f"Neutral: {current_sentiment.get('neutral_pct', 0)}%   "
             f"Segments analysed: {current_sentiment.get('chunk_count', 0)}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # AI Summary (strip markdown)
    summary_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", current_sentiment.get("summary", ""))
    summary_clean = re.sub(r"[🟢🔴⚪]", "", summary_clean).strip()
    summary_clean = summary_clean.encode("latin-1", errors="replace").decode("latin-1")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, summary_clean)
    pdf.ln(6)

    # ── Historical Sentiment ──────────────────────────────────────────────────
    if history:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(13, 17, 23)
        pdf.cell(0, 7, "SENTIMENT HISTORY - LAST 4 QUARTERS",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(88, 166, 255)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        col_w   = [15, 35, 30, 40, 25, 25, 20]
        headers = ["#", "Date", "Vibe Score", "Tone", "Pos %", "Neg %", "Segs"]

        pdf.set_fill_color(22, 27, 34)
        pdf.set_text_color(230, 237, 243)
        pdf.set_font("Helvetica", "B", 8)
        for h, w in zip(headers, col_w):
            pdf.cell(w, 7, h, fill=True, border=1,
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(7)

        pdf.set_font("Helvetica", "", 8)
        for i, row in enumerate(history):
            fill = i % 2 == 0
            pdf.set_fill_color(240, 242, 245) if fill else pdf.set_fill_color(255, 255, 255)
            vs = row.get("vibe_score", 0)
            r2, g2, b2 = _score_to_rgb(vs)
            vals = [
                str(i + 1),
                row.get("date", ""),
                f"{vs:+.3f}",
                row.get("tone", "").title(),
                f"{row.get('positive_pct', 0)}%",
                f"{row.get('negative_pct', 0)}%",
                str(row.get("chunk_count", 0)),
            ]
            for j, (val, w) in enumerate(zip(vals, col_w)):
                pdf.set_text_color(r2, g2, b2) if j == 2 else pdf.set_text_color(13, 17, 23)
                pdf.cell(w, 6, val, fill=fill, border=1,
                         new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(6)

    pdf.ln(8)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(0, 5,
             "Generated by Pulse AI  |  ProsusAI/FinBERT & SEC EDGAR  |  "
             "For educational purposes only. Not financial advice.",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())

