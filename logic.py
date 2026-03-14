"""
logic.py - Pulse AI v2
Data acquisition (SEC EDGAR + yfinance) and FinBERT sentiment pipeline.
"""

import io
import json
import os
import re
import datetime
from typing import Optional

import requests
import yfinance as yf
from google import genai
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

HF_TOKEN       = os.getenv("HF_TOKEN")
FINBERT_MODEL  = "ProsusAI/finbert"
HF_API_URL     = f"https://router.huggingface.co/hf-inference/models/{FINBERT_MODEL}"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    Fetch the last *n* 8-K **earnings releases** for *ticker* from SEC EDGAR.

    Only considers 8-Ks that:
      1. Are filed under Item 2.02 (Results of Operations) — the earnings item, OR
         have no items metadata (older filings) but still contain an EX-99.1 exhibit.
      2. Contain an EX-99.1 press-release exhibit with meaningful text (≥200 words).

    This prevents non-earnings 8-Ks (director changes, agreements, etc.) from being
    picked up and yielding misleadingly short transcripts.

    Returns a list of dicts: [{title, text, date, quarter, acc}, ...]
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
    # 'items' field lists the 8-K item numbers, e.g. "2.02" for earnings releases.
    # Not always present in older filings — default to empty string.
    items_list = recent.get("items", [""] * len(forms))

    # Collect candidate 8-K filings, preferring Item 2.02 (earnings) ones.
    # We gather up to n*6 candidates to give ourselves enough to find n valid ones.
    candidates = []
    for form, date, acc, items in zip(forms, dates, accs, items_list):
        if form != "8-K":
            continue
        # items may be a comma-separated string like "2.02,9.01" or just "2.02"
        item_nums = [i.strip() for i in str(items).split(",")]
        is_earnings = "2.02" in item_nums
        candidates.append({"date": date, "acc": acc, "is_earnings": is_earnings})
        if len(candidates) >= n * 6:
            break

    if not candidates:
        raise RuntimeError(f"No 8-K filings found for '{ticker}' on SEC EDGAR.")

    # Sort: earnings releases (2.02) first, then everything else — preserving
    # date order within each group so we still get most-recent first.
    earnings_filings = [c for c in candidates if c["is_earnings"]]
    other_filings    = [c for c in candidates if not c["is_earnings"]]
    ordered = earnings_filings + other_filings

    results = []
    result_index = 0
    for filing in ordered:
        if len(results) >= n:
            break

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

            # Skip filings without an EX-99.1 — they are not earnings press releases
            if not exhibit_filename:
                continue

            text = _download_exhibit(cik_int, acc_nodash, exhibit_filename)

            # Require a minimum of 200 words — filters out stub/cover-page filings
            if len(text.split()) < 200:
                continue

            quarter_label = "Most Recent" if result_index == 0 else f"Q-{result_index} Prior"
            results.append({
                "title"  : f"{ticker} Earnings Release · {date}",
                "text"   : text,
                "date"   : date,
                "quarter": quarter_label,
                "acc"    : acc_dashes,
            })
            result_index += 1
        except Exception:
            continue   # skip filings we can't parse

    if not results:
        raise RuntimeError(
            f"Could not find any earnings press releases (8-K/EX-99.1) for '{ticker}' on SEC EDGAR."
        )
    return results


def fetch_transcript(ticker: str) -> dict:
    """Fetch the single most-recent earnings release. Wrapper for single-ticker flow."""
    filings = fetch_last_n_filings(ticker, n=1)
    return filings[0]


# ════════════════════════════════════════════════════════════════════════════
# Earnings Calendar
# ════════════════════════════════════════════════════════════════════════════

def _fmt_large_val(n) -> str:
    if not n:
        return "N/A"
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def fetch_earnings_calendar(tickers: list) -> list:
    """
    Fetch upcoming earnings dates for a list of tickers via yfinance.
    Returns list of dicts sorted by days_until, soonest first.
    """
    today = datetime.date.today()
    results = []

    for ticker in tickers:
        entry = {
            "ticker":        ticker.upper(),
            "company_name":  ticker.upper(),
            "earnings_date": "N/A",
            "days_until":    None,
            "eps_estimate":  "N/A",
            "rev_estimate":  "N/A",
        }
        try:
            tk   = yf.Ticker(ticker)
            info = tk.info
            entry["company_name"] = info.get("longName", ticker)

            cal = tk.calendar
            if isinstance(cal, dict):
                raw_dates = cal.get("Earnings Date", [])
                chosen = None
                # Prefer the first date that is >= today
                for d in raw_dates:
                    if hasattr(d, "date"):
                        d = d.date()
                    elif isinstance(d, str):
                        try:
                            d = datetime.date.fromisoformat(d[:10])
                        except Exception:
                            continue
                    if isinstance(d, datetime.date):
                        if d >= today:
                            chosen = d
                            break
                        elif chosen is None:
                            chosen = d   # keep most-recent past date as fallback
                if chosen:
                    entry["earnings_date"] = str(chosen)
                    entry["days_until"]    = (chosen - today).days

                eps = cal.get("Earnings Average") or cal.get("EPS Estimate")
                rev = cal.get("Revenue Average")  or cal.get("Revenue Estimate")
                entry["eps_estimate"] = f"${eps:.2f}" if eps else "N/A"
                entry["rev_estimate"] = _fmt_large_val(rev) if rev else "N/A"
        except Exception:
            pass

        results.append(entry)

    results.sort(key=lambda x: (
        x["days_until"] is None,
        abs(x["days_until"]) if x["days_until"] is not None else 9999,
    ))
    return results


# ════════════════════════════════════════════════════════════════════════════
# Recent News
# ════════════════════════════════════════════════════════════════════════════

def fetch_recent_news(ticker: str, n: int = 8) -> list:
    """
    Fetch recent news for a ticker via yfinance.
    Handles both legacy (pre-0.2.50) and new nested-content formats.
    Returns list of {title, publisher, link, published_date}.
    """
    tk = yf.Ticker(ticker)
    try:
        raw = tk.news or []
    except Exception:
        return []

    results = []
    for item in raw[:n]:
        try:
            # New yfinance format (0.2.50+): nested under "content"
            if "content" in item:
                c         = item["content"]
                title     = c.get("title", "")
                publisher = (c.get("provider") or {}).get("displayName", "")
                link      = (
                    (c.get("clickThroughUrl") or c.get("canonicalUrl") or {}).get("url", "")
                )
                pub_str   = c.get("pubDate", "")
                pub_date  = pub_str[:10] if pub_str else "N/A"
            else:
                # Legacy format
                pub_time  = item.get("providerPublishTime", 0)
                pub_date  = (
                    datetime.datetime.utcfromtimestamp(pub_time).strftime("%Y-%m-%d")
                    if pub_time else "N/A"
                )
                title     = item.get("title", "")
                publisher = item.get("publisher", "")
                link      = item.get("link", "")

            if title:
                results.append({
                    "title":          title,
                    "publisher":      publisher,
                    "link":           link,
                    "published_date": pub_date,
                })
        except Exception:
            continue
    return results


# ════════════════════════════════════════════════════════════════════════════
# New Filing Detection  (auto-trigger support)
# ════════════════════════════════════════════════════════════════════════════

def check_new_filing(ticker: str, last_acc: str = None) -> dict:
    """
    Check SEC EDGAR for the latest earnings 8-K (Item 2.02) for *ticker*.
    Returns {is_new, acc, date, had_previous}.
    is_new is True only when last_acc is given AND differs from the newest earnings acc.

    Only considers 8-Ks with Item 2.02 (Results of Operations) to avoid triggering
    on non-earnings filings (director changes, agreements, etc.).
    Falls back to any 8-K if no Item 2.02 is found.
    """
    try:
        cik = _get_cik(ticker.upper())
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(sub_url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        recent = resp.json()["filings"]["recent"]

        items_list = recent.get("items", [""] * len(recent["form"]))
        first_8k = None  # fallback if no Item 2.02 found

        for form, date, acc, items in zip(
            recent["form"], recent["filingDate"], recent["accessionNumber"], items_list
        ):
            if form != "8-K":
                continue
            if first_8k is None:
                first_8k = {"acc": acc, "date": date}
            item_nums = [i.strip() for i in str(items).split(",")]
            if "2.02" in item_nums:
                # Found the most recent earnings release 8-K
                return {
                    "is_new":        last_acc is not None and acc != last_acc,
                    "acc":           acc,
                    "date":          date,
                    "had_previous":  last_acc is not None,
                }

        # Fallback: use first 8-K found if no Item 2.02 available
        if first_8k:
            return {
                "is_new":        last_acc is not None and first_8k["acc"] != last_acc,
                "acc":           first_8k["acc"],
                "date":          first_8k["date"],
                "had_previous":  last_acc is not None,
            }
        return {"is_new": False, "acc": None, "date": None, "had_previous": False}
    except Exception:
        return {"is_new": False, "acc": None, "date": None, "had_previous": False}


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
    compare_data: dict = None,
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

    # ── Multi-Company Comparison ───────────────────────────────────────────────
    if compare_data:
        # Build full set: primary ticker + all compared tickers
        all_companies = {ticker: {"stock": stock_data, "sentiment": current_sentiment}}
        all_companies.update(compare_data)

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(13, 17, 23)
        pdf.cell(0, 7, "MULTI-COMPANY COMPARISON",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(88, 166, 255)
        pdf.set_line_width(0.6)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Comparison summary table
        col_w   = [30, 50, 28, 34, 22, 22, 22]
        headers = ["Ticker", "Company", "Price", "Vibe Score", "Pos %", "Neg %", "Neu %"]

        pdf.set_fill_color(22, 27, 34)
        pdf.set_text_color(230, 237, 243)
        pdf.set_font("Helvetica", "B", 8)
        for h, w in zip(headers, col_w):
            pdf.cell(w, 7, h, fill=True, border=1,
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(7)

        pdf.set_font("Helvetica", "", 8)
        for i, (cticker, cdata) in enumerate(all_companies.items()):
            cstock = cdata.get("stock", {})
            csent  = cdata.get("sentiment", {})
            fill   = i % 2 == 0
            pdf.set_fill_color(240, 242, 245) if fill else pdf.set_fill_color(255, 255, 255)
            vs     = csent.get("vibe_score", 0)
            r2, g2, b2 = _score_to_rgb(vs)
            cprice = cstock.get("price", 0)
            chg    = cstock.get("change_pct", 0)
            cname  = cstock.get("company_name", cticker)[:22]  # truncate for cell width
            vals = [
                cticker,
                cname,
                f"${cprice:,.2f} ({chg:+.1f}%)",
                f"{vs:+.3f}",
                f"{csent.get('positive_pct', 0)}%",
                f"{csent.get('negative_pct', 0)}%",
                f"{csent.get('neutral_pct', 0)}%",
            ]
            for j, (val, w) in enumerate(zip(vals, col_w)):
                pdf.set_text_color(r2, g2, b2) if j == 3 else pdf.set_text_color(13, 17, 23)
                pdf.cell(w, 6, val, fill=fill, border=1,
                         new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(6)

        pdf.ln(6)

        # Per-company detail cards
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(13, 17, 23)
        pdf.cell(0, 7, "INDIVIDUAL COMPANY SUMMARIES",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(88, 166, 255)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        for cticker, cdata in all_companies.items():
            cstock = cdata.get("stock", {})
            csent  = cdata.get("sentiment", {})
            vs     = csent.get("vibe_score", 0)
            r2, g2, b2 = _score_to_rgb(vs)

            # Company sub-header
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(13, 17, 23)
            pdf.cell(0, 7,
                     f"{cstock.get('company_name', cticker)} ({cticker})  "
                     f"·  {cstock.get('sector', 'N/A')}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Vibe score banner
            pdf.set_fill_color(r2, g2, b2)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 9,
                     f"  Vibe Score: {vs:+.2f}  |  {csent.get('tone', 'N/A').title()}",
                     fill=True, align="L",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

            # Stock + sentiment detail line
            cprice  = cstock.get("price", 0)
            chg_pct = cstock.get("change_pct", 0)
            arrow   = "+" if chg_pct >= 0 else ""
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 5,
                     f"Price: ${cprice:,.2f} ({arrow}{chg_pct:.2f}%)   "
                     f"Mkt Cap: {cstock.get('market_cap', 'N/A')}   "
                     f"P/E: {cstock.get('pe_ratio', 'N/A')}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 5,
                     f"Positive: {csent.get('positive_pct', 0)}%   "
                     f"Negative: {csent.get('negative_pct', 0)}%   "
                     f"Neutral: {csent.get('neutral_pct', 0)}%   "
                     f"Segments analysed: {csent.get('chunk_count', 0)}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Summary (stripped markdown)
            summary_c = re.sub(r"\*\*(.+?)\*\*", r"\1", csent.get("summary", ""))
            summary_c = re.sub(r"[🟢🔴⚪]", "", summary_c).strip()
            summary_c = summary_c.encode("latin-1", errors="replace").decode("latin-1")
            if summary_c:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(0, 4, summary_c[:400])  # cap length

            pdf.ln(6)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(0, 5,
             "Generated by Pulse AI  |  ProsusAI/FinBERT & SEC EDGAR  |  "
             "For educational purposes only. Not financial advice.",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


# ════════════════════════════════════════════════════════════════════════════
# Forward-Looking Outlook Generator
# ════════════════════════════════════════════════════════════════════════════

def _run_news_sentiment(headlines: list) -> dict:
    """Run FinBERT on up to 8 news headlines; return aggregate sentiment dict."""
    if not headlines or not HF_TOKEN:
        return {"score": 0.0, "positive": 0, "negative": 0, "neutral": 0, "total": 0}

    hf_headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    label_counts  = {"positive": 0, "negative": 0, "neutral": 0}
    weighted_score = 0.0
    processed      = 0

    for headline in headlines[:8]:
        if not headline.strip():
            continue
        try:
            payload = {"inputs": headline, "parameters": {"truncation": True}}
            resp    = requests.post(HF_API_URL, headers=hf_headers, json=payload, timeout=30)
            if resp.status_code != 200:
                continue
            raw     = resp.json()
            results = raw[0] if isinstance(raw, list) and isinstance(raw[0], list) else raw
            top     = max(results, key=lambda r: r["score"])
            label   = top["label"].lower()
            conf    = top["score"]
            label_counts[label] = label_counts.get(label, 0) + 1
            weighted_score     += _label_to_score(label, conf)
            processed          += 1
        except Exception:
            continue

    if processed == 0:
        return {"score": 0.0, "positive": 0, "negative": 0, "neutral": 0, "total": 0}

    return {
        "score":    round(weighted_score / processed, 4),
        "positive": label_counts["positive"],
        "negative": label_counts["negative"],
        "neutral":  label_counts["neutral"],
        "total":    processed,
    }


def generate_outlook(
    ticker: str,
    sentiment: dict,
    history: list,
    news_items: list,
) -> dict:
    """
    Generate a forward-looking outlook by combining:
      - Current FinBERT earnings-call sentiment (weight 40 %)
      - Quarter-over-quarter sentiment trend     (weight 30 %)
      - Recent news headline sentiment           (weight 30 %)

    Returns:
        signal          : "bullish" | "neutral" | "bearish"
        confidence      : float 0-1
        composite_score : float -1 to +1
        summary         : narrative string (markdown)
        signals         : list of contributing-signal dicts
        news_sentiment  : raw news sentiment dict
    """
    score = sentiment["vibe_score"]
    tone  = sentiment["tone"]

    # ── Signal 1 : Earnings call FinBERT score ──────────────────────────────
    s1     = score
    s1_dir = "bullish" if score >= 0.25 else ("bearish" if score < -0.1 else "neutral")

    # ── Signal 2 : Q/Q trend ────────────────────────────────────────────────
    trend        = 0.0
    trend_label  = "Insufficient data"
    trend_clipped = 0.0
    if len(history) >= 2:
        trend         = history[0]["vibe_score"] - history[1]["vibe_score"]
        trend_clipped = min(max(trend * 2, -1), 1)
        if trend > 0.1:
            trend_label = f"Improving (+{trend:.2f} vs prior quarter)"
        elif trend < -0.1:
            trend_label = f"Declining ({trend:+.2f} vs prior quarter)"
        else:
            trend_label = f"Stable ({trend:+.2f} vs prior quarter)"
    s2     = trend_clipped
    s2_dir = "bullish" if s2 > 0.1 else ("bearish" if s2 < -0.1 else "neutral")

    # ── Signal 3 : News sentiment ────────────────────────────────────────────
    headlines  = [n["title"] for n in news_items if n.get("title")]
    news_sent  = _run_news_sentiment(headlines)
    s3         = news_sent["score"]
    s3_dir     = "bullish" if s3 >= 0.1 else ("bearish" if s3 < -0.05 else "neutral")

    # ── Composite ────────────────────────────────────────────────────────────
    composite = s1 * 0.40 + s2 * 0.30 + s3 * 0.30

    if composite >= 0.20:
        overall    = "bullish"
        confidence = min(0.50 + composite * 0.50, 0.95)
    elif composite <= -0.10:
        overall    = "bearish"
        confidence = min(0.50 + abs(composite) * 0.50, 0.95)
    else:
        overall    = "neutral"
        confidence = 0.50

    # ── Narrative ────────────────────────────────────────────────────────────
    tone_map = {
        "strongly optimistic": "strong optimism",
        "moderately positive": "moderate confidence",
        "broadly neutral":     "a measured, balanced stance",
        "cautiously negative": "cautious concern",
        "distinctly bearish":  "clearly bearish language",
    }
    tone_phrase = tone_map.get(tone, tone)

    news_phrase = (
        f"{news_sent['positive']} of {news_sent['total']} recent headlines carry a positive tone"
        if news_sent["total"] > 0
        else "limited news data was available"
    )

    trend_phrase = "The quarter-over-quarter sentiment trend"
    if len(history) >= 2:
        if trend > 0.1:
            trend_phrase += f" is improving, up {trend:+.2f} points Q/Q."
        elif trend < -0.1:
            trend_phrase += f" is weakening, down {trend:+.2f} points Q/Q."
        else:
            trend_phrase += " remains broadly flat Q/Q."
    else:
        trend_phrase += " cannot yet be assessed (insufficient history)."

    if overall == "bullish":
        narrative = (
            f"Combined signals point to a **bullish near-term outlook** for {ticker}. "
            f"Management expressed {tone_phrase} in the latest earnings call, "
            f"and {news_phrase}. {trend_phrase} "
            f"Watch for continuation of positive guidance or upward estimate revisions as confirmation."
        )
    elif overall == "bearish":
        narrative = (
            f"Combined signals suggest a **cautious-to-bearish near-term outlook** for {ticker}. "
            f"Management conveyed {tone_phrase} in the latest earnings release, "
            f"and {news_phrase}. {trend_phrase} "
            f"Monitor headwinds flagged in the filing and any management guidance updates before positioning."
        )
    else:
        narrative = (
            f"Combined signals indicate a **mixed or neutral near-term outlook** for {ticker}. "
            f"Management adopted {tone_phrase} in the latest earnings call, "
            f"and {news_phrase}. {trend_phrase} "
            f"Clarity may emerge from upcoming macro releases or the next quarterly filing."
        )

    return {
        "signal":          overall,
        "confidence":      round(confidence, 3),
        "composite_score": round(composite, 4),
        "summary":         narrative,
        "signals": [
            {
                "name":      "Earnings Call Tone",
                "value":     f"{score:+.2f}",
                "direction": s1_dir,
                "weight":    "40%",
                "detail":    tone.title(),
            },
            {
                "name":      "Q/Q Sentiment Trend",
                "value":     trend_label,
                "direction": s2_dir,
                "weight":    "30%",
                "detail":    f"Based on {len(history)} quarters",
            },
            {
                "name":      "Recent News Sentiment",
                "value":     f"{s3:+.2f}",
                "direction": s3_dir,
                "weight":    "30%",
                "detail":    f"{news_sent['total']} articles analysed",
            },
        ],
        "news_sentiment": news_sent,
    }


# ════════════════════════════════════════════════════════════════════════════
# Ask Pulse — Gemini 1.5 Flash LLM Chatbot
# ════════════════════════════════════════════════════════════════════════════

PULSE_SYSTEM_PROMPT = """You are Pulse, a concise equity research assistant.
You answer investor questions ONLY using the structured context provided.
Rules:
- Never speculate beyond what the context contains.
- Always cite specific quarters (e.g. Q3 2024) and paragraph tags (e.g. [P4]) when referencing data.
- Keep answers brief and factual — suitable for a professional investor audience.
- Output ONLY valid JSON. No markdown fences, no prose outside the JSON.
Output schema:
{
  "short_summary": "<1-3 sentence direct answer>",
  "drivers": ["<key driver 1>", "<key driver 2>", "<key driver 3>"],
  "data_references": ["<e.g. Q3 2024 sentiment: +0.42>", "<e.g. [P7] Revenue grew 12% YoY>"]
}"""


def build_pulse_user_prompt(
    ticker: str,
    stock: dict,
    sentiment: dict,
    history: list,
    outlook: Optional[dict],
    filing_text: str,
    user_question: str,
) -> str:
    """Build the structured context prompt sent to Gemini."""

    # ── Sentiment history block ──────────────────────────────────────────────
    hist_lines = []
    for q in history:
        hist_lines.append(
            f"  {q.get('quarter','?')}: score={q.get('score', 0):+.2f}, "
            f"label={q.get('label','?')}, positive={q.get('positive',0):.0%}, "
            f"negative={q.get('negative',0):.0%}, neutral={q.get('neutral',0):.0%}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "  (no history available)"

    # ── Most recent sentiment ────────────────────────────────────────────────
    score    = sentiment.get("score", 0)
    label    = sentiment.get("label", "unknown")
    pos      = sentiment.get("positive", 0)
    neg      = sentiment.get("negative", 0)
    neu      = sentiment.get("neutral", 0)

    # ── Stock data block ─────────────────────────────────────────────────────
    price    = stock.get("price", "N/A")
    chg_pct  = stock.get("change_pct", 0) or 0
    mktcap   = stock.get("market_cap", "N/A")
    pe       = stock.get("pe_ratio", "N/A")
    hi52     = stock.get("week_52_high", "N/A")
    lo52     = stock.get("week_52_low", "N/A")

    # ── Outlook block ────────────────────────────────────────────────────────
    if outlook:
        outlook_lines = []
        for sig in outlook.get("signals", []):
            outlook_lines.append(
                f"  {sig['name']}: {sig['value']} ({sig['direction']}, weight {sig['weight']})"
            )
        outlook_block = "\n".join(outlook_lines) if outlook_lines else "  (not available)"
        pulse_score   = outlook.get("pulse_score", "N/A")
        pulse_label   = outlook.get("pulse_label", "N/A")
    else:
        outlook_block = "  (not available)"
        pulse_score   = "N/A"
        pulse_label   = "N/A"

    # ── Tag filing paragraphs [P1]–[P10], cap each at 300 chars to save tokens ─
    paragraphs = [p.strip() for p in filing_text.split("\n\n") if len(p.strip()) > 30][:10]
    tagged_paras = "\n\n".join(
        f"[P{i+1}] {p[:300]}{'…' if len(p) > 300 else ''}"
        for i, p in enumerate(paragraphs)
    )

    return f"""=== PULSE AI CONTEXT FOR {ticker.upper()} ===

--- LATEST EARNINGS SENTIMENT ---
Score : {score:+.2f}  |  Label: {label}
Positive: {pos:.0%}  |  Negative: {neg:.0%}  |  Neutral: {neu:.0%}

--- 4-QUARTER SENTIMENT HISTORY ---
{history_block}

--- MARKET DATA ---
Price     : {price}  ({chg_pct:+.2f}% today)
Market Cap: {mktcap}
P/E Ratio : {pe}
52W High  : {hi52}
52W Low   : {lo52}

--- PULSE OUTLOOK ---
Pulse Score : {pulse_score}  |  Label: {pulse_label}
Signals:
{outlook_block}

--- LATEST EARNINGS FILING EXCERPTS (tagged) ---
{tagged_paras}

=== INVESTOR QUESTION ===
{user_question}"""


def call_pulse_llm(user_prompt: str) -> dict:
    """
    Send a prompt to Gemini 2.0 Flash and return the parsed JSON response.

    Returns a dict with keys: short_summary, drivers, data_references.
    Raises ValueError if the API key is missing or the response cannot be parsed.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    from google.genai import types as genai_types
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=genai_types.GenerateContentConfig(
            system_instruction=PULSE_SYSTEM_PROMPT,
        ),
        contents=user_prompt,
    )
    raw = response.text.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned non-JSON output: {raw[:300]}") from exc
