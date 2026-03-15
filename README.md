# 📡 Pulse AI — Earnings Intelligence Dashboard

> **ESADE MIBA · Prototyping Products with AI · Assignment 2**
> Built by [Jerico Agdan](https://github.com/JericoNeil)

Pulse AI is an AI-powered financial intelligence dashboard that turns raw SEC earnings filings into actionable insight — sentiment scores, trend analysis, multi-company comparison, PDF reports, and a conversational LLM analyst — all in one place.

---

## 🆕 What's New vs Assignment 1

Assignment 1 delivered a working 5-tab sentiment dashboard. Assignment 2 significantly expands the product across three dimensions:

| Dimension | Assignment 1 | Assignment 2 |
|-----------|-------------|-------------|
| **Tabs** | 5 | 7 (+What's Next, +Ask Pulse) |
| **Filings** | Latest only | Latest + 4-quarter history |
| **Comparison** | Side-by-side vibe scores | Full PDF export with multi-company table |
| **Automation** | Manual only | Auto-trigger on new SEC filings |
| **LLM** | None | Ask Pulse — multi-ticker Gemini chatbot |
| **News** | None | Live Yahoo Finance headlines in chatbot context |
| **PDF** | Single-ticker | Multi-company comparison report |
| **Deployment** | Local only | Streamlit Cloud |

---

## ✨ Features

### 7-Tab Dashboard

| Tab | What it does |
|-----|-------------|
| 📊 **Market Overview** | Live stock price, KPI metrics (market cap, P/E, 52W range), 3-month candlestick chart |
| 💬 **Earnings Sentiment** | FinBERT Vibe Score, sentiment donut, keyword treemap, positive/negative driver chart |
| 📈 **Sentiment History** | 4-quarter trend with Q-label charts and per-quarter insight cards |
| 🔄 **Ticker Compare** | Load up to 3 tickers side-by-side — radar chart, vibe comparison, mini cards |
| 📄 **Raw Filing** | Full EX-99.1 press release text fetched live from SEC EDGAR |
| 🔮 **What's Next** | Pulse Score outlook — forward-looking signal engine based on sentiment + market data |
| 🤖 **Ask Pulse** | LLM chatbot — ask anything about loaded tickers (see below) |

### 📅 Earnings Calendar
- Sidebar widget shows upcoming earnings dates for up to 5 tickers at once
- Flags same-day and next-day earnings with colour-coded urgency badges

### ⚡ Auto-Trigger
- Background polling monitors SEC EDGAR for new 8-K filings
- Automatically runs the full analysis pipeline when a new earnings release is detected
- Configurable check interval in the sidebar

### 📄 PDF Report
- One-click export of a branded PDF report
- Single-ticker mode: market snapshot, sentiment scores, 4-quarter history, keyword insights, Pulse Outlook
- Multi-company mode: comparison table + individual company summaries for all loaded tickers

---

## 🤖 Ask Pulse — The LLM Feature

Ask Pulse is the most technically non-trivial feature in the project. It goes beyond a simple "send question to GPT" integration in several ways:

### How it works

```
User question
      │
      ▼
┌─────────────────────────────────────┐
│       Context Assembly Layer        │
│  • Primary ticker sentiment scores  │
│  • 4-quarter sentiment history      │
│  • Market data (price, P/E, mktcap) │
│  • Pulse Outlook signals            │
│  • Tagged filing excerpts [P1–P10]  │
│  • Compared tickers (if loaded)     │  ← multi-ticker
│  • Live Yahoo Finance news          │  ← real-time
└─────────────────────────────────────┘
      │
      ▼
 Gemini 2.5 Flash (structured JSON output)
      │
      ▼
┌─────────────────────────┐
│  Structured response:   │
│  • short_summary        │
│  • drivers []           │
│  • data_references []   │
└─────────────────────────┘
```

### Why it's non-trivial

1. **Structured grounding, not hallucination** — the LLM is given a richly structured context block (tagged filing paragraphs, numeric sentiment scores, historical trend data) and instructed to answer *only* from that context. It cites `[P3]`, `Q2 2025 sentiment: +0.42`, etc.

2. **Multi-ticker context** — when you've loaded comparison tickers (e.g. JPM + GOOGL + MSFT), all their sentiment scores and market data are injected into the prompt. The chatbot can answer cross-company questions like *"Which of these three has the most negative earnings tone?"*

3. **Live news integration** — before every Gemini call, the app fetches the 5 most recent Yahoo Finance headlines for each loaded ticker via `yfinance`. These are injected as a `RECENT NEWS HEADLINES` block so the LLM can factor in breaking news when answering.

4. **Fragment-isolated UI** — `st.chat_input` in Streamlit normally pins outside all tab containers, causing any interaction to reset the active tab to Tab 1. Ask Pulse wraps the entire chat UI in a `@st.fragment`, so button clicks and chat submissions only trigger a fragment-scoped rerun — the tab never jumps.

5. **Structured JSON output** — the system prompt instructs Gemini to return strict JSON (`short_summary`, `drivers`, `data_references`). The app parses and renders each field separately with distinct visual styling — not just a plain text blob.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | [Streamlit](https://streamlit.io) 1.50 |
| Market data | [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance) |
| Filings | [SEC EDGAR](https://www.sec.gov/edgar/) REST API |
| Sentiment AI | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) via HuggingFace Inference API |
| LLM | [Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) via Google GenAI SDK |
| Charts | [Plotly](https://plotly.com/python/) |
| PDF reports | [fpdf2](https://py-pdf.github.io/fpdf2/) |

---

## 🚀 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/JericoNeil/Pulse_AI.git
cd Pulse_AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```
HF_TOKEN=your_huggingface_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

- **HF_TOKEN** — free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **GEMINI_API_KEY** — free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 4. Run
```bash
streamlit run main.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Streamlit Cloud Deployment

1. Fork or push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `main.py`
4. Under **Advanced settings → Secrets**, add:
```toml
HF_TOKEN = "your_huggingface_token"
GEMINI_API_KEY = "your_gemini_api_key"
```
5. Click **Deploy**

---

## 📁 Project Structure

```
Pulse_AI/
├── main.py              # Streamlit UI — 7-tab dashboard + fragment chat
├── logic.py             # Full data pipeline:
│                        #   SEC EDGAR fetcher, FinBERT sentiment,
│                        #   yfinance market data, Gemini LLM,
│                        #   PDF generator, Pulse Outlook engine
├── requirements.txt     # Pinned dependencies
├── .streamlit/
│   └── config.toml      # Dark theme + server config
├── .env                 # Local secrets (gitignored)
└── README.md
```

---

## 📊 Usage

1. Enter a US ticker (e.g. `AAPL`, `JPM`, `TSLA`) in the sidebar
2. Click **⚡ Run Analysis** — fetches filing, runs FinBERT, computes outlook
3. Explore the 7 tabs
4. Optionally add tickers to **Compare Tickers** and click **🔄 Run Comparison**
5. Download the PDF report from the sidebar
6. Go to **🤖 Ask Pulse** and ask questions about the loaded companies

---

> *For educational purposes only. Not financial advice.*
