# 📡 Pulse AI — Earnings Call Sentiment Monitor

> **ESADE MIBA · Prototyping Products with AI Assignment**

An AI-powered financial intelligence dashboard that analyses earnings press releases using FinBERT sentiment analysis, live stock data, and keyword insights.

## Features

| Tab | Description |
|-----|-------------|
| 📊 Market Overview | Live stock price, KPI metrics, 3-month price chart + candlestick |
| 💬 Earnings Sentiment | Sentiment Score, donut chart, keyword treemap & pos/neg driver chart |
| 📈 Sentiment History | 4-quarter trend with Q-label charts and per-quarter insight cards |
| 🔄 Ticker Compare | Side-by-side vibe, radar chart, mini ticker cards |
| 📄 Raw Filing | Full EX-99.1 press release text from SEC EDGAR |

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)
- **Market data**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance)
- **Filings**: [SEC EDGAR](https://www.sec.gov/edgar/) API
- **AI model**: [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) via Hugging Face Inference API
- **Charts**: [Plotly](https://plotly.com/python/)
- **PDF reports**: [fpdf2](https://py-pdf.github.io/fpdf2/)

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```
HF_TOKEN=your_hugging_face_api_token_here
```
Get your free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Run
```bash
streamlit run main.py
```

Then open **http://localhost:8501** in your browser.

## Usage

1. Enter a US stock ticker (e.g. `AAPL`, `MSFT`, `TSLA`) in the sidebar
2. Click **⚡ Run Analysis**
3. Explore all 5 tabs — sentiment score, keyword drivers, quarterly trend, and more
4. Download the PDF report from the sidebar
5. Optionally enter tickers in **Compare Tickers** and click **🔄 Run Comparison**

## Project Structure

```
├── main.py          # Streamlit UI — 5-tab dashboard
├── logic.py         # Data pipeline: SEC EDGAR, yfinance, FinBERT, PDF
├── requirements.txt
├── .env             
└── README.md
```
