---
license: mit
title: 🪙 CryptoLens
sdk: gradio
emoji: 🦀
colorFrom: indigo
colorTo: blue
pinned: false
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/677150afdd9607b6253d0d09/j-k-lxW44Enta3kt9RG_a.jpeg
short_description: Crypto Sentiment & On-Chain Analyzer
sdk_version: 5.37.0
---

# 🪙 CryptoLens: Crypto Sentiment & On-Chain Analyzer

**Version:** 0.2.0  
**Author:** Dr. Partha Majumdar

CryptoLens is a Gradio-powered dashboard for performing **real-time sentiment analysis** on news and tweets, and **auto-generating on-chain query logic** using Dune SQL. It integrates sentiment models (VADER, TextBlob, FinBERT) and Google's Gemini Pro model to produce explainable results for analysts.

## 🔍 Features

- 📰 Fetch crypto headlines from CoinDesk RSS
- 🔦 Scrape tweets from Twitter/X using Tweepy
- 🧠 Analyze sentiment with:
  - VADER (Lexicon-based)
  - TextBlob (Polarity-based)
  - FinBERT (Transformer model)
- 🎓 Get Gemini-generated sentiment explanations
- ➕ Gemini also writes Dune-compatible SQL queries based on headlines
- 🔢 Sample report with charts from Dune output (CSV)
  - Daily transfer trends
  - Totals per chain
  - Distribution analysis
  - Moving averages

## 🧪 Use Cases

- Track sentiment and on-chain response to crypto news
- Compare chains like Arbitrum, Avalanche, Gnosis
- Empower DeFi, NFT, or DAO analysts with explainable insights

## 👁️ Sample Output
VADER: Compound Score: 0.77 (VADER)
TextBlob: Polarity Score: 0.50 (TextBlob)
FinBERT: Positive (0.92)
Gemini: The sentiment toward Solana is strongly bullish, driven by both developer growth and increasing adoption metrics.
Dune SQL: SELECT …

## 📦 Dependencies

- `gradio`
- `tweepy`
- `vaderSentiment`
- `textblob`
- `transformers`
- `google-generativeai`
- `matplotlib`, `seaborn`, `pandas`, `bs4`

## ⚙️ Setup Instructions

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2.	**Download required NLP assets**:
```bash
python -m textblob.download_corpora
python -m nltk.downloader vader_lexicon
```

3.	**Set environment variables**:
```bash
export GEMINI_API_KEY=your_gemini_key
export TWITTER_BEARER_TOKEN=your_twitter_key
```

4.	**Run the app locally**:
```bash
python app.py
```

🌐 Deployment

Optimised for Hugging Face Spaces. App uses Gradio Tabs for modular usage:
	•	"Scrap from CoinDesk"
	•	"Scrap from X"
	•	"Sample Dune Query Analysis"

⸻

📄 License

MIT License © Dr. Partha Majumdar