import os
import gradio as gr
import google.generativeai as genai
import requests
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tweepy
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from bs4 import BeautifulSoup

# Constants
MODEL_NOT_AVAILABLE = "Model not available."
MODEL_VADER = "VADER"
MODEL_TEXTBLOB = "TextBlob"
MODEL_FINBERT = "FinBERT"
GEMINI_MODEL = "gemini-1.5-pro"
COINDESK_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"

# Get the Environment Variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Authenticate X using Bearer Token
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def scrape_twitter(query="#bitcoin", max_results=10):
    try:
        response = client.search_recent_tweets(
            query=query,
            tweet_fields=["created_at", "text", "author_id"],
            max_results=max_results
        )
        
        # Deduplicate and return text only
        seen = set()
        tweets = []
        if response.data:
            for tweet in response.data:
                text = tweet.text.strip()
                if text not in seen:
                    seen.add(text)
                    tweets.append(text)
        else:
            tweets = ["No tweets found"]
            
        return tweets

    except tweepy.TooManyRequests:
        print("Rate limit hit. Waiting for 15 minutes...")
        time.sleep(15 * 60)
        return scrape_twitter(query, max_results)

    except Exception as e:
        return [f"Error: {str(e)}"]

def generate_dune_query_from_headline(headline: str) -> str:
    prompt = (
        f"Crypto Headline: {headline}\n\n"
        "Write a Dune-compatible SQL query that investigates relevant on-chain activity "
        "related to the news item above. Focus on usage metrics like transfers, active wallets, "
        "TVL, or volume. Use well-known public tables like those in Ethereum, Arbitrum, or Avalanche datasets. "
        "Include appropriate time filtering (e.g., past 7 days).\n\n"
        "Only return the SQL query. Do not include any commentary or notes."
    )
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()
    
def scrape_coindesk_headlines():
    url = COINDESK_URL
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, "xml")

    headlines = [item.title.text for item in soup.find_all("item")][:10]
    return headlines if headlines else ["No headlines found"]
    
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_and_onchain(text):
    sentiment_result_vader, explanation_vader = analyze_sentiment(text, MODEL_VADER)
    sentiment_result_textblob, explanation_textblob = analyze_sentiment(text, MODEL_TEXTBLOB)
    sentiment_result_finbert, explanation_finbert = analyze_sentiment(text, MODEL_FINBERT)
    query_str = generate_dune_query_from_headline(text)
    return sentiment_result_vader, explanation_vader, sentiment_result_textblob, explanation_textblob, sentiment_result_finbert, explanation_finbert, query_str
    
def analyze_sentiment(text, model_choice):
    sentiment_result = compute_sentiment(text, model_choice)
    if sentiment_result == MODEL_NOT_AVAILABLE:
        explanation = ""
    else:
        explanation = explain_sentiment_with_gemini(text, model_choice, sentiment_result)
    return sentiment_result, explanation
    
def compute_sentiment(text, model):
    if model == MODEL_VADER:
        score = vader_analyzer.polarity_scores(text)
        return f"Compound Score: {score['compound']:.3f} (VADER)"
    elif model == MODEL_TEXTBLOB:
        polarity = TextBlob(text).sentiment.polarity
        return f"Polarity Score: {polarity:.3f} (TextBlob)"
    elif model == MODEL_FINBERT:
        from transformers import pipeline
        classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        result = classifier(text)
        return result[0]["label"] + f" ({result[0]['score']:.2f})"
    else:
        return MODEL_NOT_AVAILABLE

def explain_sentiment_with_gemini(text, model_name, sentiment_result):
    prompt = (
        f"Text: {text}\n"
        f"Model: {model_name}\n"
        f"Sentiment Result: {sentiment_result}\n\n"
        "Based on the above result, provide a brief interpretation of the sentiment "
        "as if explaining to a business analyst tracking cryptocurrency trends. "
        "Be objective and focused."
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

##### FOR SAMPLE REPORT #####
# Define your SQL query as a multi-line string
dune_query = """
/*
Table(s):  bitcoinbr_arbitrum.bitcoinbr_evt_transfer, bitcoinbr_avalanche_c.bitcoinbr_evt_transfer,  bitcoinbr_gnosis.bitcoinbr_evt_transfer,  bitcoinbr_multichain.bitcoinbr_evt_transfer
Reason: Based on contract addresses or possible project names
*/

WITH ArbitrumTransfers AS (
  SELECT
    'Arbitrum' AS chain,
    DATE_TRUNC('day', evt_block_time) AS day,
    SUM(value) AS total_value
  FROM bitcoinbr_arbitrum.bitcoinbr_evt_transfer
  GROUP BY
    1,
    2
), AvalancheTransfers AS (
  SELECT
    'Avalanche' AS chain,
    DATE_TRUNC('day', evt_block_time) AS day,
    SUM(value) AS total_value
  FROM bitcoinbr_avalanche_c.bitcoinbr_evt_transfer
  GROUP BY
    1,
    2
), GnosisTransfers AS (
  SELECT
    'Gnosis' AS chain,
    DATE_TRUNC('day', evt_block_time) AS day,
    SUM(value) AS total_value
  FROM bitcoinbr_gnosis.bitcoinbr_evt_transfer
  GROUP BY
    1,
    2
), MultichainTransfers AS (
  SELECT
    chain,
    DATE_TRUNC('day', evt_block_time) AS day,
    SUM(value) AS total_value
  FROM bitcoinbr_multichain.bitcoinbr_evt_transfer
  GROUP BY
    1,
    2
)
SELECT
  *
FROM ArbitrumTransfers
UNION ALL
SELECT
  *
FROM AvalancheTransfers
UNION ALL
SELECT
  *
FROM GnosisTransfers
UNION ALL
SELECT
  *
FROM MultichainTransfers
ORDER BY
  day,
  chain
"""

# Load and clean data
df = pd.read_csv("results.csv")
df["day"] = pd.to_datetime(df["day"])
df["chain"] = df["chain"].str.strip().str.lower()
df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
df = df[df["total_value"].notnull()]
df = df[df["total_value"] < df["total_value"].quantile(0.99)]

# Output folder
OUTPUT_FOLDER = "."  # "." for local dev

def save_plot(fig, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)

def plot1_img():
    fig = plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="day", y="total_value", hue="chain")
    plt.title("Daily Total Transfer Value per Chain")
    plt.xticks(rotation=45)
    save_plot(fig, "plot1.png")

def plot2_img():
    agg = df.groupby("chain")["total_value"].sum().sort_values(ascending=False)
    fig = plt.figure(figsize=(8, 5))
    sns.barplot(x=agg.index, y=agg.values)
    plt.title("Total Transfers by Chain")
    save_plot(fig, "plot2.png")

def plot3_img():
    g = sns.FacetGrid(df, col="chain", col_wrap=2, height=4, sharey=False)
    g.map_dataframe(sns.lineplot, x="day", y="total_value")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)
    save_plot(g.fig, "plot3.png")

def plot4_img():
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="chain", y="total_value")
    plt.title("Distribution of Transfer Values per Chain")
    save_plot(fig, "plot4.png")

def plot5_img():
    df_sorted = df.sort_values("day")

    # Remove 'chain' from the group before apply
    df_ma = (
        df_sorted.groupby("chain")
        .apply(lambda g: g[["day", "total_value"]]
               .set_index("day")
               .rolling("3D")
               .mean()
               .reset_index())
        .reset_index(drop=True)
    )

    # Reattach chain values (repeat per group length)
    chains = df_sorted.groupby("chain").size().repeat(
        df_sorted.groupby("chain").size()
    ).reset_index(drop=True)
    df_ma["chain"] = chains[:len(df_ma)]

    fig = plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_ma, x="day", y="total_value", hue="chain")
    plt.title("3-Day Moving Average of Transfers per Chain")
    plt.xticks(rotation=45)
    save_plot(fig, "plot5.png")
    
# Generate and save all plots
plot1_img()
plot2_img()
plot3_img()
plot4_img()
plot5_img()
#####        END        #####

# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

APP_TITLE = "📊 CryptoLens: Crypto Sentiment Analyzer"
APP_DESCRIPTION = (
    "Run sentiment analysis on tweets and Coinbase posts using "
    "offline-compatible tools: VADER, TextBlob, and FinBERT."
)

with gr.Blocks() as app:
    # Title and Description
    gr.HTML(
        f"""
        <p style='text-align: center; font-size: 40px; font-weight: bold;'>{APP_TITLE}</p>
        <p style='text-align: center; font-size: 20px; color: #555;'><sub>{APP_DESCRIPTION}</sub></p>
        <hr>
        """
    )

    with gr.Tabs():

        with gr.TabItem("Scrape From Coinbase"):

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        headline_button = gr.Button("🔄 Fetch from CoinDesk")

                    with gr.Row():
                        headline_dropdown = gr.Dropdown(
                            label="Select headline to analyse", 
                            choices=[], 
                            value=None,
                            interactive=True
                        )                    
    
                    # Buttons
                    with gr.Row():
                        submit_btn_coin = gr.Button("Submit")
                        clear_btn_coin = gr.Button("Clear")

            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    Model: VADER""")
                    sentiment_result_box_vader_coin = gr.Text(label="Sentiment Result")
                    interpretation_box_vader_coin = gr.Textbox(label="Gemini Interpretation", lines=3, max_lines=5, autoscroll=False)
                with gr.Column():
                    gr.HTML("""
                    Model: TEXTBLOB""")
                    sentiment_result_box_textblob_coin = gr.Text(label="Sentiment Result")
                    interpretation_box_textblob_coin = gr.Textbox(label="Gemini Interpretation", lines=3, max_lines=5, autoscroll=False)
                with gr.Column():
                    gr.HTML("""
                    Model: FinBERT""")
                    sentiment_result_box_finbert_coin = gr.Text(label="Sentiment Result")
                    interpretation_box_finbert_coin = gr.Textbox(label="Gemini Interpretation", lines=3, max_lines=5, autoscroll=False)

            # Text output for the DUNE QUERY
            dune_query_text_coin = gr.Textbox(
                label="DUNE Query is provided here. This query can be executed on www.DUNE.com using a FREE DUNE Account. To run the query and show the results in this App needs a monthly subscription of at least USD 339.", 
                visible=True, 
                lines=5, 
                max_lines=10,
                autoscroll=False,
                show_copy_button=True
            )

            def update_headline_choices():
                headlines = scrape_coindesk_headlines()
                if headlines:
                    return gr.update(choices=headlines, value=headlines[0])
                else:
                    return gr.update(choices=[], value=None)
            
            # Add this event binding
            headline_button.click(fn=update_headline_choices, outputs=headline_dropdown)

            # Function to clear inputs/outputs
            def clear_fields_coin():
                # Fetch fresh Headlines. Clear Sentiments and Interpretations for VADER, TEXTBLOB and FinBERT. Clear Query String.
                return update_headline_choices(), "", "", "", "", "", "", ""
        
            # Button bindings
            submit_btn_coin.click(
                fn=analyze_sentiment_and_onchain,
                inputs=[headline_dropdown],
                outputs=[
                    sentiment_result_box_vader_coin,
                    interpretation_box_vader_coin,
                    sentiment_result_box_textblob_coin,
                    interpretation_box_textblob_coin,
                    sentiment_result_box_finbert_coin,
                    interpretation_box_finbert_coin,
                    dune_query_text_coin
                    ]
            )
            clear_btn_coin.click(
                fn=clear_fields_coin, 
                outputs=[headline_dropdown,
                         sentiment_result_box_vader_coin, 
                         interpretation_box_vader_coin,
                         sentiment_result_box_textblob_coin, 
                         interpretation_box_textblob_coin,
                         sentiment_result_box_finbert_coin, 
                         interpretation_box_finbert_coin,
                         dune_query_text_coin
                        ])

        with gr.TabItem("Scrape from X"):
        
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        headline_button_x = gr.Button("🔄 Fetch from X")

                    with gr.Row():
                        headline_dropdown_x = gr.Dropdown(
                            label="Select headline to analyse", 
                            choices=[], 
                            value=None,
                            interactive=True
                        )                    
    
                    # Buttons
                    with gr.Row():
                        submit_btn_x = gr.Button("Submit")
                        clear_btn_x = gr.Button("Clear")

            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    Model: VADER""")
                    sentiment_result_box_vader_x = gr.Text(label="Sentiment Result")
                    interpretation_box_vader_x = gr.Textbox(label="Gemini Interpretation", lines=3, max_lines=5, autoscroll=False)
                with gr.Column():
                    gr.HTML("""
                    Model: TEXTBLOB""")
                    sentiment_result_box_textblob_x = gr.Text(label="Sentiment Result")
                    interpretation_box_textblob_x = gr.Textbox(label="Gemini Interpretation", lines=3, max_lines=5, autoscroll=False)
                with gr.Column():
                    gr.HTML("""
                    Model: FinBERT""")
                    sentiment_result_box_finbert_x = gr.Text(label="Sentiment Result")
                    interpretation_box_finbert_x = gr.Textbox(label="Gemini Interpretation", lines=3, max_lines=5, autoscroll=False)

            # Text output for the DUNE QUERY
            dune_query_text_x = gr.Textbox(
                label="DUNE Query is provided here. This query can be executed on www.DUNE.com using a FREE DUNE Account. To run the query and show the results in this App needs a monthly subscription of at least USD 339.", 
                visible=True, 
                lines=5, 
                max_lines=10,
                autoscroll=False,
                show_copy_button=True
            )

            def update_headline_choices_x():
                headlines = scrape_twitter()
                if headlines:
                    return gr.update(choices=headlines, value=headlines[0])
                else:
                    return gr.update(choices=[], value=None)
            
            # Add this event binding
            headline_button_x.click(fn=update_headline_choices_x, outputs=headline_dropdown_x)

            # Function to clear inputs/outputs
            def clear_fields_x():
                # Fetch fresh Headlines. Clear Sentiments and Interpretations for VADER, TEXTBLOB and FinBERT. Clear Query String.
                return update_headline_choices_x(), "", "", "", "", "", "", ""
        
            # Button bindings
            submit_btn_x.click(
                fn=analyze_sentiment_and_onchain,
                inputs=[headline_dropdown_x],
                outputs=[
                    sentiment_result_box_vader_x,
                    interpretation_box_vader_x,
                    sentiment_result_box_textblob_x,
                    interpretation_box_textblob_x,
                    sentiment_result_box_finbert_x,
                    interpretation_box_finbert_x,
                    dune_query_text_x
                    ]
            )
            clear_btn_x.click(
                fn=clear_fields_x, 
                outputs=[headline_dropdown_x,
                         sentiment_result_box_vader_x, 
                         interpretation_box_vader_x,
                         sentiment_result_box_textblob_x, 
                         interpretation_box_textblob_x,
                         sentiment_result_box_finbert_x, 
                         interpretation_box_finbert_x,
                         dune_query_text_x
                        ])


        with gr.TabItem("Sample Analysis of DUNE Query Results"):
            gr.Markdown("## 📊 BitcoinBR Daily Transfers Query")
        
            with gr.Row():
                with gr.Column():
                    query_box = gr.Textbox(
                        value=dune_query.strip(),
                        label="Dune Query (copy and use in your Dune account)",
                        lines=5,
                        max_lines=23,
                        show_copy_button=True
                    )

                with gr.Column():
                    # Display the DataFrame (not editable by default)
                    gr.Dataframe(
                        value=df,
                        headers="keys",  # use DataFrame headers
                        datatype="auto",  # infer datatypes automatically
                        interactive=False,  # set to True if you want users to modify
                        label="Query Output Table"
                    )

            gr.Markdown("## 📊 Blockchain Transfer Analytics")
            with gr.Row():
                with gr.Column():
                    gr.Image(value="./plot1.png", label="Daily Total Transfer Value per Chain")
                    gr.Markdown("""
### 📊 Analysis: Daily Total Transfer Value per Chain

This line chart compares the **daily total transfer value** across four blockchain networks over time: **Avalanche**, **Avalanche C-Chain**, **Gnosis**, and **Arbitrum**.

#### Key Observations:

- **Sparse Activity Until Mid-2022**: All chains show near-zero transfer values until around mid-2022, indicating low or inactive usage during this early period.
- **Short Spikes in Late 2022**:
  - Around **October–November 2022**, **Gnosis** and **Arbitrum** experienced sudden peaks reaching above **2.0 units** of transfer value. This suggests a brief surge in usage or a significant single-day event.
  - The **Avalanche C-Chain** also shows some small activity bursts during this window.
- **Increased Activity in Mid-2023**:
  - In **June–August 2023**, **Gnosis**, **Avalanche C-Chain**, and **Arbitrum** all show recurrent, though intermittent, spikes.
  - **Arbitrum** and **Avalanche C-Chain** seem to dominate this period with the highest peaks.
- **Flatlines in Between**: Most of the chains exhibit large time spans of **zero or near-zero** activity between spikes, indicating inconsistent usage or data reporting.

#### Insights:

- **Arbitrum and Avalanche C-Chain** emerge as more active chains in terms of high transfer spikes post mid-2022.
- **Gnosis** shows periodic relevance, but with fewer occurrences.
- **Avalanche (base)** line is nearly flat throughout, suggesting negligible or missing data.

#### Recommendations:

- Investigate the causes of the peaks — these may correspond to large transfers, airdrops, exploits, or major protocol events.
- Normalize data if comparing chains directly — units may differ unless confirmed otherwise.
- Consider moving average smoothing for better trend visibility given the sparse yet high spikes.

---
🧠 _Visual summary suggests sporadic but impactful activity across chains, with Arbitrum and Avalanche C-Chain showing dominant spikes in late 2022 and mid-2023._
                                """)
                    
                with gr.Column():
                    gr.Image(value="./plot2.png", label="Total Transfers by Chain")
                    gr.Markdown("""
### 📦 Analysis: Total Transfers by Chain

This bar chart presents a comparison of **total transfer counts** across four blockchain networks: **Arbitrum**, **Gnosis**, **Avalanche**, and **Avalanche C-Chain**.

#### Key Insights:

- **🏆 Arbitrum leads significantly**, recording the highest number of total transfers—nearly **double** that of Gnosis and about **four times** that of Avalanche and Avalanche C-Chain.
- **🥈 Gnosis ranks second,** showing moderate activity, suggesting it is an actively used chain, though not as dominant as Arbitrum.
- **🥉 Avalanche and Avalanche C-Chain** are nearly tied, each showing lower total transfers, possibly indicating limited adoption, niche use cases, or lower transaction throughput.

#### Implications:

- **Arbitrum's high transfer volume** may reflect greater adoption for DeFi, NFT activity, or strong ecosystem incentives.
- **Lower figures for Avalanche chains** do not necessarily imply lower value—some chains prioritise high-value, low-frequency transactions.
- **Gnosis’s position** as a mid-level performer suggests a balance between utility and adoption.

---
📈 _Overall, Arbitrum dominates total transfer activity, suggesting it may be the most widely used chain among the four in terms of user-level interactions._                    
                    """)
                    
            with gr.Row():
                with gr.Column():
                    gr.Image(value="./plot3.png", label="Daily Comparison Across Chains")
                    gr.Markdown("""
### 🔍 Analysis: Daily Transfer Value Comparison Across Chains

This multi-panel line chart displays the **daily total transfer value** for each blockchain—**Avalanche C-Chain**, **Gnosis**, and **Arbitrum**—independently over time, allowing for clearer per-chain trend analysis.

#### Key Observations:

- **🔷 Avalanche C-Chain**:
  - Exhibits multiple short-lived spikes in activity.
  - The largest spike reaches just over **1.6e28**, suggesting a few high-volume transactions on specific days.

- **🟢 Gnosis**:
  - Shows a dominant, isolated spike peaking above **2.0e28**—the highest among all chains—followed by relatively flat activity.
  - Implies a single or rare large transaction drove total value significantly.

- **🔴 Arbitrum**:
  - Has multiple distinct spikes spread across time, with values peaking above **2.0e28**.
  - Reflects more frequent high-value activity compared to others, possibly driven by broader ecosystem engagement.

#### Additional Notes:

- **Scale Consistency**: All subplots share a similar y-axis scale (scientific notation `1e28`), which enables fair visual comparison.
- **Low Baseline Activity**: All chains show near-zero daily totals for most of the timeline, suggesting occasional bursts rather than steady usage.

---
📊 _While all chains experience sparse but large spikes, **Arbitrum** appears more consistently active with multiple high-value days. **Gnosis** shows the single highest peak, possibly due to a one-off large transfer event._                    
                    """)
                    
                with gr.Column():
                    gr.Image(value="./plot4.png", label="Distribution of Transfer Values per Chain")
                    gr.Markdown("""
### 📦 Analysis: Distribution of Transfer Values per Chain

This plot illustrates the **distribution of daily transfer values** across four blockchain networks—**Avalanche**, **Avalanche C-Chain**, **Gnosis**, and **Arbitrum**—using a boxplot with overlaid outlier points.

#### Key Insights:

- **Low Median Activity**:
  - All four chains have their **box (IQR)** very close to **zero**, indicating that **most daily transfer values are minimal or near-zero**.
  - This reflects that high-value transactions are rare and not representative of typical daily activity.

- **Presence of Outliers**:
  - All chains show **numerous outlier points**, scattered well above the interquartile range, including values beyond **1.5** and even **2.0**.
  - **Arbitrum and Gnosis** especially show more frequent and extreme outliers, suggesting occasional high-value transactions or bursts of activity.

- **Avalanche vs Avalanche C-Chain**:
  - Both show very similar distribution shapes, implying closely matched usage patterns or overlapping transaction sources.

#### Interpretation:

- The **transfer value distributions are highly skewed** for all chains, dominated by **a few very large transactions** amidst a sea of low-volume days.
- Such skewness may imply:
  - Network-specific utility (e.g., Arbitrum for large-scale DeFi).
  - Rare institutional transactions or bridge events.
  - Low regular user activity compared to a few large players.

---
📈 _While daily averages may be low, the presence of outliers across all chains—especially Arbitrum and Gnosis—suggests significant irregular, high-value activity driving overall value._                    
                    """)
                    
            with gr.Row():
                    gr.Image(value="./plot5.png", label="3-Day Moving Average of Transfers per Chain")
                    gr.Markdown("""
### 📈 Analysis: 3-Day Moving Average of Transfers per Chain

This line chart displays the **3-day moving average** of daily transfer values across three blockchain chains (identified by chain codes `102`, `146`, and `596`) over a multi-year period.

#### Key Insights:

- **Moving Average Smoothing**:
  - The moving average has successfully smoothed out sharp single-day spikes, helping to reveal **sustained transfer patterns** rather than one-off anomalies.
  
- **Activity Timeline**:
  - From **early 2022 to mid-2023**, the moving average remains near-zero, indicating long stretches of **inactivity or minimal transfer values**.
  - **Chain 102** shows **early activity spikes** around **Q4 2022** and **Q2–Q3 2023**, with a notable peak reaching **above 2.0**.
  - **Chain 596** exhibits **sharper but fewer peaks**, clustered around **late 2022** and **mid-2023**.
  - **Chain 146** shows similar bursts of activity, slightly trailing Chain 102.

- **Comparison Across Chains**:
  - **Chain 102** appears to be the most active overall, with the **highest peaks and most persistent moving averages**.
  - **Chain 596**, despite fewer events, demonstrates **sharp rises**—likely high-value, short-duration transfer periods.
  - **Chain 146** stays between the two in terms of frequency and intensity.

#### Interpretation:

- The use of a **3-day moving average** helps distinguish **consistent periods of high activity** from isolated events.
- The overall pattern suggests that **transfer activity is episodic**, with **short-lived yet intense bursts**.

---
📊 _Chain 102 leads in sustained activity, while Chains 146 and 596 show more sporadic engagement. The 3-day moving average confirms that most transfer events are clustered in distinct high-volume windows._                    
                    """)
                            
    # On loading, fetch from Coinbase
    app.load(fn=update_headline_choices, 
             inputs=[], 
             outputs=headline_dropdown)

    gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)

if __name__ == "__main__":
    # Determine if running on Hugging Face Spaces
    on_spaces = os.environ.get("SPACE_ID") is not None
    
    # Launch the app conditionally
    app.launch(share=not on_spaces)