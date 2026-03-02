import streamlit as st
import yfinance as yf
import requests
import os
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
from dotenv import load_dotenv
from transformers import pipeline
from datetime import datetime

load_dotenv()

yf_session = requests.Session()
yf_session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD", "password")

WATCHLIST_FILE = "my_watchlist.json"

PENNY_SEED = [
    "SNDL", "CLOV", "EXPR", "BBIG", "NKLA", "RIDE", "WKHS", "GOEV",
    "PHUN", "CEAD", "MARK", "PALI", "ATER", "GFAI",
    "SBEV", "HCDI", "IMPP", "BFRI", "MGOL", "MEGL"
]

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "wrong_password" not in st.session_state:
    st.session_state.wrong_password = False

if not st.session_state.authenticated:
    if st.session_state.wrong_password:
        col1, col2 = st.columns(2)
        with col1:
            st.title("🔒 Access Required")
            st.warning("You must say the magic word to try again.")
            please_input = st.text_input("...", placeholder="Say the magic word")
            if st.button("Submit"):
                if please_input.strip().lower() == "please":
                    st.session_state.wrong_password = False
                    st.rerun()
                else:
                    st.error("That's not it.")
        with col2:
            ah_text = "AH AH AH! " * 150
            st.markdown(
                f"""
                <div style="
                    background-color: black;
                    color: red;
                    font-weight: bold;
                    font-size: 18px;
                    height: 600px;
                    overflow: hidden;
                    padding: 20px;
                    word-wrap: break-word;
                    line-height: 1.8;
                ">
                {ah_text}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image("https://media.giphy.com/media/4NnSe87mg3h25JYIDh/giphy.gif", width=400)
            st.error("You didn't say the magic word.")
    else:
        st.title("🔒 Access Required")
        st.markdown(
            """
            <iframe width="560" height="315"
            src="https://www.youtube.com/embed/RfiQYRn7fBg"
            frameborder="0" allowfullscreen>
            </iframe>
            """,
            unsafe_allow_html=True
        )
        password_input = st.text_input("Enter password", type="password")
        if st.button("Enter"):
            if password_input == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.session_state.wrong_password = True
                st.rerun()
    st.stop()

@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

sentiment_model = load_sentiment_model()

if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

def go_to_detail(ticker):
    st.session_state.selected_ticker = ticker
    st.session_state.page = "detail"

def go_home():
    st.session_state.page = "home"

def load_my_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    return []

def save_my_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f)

def normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)

def find_similar_patterns(history, pattern_days=30, top_n=5):
    closes = history["Close"].reset_index(drop=True)
    current_pattern = normalize(closes.tail(pattern_days)).values
    matches = []
    for i in range(pattern_days, len(closes) - 252):
        window = normalize(closes.iloc[i - pattern_days:i]).values
        if len(window) != pattern_days:
            continue
        correlation = np.corrcoef(current_pattern, window)[0, 1]
        matches.append((i, correlation))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_n]

def get_outcome_after(history, start_idx, days):
    closes = history["Close"].reset_index(drop=True)
    end_idx = start_idx + days
    if end_idx >= len(closes):
        return None
    return ((closes.iloc[end_idx] - closes.iloc[start_idx]) / closes.iloc[start_idx]) * 100

def analyze_timeframes(history, similar_pattern_indices):
    timeframes = {"Today": 1, "This Week": 5, "This Month": 21, "3 Months": 63, "6 Months": 126, "1 Year+": 252}
    results = {}
    for label, days in timeframes.items():
        outcomes = [get_outcome_after(history, idx, days) for idx, _ in similar_pattern_indices]
        outcomes = [o for o in outcomes if o is not None]
        if outcomes:
            results[label] = {
                "avg_change": round(np.mean(outcomes), 2),
                "positive_rate": round(sum(1 for o in outcomes if o > 0) / len(outcomes) * 100),
                "sample_size": len(outcomes)
            }
        else:
            results[label] = None
    return results

def analyze_technicals(history):
    signals = []
    score = 0
    h = history.copy()
    h["RSI"] = ta.rsi(h["Close"], length=14)
    macd = ta.macd(h["Close"])
    h["MACD"] = macd["MACD_12_26_9"]
    h["MACD_signal"] = macd["MACDs_12_26_9"]
    h["SMA20"] = ta.sma(h["Close"], length=20)
    h["SMA50"] = ta.sma(h["Close"], length=50)
    latest = h.iloc[-1]
    current_price = latest["Close"]

    rsi = latest["RSI"]
    if pd.notna(rsi):
        if rsi < 30:
            signals.append(("RSI", "🟢 Oversold — potential bounce incoming", "+"))
            score += 1
        elif rsi > 70:
            signals.append(("RSI", "🔴 Overbought — potential pullback incoming", "-"))
            score -= 1
        else:
            signals.append(("RSI", f"🟡 Neutral ({round(rsi, 1)})", "0"))

    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_signal"]):
        if latest["MACD"] > latest["MACD_signal"]:
            signals.append(("MACD", "🟢 Bullish crossover — upward momentum", "+"))
            score += 1
        else:
            signals.append(("MACD", "🔴 Bearish crossover — downward momentum", "-"))
            score -= 1

    if pd.notna(latest["SMA20"]) and pd.notna(latest["SMA50"]):
        if current_price > latest["SMA20"] > latest["SMA50"]:
            signals.append(("Moving Averages", "🟢 Price above both averages — strong uptrend", "+"))
            score += 2
        elif current_price < latest["SMA20"] < latest["SMA50"]:
            signals.append(("Moving Averages", "🔴 Price below both averages — strong downtrend", "-"))
            score -= 2
        else:
            signals.append(("Moving Averages", "🟡 Mixed signals — no clear trend", "0"))

    avg_volume = history["Volume"].tail(20).mean()
    latest_volume = latest["Volume"]
    if latest_volume > avg_volume * 1.5:
        signals.append(("Volume", "🟢 High volume — strong market interest", "+"))
        score += 1
    elif latest_volume < avg_volume * 0.5:
        signals.append(("Volume", "🔴 Low volume — weak market interest", "-"))
        score -= 1
    else:
        signals.append(("Volume", "🟡 Normal volume", "0"))

    recent_prices = history["Close"].tail(20)
    price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
    if price_change > 5:
        signals.append(("Price Trend (20 days)", f"🟢 Up {round(price_change, 1)}% over 20 days", "+"))
        score += 1
    elif price_change < -5:
        signals.append(("Price Trend (20 days)", f"🔴 Down {round(abs(price_change), 1)}% over 20 days", "-"))
        score -= 1
    else:
        signals.append(("Price Trend (20 days)", f"🟡 Relatively flat ({round(price_change, 1)}%)", "0"))

    return signals, score

def get_news_and_sentiment(name, page_size=5):
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={name}&sortBy=publishedAt&language=en&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    )
    news_data = requests.get(news_url).json()
    articles_out = []
    sentiment_scores = []
    if news_data.get("articles"):
        for article in news_data["articles"]:
            text = f"{article['title']}. {article.get('description', '')}"[:512]
            result = sentiment_model(text)[0]
            label = result["label"]
            score = result["score"]
            if label == "positive":
                sentiment_scores.append(1)
                badge = "🟢 Positive"
            elif label == "negative":
                sentiment_scores.append(-1)
                badge = "🔴 Negative"
            else:
                sentiment_scores.append(0)
                badge = "🟡 Neutral"
            articles_out.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "date": article["publishedAt"][:10],
                "description": article.get("description", ""),
                "badge": badge,
                "confidence": round(score * 100)
            })
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return articles_out, avg_sentiment

def compute_confidence(history, name):
    _, tech_score = analyze_technicals(history)
    _, avg_sentiment = get_news_and_sentiment(name, page_size=3)
    sentiment_score = round(avg_sentiment * 3)
    total_score = tech_score + sentiment_score
    confidence_pct = round((total_score / 8) * 100)
    return max(-100, min(100, confidence_pct))

def format_verdict(confidence_pct):
    abs_conf = abs(confidence_pct)
    direction = "📈 UP" if confidence_pct >= 0 else "📉 DOWN"
    if abs_conf >= 50:
        rating = "High"
    elif abs_conf >= 20:
        rating = "Moderate"
    else:
        rating = "Low"
    if abs_conf < 20:
        return "➡️ UNCLEAR", "Low", abs_conf, "🟡"
    return f"{direction}", rating, abs_conf, "🟢" if confidence_pct >= 0 else "🔴"

@st.cache_data(ttl=3600)
def get_quick_analysis(ticker):
    try:
        stock = yf.Ticker(ticker, session=yf_session)
        history = stock.history(period="5y")
        if history.empty:
            return None
        price = round(history["Close"].iloc[-1], 2)
        confidence_pct = compute_confidence(history, ticker)
        return {"name": ticker, "price": price, "confidence": confidence_pct, "ticker": ticker}
    except:
        return None

@st.cache_data(ttl=3600)
def get_trending_tickers():
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US?count=10"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        data = response.json()
        quotes = data["finance"]["result"][0]["quotes"]
        return [q["symbol"] for q in quotes if "." not in q["symbol"]][:10]
    except:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "WMT"]

@st.cache_data(ttl=3600)
def get_active_penny_stocks():
    try:
        results = []
        for ticker in PENNY_SEED:
            try:
                history = yf.Ticker(ticker, session=yf_session).history(period="1mo")
                if history.empty:
                    continue
                price = history["Close"].iloc[-1]
                volume = history["Volume"].mean()
                if price and price < 5:
                    results.append({"ticker": ticker, "price": price, "volume": volume})
            except:
                continue
        results.sort(key=lambda x: x["volume"], reverse=True)
        return [r["ticker"] for r in results[:10]]
    except:
        return PENNY_SEED[:10]

def get_buy_hold_sell(confidence, gain_loss_pct, signals):
    positive_signals = [s for s in signals if s[2] == "+"]
    negative_signals = [s for s in signals if s[2] == "-"]
    reasons = []
    if confidence >= 50 and gain_loss_pct >= 0:
        verdict = "🟢 BUY MORE"
        reasons.append(f"Strong bullish signal at {confidence}% confidence")
        reasons.append(f"Position is up {round(gain_loss_pct, 1)}% — momentum is in your favor")
        if positive_signals:
            reasons.append(f"Supporting signals: {', '.join([s[0] for s in positive_signals])}")
    elif confidence >= 50 and gain_loss_pct < 0:
        verdict = "🟡 HOLD"
        reasons.append(f"Analysis is bullish at {confidence}% — recovery is indicated")
        reasons.append(f"Position is down {round(abs(gain_loss_pct), 1)}% but signals suggest patience")
        reasons.append("Selling now locks in a loss before the indicated recovery")
    elif confidence <= -50:
        verdict = "🔴 SELL"
        reasons.append(f"Strong bearish signal at {abs(confidence)}% confidence")
        if negative_signals:
            reasons.append(f"Warning signals: {', '.join([s[0] for s in negative_signals])}")
        if gain_loss_pct < 0:
            reasons.append(f"Position already down {round(abs(gain_loss_pct), 1)}% — further decline indicated")
        else:
            reasons.append("Consider locking in gains before conditions deteriorate")
    elif confidence <= -20 and gain_loss_pct < -15:
        verdict = "🔴 SELL"
        reasons.append(f"Moderate bearish signal with a {round(abs(gain_loss_pct), 1)}% loss")
        reasons.append("Downside risk outweighs recovery probability at this point")
    elif -20 <= confidence <= 20:
        verdict = "🟡 HOLD"
        reasons.append("Signals are mixed — no strong case to act in either direction")
        reasons.append("Wait for a clearer trend before changing your position")
    else:
        verdict = "🟡 HOLD"
        reasons.append(f"Signal is {confidence}% — not strong enough to act decisively")
        reasons.append("Monitor for a stronger signal before moving")
    return verdict, reasons

def render_stock_card(data, section_key):
    if not data:
        return
    direction, rating, abs_conf, color = format_verdict(data["confidence"])
    with st.container(border=True):
        st.markdown(f"### {data['ticker']}")
        st.write(f"**{data['name']}**")
        st.write(f"Price: **${data['price']}**")
        st.markdown(f"**Algorithmic outlook:** {direction} — {rating} ({abs_conf}%)")
        st.caption("Based on technical indicators and recent news sentiment.")
        if st.button("Full Analysis →", key=f"{section_key}_{data['ticker']}"):
            go_to_detail(data["ticker"])
            st.rerun()

def show_home():
    st.title("📈 Stock Analyzer")
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        search_input = st.text_input("Search any stock ticker", placeholder="e.g. AAPL, TSLA, NVDA").upper()
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Analyze →", use_container_width=True) and search_input:
            go_to_detail(search_input)
            st.rerun()

    st.markdown("---")
    st.subheader("🔥 Trending Now")
    st.caption("Most-watched tickers on US markets today. Click any card for the full breakdown.")
    trending_tickers = get_trending_tickers()
    with st.spinner("Analyzing trending stocks..."):
        cols = st.columns(2)
        for i, ticker in enumerate(trending_tickers):
            with cols[i % 2]:
                render_stock_card(get_quick_analysis(ticker), "trend")

    st.markdown("---")
    st.subheader("⚠️ High Risk — Penny Stocks")
    st.caption("All under $5. Extremely volatile. Do not invest more than you can afford to lose entirely.")
    penny_tickers = get_active_penny_stocks()
    with st.spinner("Analyzing penny stocks..."):
        cols = st.columns(2)
        for i, ticker in enumerate(penny_tickers):
            with cols[i % 2]:
                render_stock_card(get_quick_analysis(ticker), "penny")

def show_my_watchlist():
    st.title("📁 My Watchlist")
    st.caption("⚠️ Algorithmic analysis only — not financial advice. Do not make investment decisions based solely on this data.")
    st.markdown("---")

    my_list = load_my_watchlist()

    with st.expander("➕ Add a position"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_ticker = st.text_input("Ticker", placeholder="e.g. AAPL").upper()
        with col2:
            buy_price = st.number_input("Your buy price ($)", min_value=0.01, value=1.00, step=0.01)
        with col3:
            shares = st.number_input("Shares owned", min_value=1, value=1, step=1)
        if st.button("Add to Watchlist"):
            if new_ticker:
                my_list.append({
                    "ticker": new_ticker,
                    "buy_price": buy_price,
                    "shares": shares,
                    "added": datetime.now().strftime("%Y-%m-%d")
                })
                save_my_watchlist(my_list)
                st.success(f"Added {new_ticker}")
                st.rerun()

    if not my_list:
        st.write("No positions added yet. Use the form above to add your first stock.")
        return

    for i, position in enumerate(my_list):
        ticker = position["ticker"]
        buy_price = position["buy_price"]
        shares = position["shares"]
        try:
            stock = yf.Ticker(ticker, session=yf_session)
            history = stock.history(period="5y")
            if history.empty:
                st.warning(f"Could not load data for {ticker}")
                continue
            current_price = round(history["Close"].iloc[-1], 2)
            name = ticker

            if not history.empty:
                gain_loss_pct = ((current_price - buy_price) / buy_price) * 100
                gain_loss_dollar = (current_price - buy_price) * shares
                signals, _ = analyze_technicals(history)
                confidence = compute_confidence(history, name)
                verdict, reasons = get_buy_hold_sell(confidence, gain_loss_pct, signals)

                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.markdown(f"### {ticker}")
                    with col2:
                        st.metric("Current Price", f"${round(current_price, 2)}")
                    with col3:
                        st.metric("Buy Price", f"${round(buy_price, 2)}")
                    with col4:
                        st.metric("Your Gain / Loss", f"${round(gain_loss_dollar, 2)}", f"{round(gain_loss_pct, 1)}%")

                    st.markdown(f"## {verdict}")
                    for reason in reasons:
                        st.write(f"• {reason}")

                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        if st.button("Full Analysis", key=f"detail_{ticker}_{i}"):
                            go_to_detail(ticker)
                            st.rerun()
                    with col_b:
                        if st.button("Remove Position", key=f"remove_{ticker}_{i}"):
                            my_list.pop(i)
                            save_my_watchlist(my_list)
                            st.rerun()
            else:
                st.warning(f"Could not load data for {ticker}")
        except Exception as e:
            st.warning(f"Error loading {ticker}: {str(e)}")

def show_detail(ticker):
    if st.button("← Back to Home"):
        go_home()
        st.rerun()

    stock = yf.Ticker(ticker, session=yf_session)
    history = stock.history(period="5y")
    price = round(history["Close"].iloc[-1], 2) if not history.empty else "N/A"
    previous_close = round(history["Close"].iloc[-2], 2) if len(history) > 1 else "N/A"
    volume = int(history["Volume"].iloc[-1]) if not history.empty else "N/A"
    market_cap = "N/A"

    st.title(f"{ticker}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${price}")
    col2.metric("Previous Close", f"${previous_close}")
    col3.metric("Volume", f"{volume:,}" if isinstance(volume, int) else volume)
    col4.metric("Market Cap", market_cap)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Chart",
        "📊 Technical Analysis",
        "🔁 Historical Patterns",
        "📰 News & Sentiment",
        "📋 Final Score"
    ])

    with tab1:
        st.subheader("Price Chart")
        period = st.selectbox("Time Range", ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"])
        period_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "5 Years": len(history)}
        st.line_chart(history["Close"].tail(period_map[period]))

    with tab2:
        st.subheader("Technical Indicators")
        st.caption("Each row is one signal the algorithm evaluated. Click to see what it means.")
        signals, _ = analyze_technicals(history)
        for indicator, description, direction in signals:
            with st.expander(f"{indicator}"):
                st.write(description)

    with tab3:
        st.subheader("Historical Pattern Matching")
        st.caption("The algorithm found the 5 most similar 30-day price shapes in the last 5 years and recorded what happened after each one.")
        reliability_note = {
            "Today": "High",
            "This Week": "High",
            "This Month": "Moderate",
            "3 Months": "Moderate",
            "6 Months": "Lower",
            "1 Year+": "Lowest"
        }
        with st.spinner("Scanning 5 years of history for similar patterns..."):
            similar = find_similar_patterns(history)
            timeframe_results = analyze_timeframes(history, similar)

        cols = st.columns(3)
        for idx, (label, data) in enumerate(timeframe_results.items()):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.caption(f"Reliability: {reliability_note[label]}")
                    if data:
                        direction = "📈 UP" if data["avg_change"] > 0 else "📉 DOWN"
                        st.markdown(f"### {direction}")
                        st.write(f"Avg change: **{data['avg_change']}%**")
                        st.write(f"Went up in **{data['positive_rate']}%** of past cases")
                        st.caption(f"Based on {data['sample_size']} similar situations")
                    else:
                        st.write("Not enough data")

    with tab4:
        st.subheader("News & Sentiment")
        st.caption("Recent articles analyzed for positive, negative, or neutral financial sentiment.")
        with st.spinner("Fetching and analyzing articles..."):
            articles, avg_sentiment = get_news_and_sentiment(ticker)
        if avg_sentiment > 0.2:
            st.success("Overall news sentiment: POSITIVE 🟢")
        elif avg_sentiment < -0.2:
            st.error("Overall news sentiment: NEGATIVE 🔴")
        else:
            st.warning("Overall news sentiment: NEUTRAL 🟡")
        st.markdown("---")
        for article in articles:
            with st.expander(f"{article['badge']} {article['title']}"):
                st.caption(f"{article['source']} — {article['date']}")
                st.write(article["description"])
                st.write(f"Sentiment confidence: **{article['confidence']}%**")

    with tab5:
        st.subheader("Final Score")
        st.caption("All signals combined into one directional score. You make the final call.")
        signals, tech_score = analyze_technicals(history)
        _, avg_sentiment = get_news_and_sentiment(ticker, page_size=3)
        sentiment_score = round(avg_sentiment * 3)
        total_score = tech_score + sentiment_score
        confidence_pct = round((total_score / 8) * 100)
        confidence_pct = max(-100, min(100, confidence_pct))

        direction, rating, abs_conf, color = format_verdict(confidence_pct)

        if color == "🟢":
            st.success(f"{direction} — {rating} ({abs_conf}%)")
        elif color == "🔴":
            st.error(f"{direction} — {rating} ({abs_conf}%)")
        else:
            st.warning(f"{direction} — {rating} ({abs_conf}%)")

        st.metric("Confidence Score", f"{abs_conf}%")
        st.write(f"**Signal strength: {rating}**")
        st.caption("This tool presents data and reasoning only. The final decision is yours.")

st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Home", use_container_width=True):
    go_home()
    st.rerun()
if st.sidebar.button("📁 My Watchlist", use_container_width=True):
    st.session_state.page = "watchlist"
    st.rerun()

if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "detail":
    show_detail(st.session_state.selected_ticker)
elif st.session_state.page == "watchlist":
    show_my_watchlist()
