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

@st.cache_data(ttl=3600)
def get_market_context():
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo")
        if spy_hist.empty:
            return 0, "Market data unavailable"
        spy_sma20 = spy_hist["Close"].tail(20).mean()
        spy_sma50 = spy_hist["Close"].tail(50).mean()
        spy_current = spy_hist["Close"].iloc[-1]
        spy_change_1mo = ((spy_current - spy_hist["Close"].iloc[-21]) / spy_hist["Close"].iloc[-21]) * 100
        if spy_current > spy_sma20 > spy_sma50 and spy_change_1mo > 2:
            return 2, f"🟢 Broad market is in a strong uptrend (S&P 500 up {round(spy_change_1mo, 1)}% this month)"
        elif spy_current > spy_sma20:
            return 1, f"🟢 Broad market is trending upward (S&P 500 up {round(spy_change_1mo, 1)}% this month)"
        elif spy_current < spy_sma20 < spy_sma50 and spy_change_1mo < -2:
            return -2, f"🔴 Broad market is in a strong downtrend (S&P 500 down {round(abs(spy_change_1mo), 1)}% this month)"
        elif spy_current < spy_sma20:
            return -1, f"🔴 Broad market is trending downward (S&P 500 down {round(abs(spy_change_1mo), 1)}% this month)"
        else:
            return 0, f"🟡 Broad market is neutral (S&P 500 {round(spy_change_1mo, 1)}% this month)"
    except:
        return 0, "🟡 Market context unavailable"

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
    bb = ta.bbands(h["Close"], length=20)
    bb_upper_col = [c for c in bb.columns if c.startswith("BBU")]
    bb_lower_col = [c for c in bb.columns if c.startswith("BBL")]
    h["BB_upper"] = bb[bb_upper_col[0]] if bb_upper_col else np.nan
    h["BB_lower"] = bb[bb_lower_col[0]] if bb_lower_col else np.nan
    h["OBV"] = ta.obv(h["Close"], h["Volume"])
    h["WILLR"] = ta.willr(h["High"], h["Low"], h["Close"], length=14)
    latest = h.iloc[-1]
    current_price = latest["Close"]
    week_high_52 = h["Close"].tail(252).max()
    week_low_52 = h["Close"].tail(252).min()

    rsi = latest["RSI"]
    if pd.notna(rsi):
        if rsi < 30:
            signals.append(("RSI", f"🟢 Oversold at {round(rsi, 1)} — historically precedes a recovery. Readings below 30 indicate selling may be exhausted.", "+", 1))
            score += 1
        elif rsi > 70:
            signals.append(("RSI", f"🔴 Overbought at {round(rsi, 1)} — stock may be due for a pullback. Readings above 70 indicate buyers may be exhausted.", "-", 1))
            score -= 1
        else:
            signals.append(("RSI", f"🟡 Neutral at {round(rsi, 1)} — no strong overbought or oversold signal.", "0", 1))

    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_signal"]):
        if latest["MACD"] > latest["MACD_signal"]:
            signals.append(("MACD", "🟢 Bullish crossover — short-term momentum is accelerating upward. Most reliable when confirmed by volume.", "+", 1))
            score += 1
        else:
            signals.append(("MACD", "🔴 Bearish crossover — short-term momentum is declining. Watch for reversal before considering a buy.", "-", 1))
            score -= 1

    if pd.notna(latest["SMA20"]) and pd.notna(latest["SMA50"]):
        if current_price > latest["SMA20"] > latest["SMA50"]:
            signals.append(("Moving Averages", "🟢 Price is above both the 20-day and 50-day averages — textbook uptrend structure.", "+", 2))
            score += 2
        elif current_price < latest["SMA20"] < latest["SMA50"]:
            signals.append(("Moving Averages", "🔴 Price is below both the 20-day and 50-day averages — textbook downtrend structure.", "-", 2))
            score -= 2
        else:
            signals.append(("Moving Averages", "🟡 Price is between the averages — mixed structure, no confirmed trend direction.", "0", 2))

    if pd.notna(latest["BB_upper"]) and pd.notna(latest["BB_lower"]):
        bb_range = latest["BB_upper"] - latest["BB_lower"]
        if bb_range > 0:
            bb_position = (current_price - latest["BB_lower"]) / bb_range
            if bb_position > 0.95:
                signals.append(("Bollinger Bands", "🔴 Price is at the upper band — statistically expensive relative to recent volatility. High probability of mean reversion.", "-", 1))
                score -= 1
            elif bb_position < 0.05:
                signals.append(("Bollinger Bands", "🟢 Price is at the lower band — statistically cheap relative to recent volatility. High probability of bounce.", "+", 1))
                score += 1
            else:
                signals.append(("Bollinger Bands", f"🟡 Price is at {round(bb_position * 100)}% of the Bollinger Band range — within normal volatility bounds.", "0", 1))

    obv_series = h["OBV"].dropna()
    if len(obv_series) > 10:
        obv_trend = obv_series.iloc[-1] - obv_series.iloc[-10]
        price_trend = h["Close"].iloc[-1] - h["Close"].iloc[-10]
        if obv_trend > 0 and price_trend > 0:
            signals.append(("On-Balance Volume", "🟢 Volume is flowing in as price rises — buyers are committed. Confirms the upward move is real.", "+", 1))
            score += 1
        elif obv_trend < 0 and price_trend < 0:
            signals.append(("On-Balance Volume", "🔴 Volume is flowing out as price falls — sellers are committed. Confirms the downward move is real.", "-", 1))
            score -= 1
        elif obv_trend > 0 and price_trend < 0:
            signals.append(("On-Balance Volume", "🟢 Volume rising while price falls — buyers quietly accumulating. Often precedes a reversal upward.", "+", 1))
            score += 1
        else:
            signals.append(("On-Balance Volume", "🟡 Volume and price not confirming each other — mixed signal.", "0", 1))

    willr = latest["WILLR"]
    if pd.notna(willr):
        if willr < -80:
            signals.append(("Williams %R", f"🟢 Deeply oversold at {round(willr, 1)} — strong bounce signal, complements RSI.", "+", 1))
            score += 1
        elif willr > -20:
            signals.append(("Williams %R", f"🔴 Deeply overbought at {round(willr, 1)} — pullback signal, complements RSI.", "-", 1))
            score -= 1
        else:
            signals.append(("Williams %R", f"🟡 Neutral at {round(willr, 1)} — no extreme signal.", "0", 1))

    if week_high_52 > 0:
        pct_from_high = ((current_price - week_high_52) / week_high_52) * 100
        pct_from_low = ((current_price - week_low_52) / week_low_52) * 100
        if pct_from_high > -5:
            signals.append(("52-Week Range", f"🟢 Within 5% of 52-week high — strong long-term momentum. Breakouts above yearly highs often continue.", "+", 1))
            score += 1
        elif pct_from_low < 10:
            signals.append(("52-Week Range", f"🔴 Near 52-week low (down {round(abs(pct_from_high), 1)}% from high) — significant long-term weakness.", "-", 1))
            score -= 1
        else:
            signals.append(("52-Week Range", f"🟡 Mid-range (down {round(abs(pct_from_high), 1)}% from 52-week high) — no extreme positioning.", "0", 1))

    avg_volume = history["Volume"].tail(20).mean()
    latest_volume = latest["Volume"]
    if latest_volume > avg_volume * 1.5:
        signals.append(("Volume", "🟢 Volume 50%+ above 20-day average — strong market interest. High-volume moves are more likely to continue.", "+", 1))
        score += 1
    elif latest_volume < avg_volume * 0.5:
        signals.append(("Volume", "🔴 Volume below normal — low conviction. Moves on low volume are more likely to reverse.", "-", 1))
        score -= 1
    else:
        signals.append(("Volume", "🟡 Volume is normal — no unusual activity.", "0", 1))

    recent_prices = history["Close"].tail(20)
    price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
    if price_change > 5:
        signals.append(("Price Trend (20 days)", f"🟢 Up {round(price_change, 1)}% over 20 trading days — sustained short-term momentum.", "+", 1))
        score += 1
    elif price_change < -5:
        signals.append(("Price Trend (20 days)", f"🔴 Down {round(abs(price_change), 1)}% over 20 trading days — sustained short-term weakness.", "-", 1))
        score -= 1
    else:
        signals.append(("Price Trend (20 days)", f"🟡 Relatively flat ({round(price_change, 1)}%) over 20 trading days.", "0", 1))

    return signals, score

def get_news_and_sentiment(ticker, page_size=10):
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&sortBy=publishedAt&language=en&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    )
    news_data = requests.get(news_url).json()
    articles_out = []
    sentiment_scores = []
    if news_data.get("articles"):
        for i, article in enumerate(news_data["articles"]):
            text = f"{article['title']}. {article.get('description', '')}"[:512]
            result = sentiment_model(text)[0]
            label = result["label"]
            score = result["score"]
            recency_weight = 1.0 if i < 3 else 0.7
            if label == "positive":
                sentiment_scores.append(1 * recency_weight)
                badge = "🟢 Positive"
            elif label == "negative":
                sentiment_scores.append(-1 * recency_weight)
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

def compute_full_analysis(history, ticker):
    signals, tech_score = analyze_technicals(history)
    articles, avg_sentiment = get_news_and_sentiment(ticker, page_size=10)
    market_score, market_description = get_market_context()
    sentiment_score = round(avg_sentiment * 3)
    market_weight = market_score * 2
    raw_total = tech_score + sentiment_score + market_weight
    max_possible = 18
    confidence_pct = round((raw_total / max_possible) * 100)
    confidence_pct = max(-100, min(100, confidence_pct))
    top_signals = []
    for s in signals:
        pos_count = len([x for x in top_signals if x[2] == "+"])
        neg_count = len([x for x in top_signals if x[2] == "-"])
        if s[2] == "+" and pos_count < 2:
            top_signals.append(s)
        elif s[2] == "-" and neg_count < 1:
            top_signals.append(s)
    return {
        "signals": signals,
        "tech_score": tech_score,
        "articles": articles,
        "avg_sentiment": avg_sentiment,
        "sentiment_score": sentiment_score,
        "market_score": market_score,
        "market_description": market_description,
        "confidence_pct": confidence_pct,
        "top_signals": top_signals
    }

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

def render_confidence_gauge(confidence_pct):
    abs_conf = abs(confidence_pct)
    direction, rating, _, _ = format_verdict(confidence_pct)
    if confidence_pct >= 20:
        color = "#28a745"
    elif confidence_pct <= -20:
        color = "#dc3545"
    else:
        color = "#ffc107"
    st.markdown(
        f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: bold; font-size: 16px;">{direction} — {rating}</span>
                <span style="font-weight: bold; font-size: 16px; color: {color};">{abs_conf}%</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 8px; height: 20px; overflow: hidden;">
                <div style="
                    width: {abs_conf}%;
                    background-color: {color};
                    height: 100%;
                    border-radius: 8px;
                "></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                <span style="font-size: 11px; color: #888;">0%</span>
                <span style="font-size: 11px; color: #888;">100%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(ttl=3600)
def get_quick_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="5y")
        if history.empty:
            return None
        price = round(history["Close"].iloc[-1], 2)
        analysis = compute_full_analysis(history, ticker)
        return {
            "name": ticker,
            "price": price,
            "confidence": analysis["confidence_pct"],
            "ticker": ticker,
            "updated": datetime.now().strftime("%H:%M")
        }
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
                history = yf.Ticker(ticker).history(period="1mo")
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
            reasons.append(f"Supporting signals: {', '.join([s[0] for s in positive_signals[:3]])}")
    elif confidence >= 50 and gain_loss_pct < 0:
        verdict = "🟡 HOLD"
        reasons.append(f"Analysis is bullish at {confidence}% — recovery is indicated")
        reasons.append(f"Position is down {round(abs(gain_loss_pct), 1)}% but signals suggest patience")
        reasons.append("Selling now locks in a loss before the indicated recovery")
    elif confidence <= -50:
        verdict = "🔴 SELL"
        reasons.append(f"Strong bearish signal at {abs(confidence)}% confidence")
        if negative_signals:
            reasons.append(f"Warning signals: {', '.join([s[0] for s in negative_signals[:3]])}")
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
    with st.container(border=True):
        st.markdown(f"### {data['ticker']}")
        st.write(f"Price: **${data['price']}**")
        render_confidence_gauge(data["confidence"])
        st.caption(f"Based on 9 technical signals, news sentiment, and market context. Updated {data.get('updated', 'recently')}.")
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

    market_score, market_description = get_market_context()
    if market_score >= 1:
        st.success(f"**Market Context:** {market_description}")
    elif market_score <= -1:
        st.error(f"**Market Context:** {market_description}")
    else:
        st.warning(f"**Market Context:** {market_description}")

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
            stock = yf.Ticker(ticker)
            history = stock.history(period="5y")
            if history.empty:
                st.warning(f"Could not load data for {ticker}")
                continue
            current_price = round(history["Close"].iloc[-1], 2)
            gain_loss_pct = ((current_price - buy_price) / buy_price) * 100
            gain_loss_dollar = (current_price - buy_price) * shares
            analysis = compute_full_analysis(history, ticker)
            signals = analysis["signals"]
            confidence = analysis["confidence_pct"]
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

                render_confidence_gauge(confidence)
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
        except Exception as e:
            st.warning(f"Error loading {ticker}: {str(e)}")

def show_detail(ticker):
    if st.button("← Back to Home"):
        go_home()
        st.rerun()

    stock = yf.Ticker(ticker)
    history = stock.history(period="5y")
    price = round(history["Close"].iloc[-1], 2) if not history.empty else "N/A"
    previous_close = round(history["Close"].iloc[-2], 2) if len(history) > 1 else "N/A"
    volume = int(history["Volume"].iloc[-1]) if not history.empty else "N/A"

    st.title(f"{ticker}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${price}")
    col2.metric("Previous Close", f"${previous_close}")
    col3.metric("Volume", f"{volume:,}" if isinstance(volume, int) else volume)

    with st.spinner("Running full analysis..."):
        analysis = compute_full_analysis(history, ticker)

    confidence_pct = analysis["confidence_pct"]

    st.markdown("---")
    st.subheader("📋 Algorithmic Verdict")
    render_confidence_gauge(confidence_pct)

    if analysis["market_score"] >= 1:
        st.success(f"**Market Context:** {analysis['market_description']}")
    elif analysis["market_score"] <= -1:
        st.error(f"**Market Context:** {analysis['market_description']}")
    else:
        st.warning(f"**Market Context:** {analysis['market_description']}")

    st.markdown("**What's driving this signal:**")
    for s in analysis["top_signals"]:
        st.write(f"• {s[1]}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Chart",
        "📊 Technical Analysis",
        "🔁 Historical Patterns",
        "📰 News & Sentiment"
    ])

    with tab1:
        st.subheader("Price Chart")
        period = st.selectbox("Time Range", ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"])
        period_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "5 Years": len(history)}
        st.line_chart(history["Close"].tail(period_map[period]))

    with tab2:
        st.subheader("Technical Indicators")
        st.caption("9 signals evaluated. Click any indicator to see what it means and why it matters.")
        positive_count = sum(1 for s in analysis["signals"] if s[2] == "+")
        negative_count = sum(1 for s in analysis["signals"] if s[2] == "-")
        neutral_count = sum(1 for s in analysis["signals"] if s[2] == "0")
        st.write(f"🟢 {positive_count} bullish &nbsp;&nbsp; 🔴 {negative_count} bearish &nbsp;&nbsp; 🟡 {neutral_count} neutral")
        st.markdown("---")
        for indicator, description, direction, weight in analysis["signals"]:
            with st.expander(f"{indicator}"):
                st.write(description)
                if weight > 1:
                    st.caption("⚡ High-weight signal — this indicator has double impact on the final score.")

    with tab3:
        st.subheader("Historical Pattern Matching")
        st.caption("The algorithm found the 5 most similar 30-day price shapes in the last 5 years and recorded what happened after each one.")
        reliability_note = {
            "Today": "High", "This Week": "High", "This Month": "Moderate",
            "3 Months": "Moderate", "6 Months": "Lower", "1 Year+": "Lowest"
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
                        direction_label = "📈 UP" if data["avg_change"] > 0 else "📉 DOWN"
                        st.markdown(f"### {direction_label}")
                        st.write(f"Avg change: **{data['avg_change']}%**")
                        st.write(f"Went up in **{data['positive_rate']}%** of past cases")
                        st.caption(f"Based on {data['sample_size']} similar situations")
                    else:
                        st.write("Not enough data")

    with tab4:
        st.subheader("News & Sentiment")
        st.caption("10 most recent articles analyzed. Recent articles weighted more heavily in the score.")
        avg_sentiment = analysis["avg_sentiment"]
        if avg_sentiment > 0.2:
            st.success("Overall news sentiment: POSITIVE 🟢")
        elif avg_sentiment < -0.2:
            st.error("Overall news sentiment: NEGATIVE 🔴")
        else:
            st.warning("Overall news sentiment: NEUTRAL 🟡")
        st.markdown("---")
        for article in analysis["articles"]:
            with st.expander(f"{article['badge']} {article['title']}"):
                st.caption(f"{article['source']} — {article['date']}")
                st.write(article["description"])
                st.write(f"Sentiment confidence: **{article['confidence']}%**")

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
