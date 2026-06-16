import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(
    page_title="SmartStock Advisor",
    layout="wide"
)

st.title("📈 SmartStock Advisor")
st.markdown("AI-powered Stock Analysis & Recommendation System")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if ticker:

    stock = yf.Ticker(ticker)

    df = stock.history(period="1y")

    if len(df) == 0:
        st.error("No data found")
        st.stop()

    # =========================
    # COMPANY INFO
    # =========================

    info = stock.info

    st.header("🏢 Company Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Price", round(info.get("currentPrice", 0), 2))

    with col2:
        st.metric("Market Cap", f"{info.get('marketCap',0):,}")

    with col3:
        st.metric("PE Ratio", info.get("trailingPE", "N/A"))

    # =========================
    # PRICE CHART
    # =========================

    st.header("📊 Stock Price")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close Price"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TECHNICAL ANALYSIS
    # =========================

    st.header("📈 Technical Analysis")

    rsi = RSIIndicator(df["Close"]).rsi()
    latest_rsi = float(rsi.iloc[-1])

    macd = MACD(df["Close"])
    latest_macd = float(macd.macd().iloc[-1])

    col1, col2 = st.columns(2)

    with col1:
        st.metric("RSI", round(latest_rsi, 2))

    with col2:
        st.metric("MACD", round(latest_macd, 2))

    technical_score = 50

    if latest_rsi < 30:
        technical_score += 20

    elif latest_rsi > 70:
        technical_score -= 20

    if latest_macd > 0:
        technical_score += 15

    else:
        technical_score -= 15

    technical_score = max(0, min(100, technical_score))

    # =========================
    # FORECAST
    # =========================

    st.header("🔮 10-Day Forecast")

    data = df.reset_index()

    data["Day"] = np.arange(len(data))

    X = data[["Day"]]
    y = data["Close"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(
        len(data),
        len(data) + 10
    ).reshape(-1, 1)

    forecast = model.predict(future_days)

    future_df = pd.DataFrame({
        "Day": range(1, 11),
        "Forecast Price": forecast
    })

    st.dataframe(future_df)

    future_return = (
        (forecast[-1] - df["Close"].iloc[-1])
        / df["Close"].iloc[-1]
    ) * 100

    st.metric(
        "Expected Return (10 Days)",
        f"{future_return:.2f}%"
    )

    # =========================
    # RECOMMENDATION
    # =========================

    st.header("🤖 AI Recommendation")

    score = technical_score

    if future_return > 5:
        score += 20

    elif future_return > 0:
        score += 10

    else:
        score -= 10

    score = max(0, min(100, score))

    if score >= 80:
        recommendation = "🟢 STRONG BUY"

    elif score >= 60:
        recommendation = "🟡 BUY"

    elif score >= 40:
        recommendation = "⚪ HOLD"

    else:
        recommendation = "🔴 SELL"

    st.subheader(recommendation)

    st.write(f"Overall Score: {score}/100")

    reasons = []

    if latest_rsi < 30:
        reasons.append(
            "RSI indicates oversold condition."
        )

    elif latest_rsi > 70:
        reasons.append(
            "RSI indicates overbought condition."
        )

    else:
        reasons.append(
            "RSI is within normal range."
        )

    if latest_macd > 0:
        reasons.append(
            "MACD suggests bullish momentum."
        )

    else:
        reasons.append(
            "MACD suggests bearish momentum."
        )

    if future_return > 0:
        reasons.append(
            "Forecast indicates potential upside."
        )

    else:
        reasons.append(
            "Forecast indicates downside risk."
        )

    for r in reasons:
        st.write("•", r)

    st.info(
        "This tool is for educational purposes only and not financial advice."
    )
