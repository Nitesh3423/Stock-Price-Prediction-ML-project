
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


try:
    from newsapi import NewsApiClient
    from textblob import TextBlob
    NEWSAPI_AVAILABLE = True
except Exception:
    NEWSAPI_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -------------------- Helper Functions --------------------
def indian_number_format(n):
    """Format number in Indian style (e.g., 12,34,567.89) — best-effort."""
    try:
        s = f"{n:,.2f}"
        # Convert western grouping to indian grouping
        if ',' not in s:
            return s
        parts = s.split('.')
        left = parts[0]
        right = parts[1] if len(parts) > 1 else "00"
        left = left.replace(',', '')
        if len(left) <= 3:
            return f"{left}.{right}"
        # last 3
        last3 = left[-3:]
        rest = left[:-3]
        rest_groups = []
        while len(rest) > 2:
            rest_groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            rest_groups.append(rest)
        rest_groups.reverse()
        return ','.join(rest_groups) + ',' + last3 + '.' + right
    except Exception:
        return f"{n:,.2f}"

@st.cache_data(ttl=3600)
def load_market_data(symbol, start='2012-01-01', end=None):
    if end is None:
        end = datetime.date.today()
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    return df

@st.cache_resource
def load_trained_model(path):
    return load_model(path)

def compute_technical_indicators(df):
    df = df.copy()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA100'] = df['Close'].rolling(100).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def detect_currency(symbol):
    # Basic detection: NSE/BSE Indian tickers end with .NS or .BO
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return "INR", "₹"
    # extend here for other exchanges if desired
    return "USD", "$"

def inverse_scale(scaler, arr):
    """Safely inverse transform arrays with sklearn MinMaxScaler"""
    try:
        return scaler.inverse_transform(arr)
    except Exception:
        # If scaler was not fit (shouldn't happen), return original
        return arr

# -------------------- App Config --------------------
st.set_page_config(page_title="Stock Market Predictor", page_icon="", layout="wide")
st.title(" Stock Market Price Predictor — Advanced Dashboard")
st.caption("LSTM-based forecasting • Real-time data from Yahoo Finance • Enhanced analytics")

# -------------------- Company List --------------------
company_dict = {
    "Apple Inc. (AAPL)": "AAPL",
    "Google (Alphabet Inc.) (GOOG)": "GOOG",
    "Microsoft Corporation (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Tesla Inc. (TSLA)": "TSLA",
    "Meta Platforms (META)": "META",
    "NVIDIA Corporation (NVDA)": "NVDA",
    "Netflix Inc. (NFLX)": "NFLX",
    "Intel Corporation (INTC)": "INTC",
    "Advanced Micro Devices (AMD)": "AMD",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys Limited (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Asian Paints (ASIANPAINT.NS)": "ASIANPAINT.NS"
}

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Controls & Options")
selected_company = st.sidebar.selectbox("Select a company", list(company_dict.keys()), index=1)
stock_symbol = company_dict[selected_company]

start_date = st.sidebar.date_input("Start date", value=datetime.date(2012, 1, 1))
# data caching uses today's date by default
use_news = st.sidebar.checkbox("Enable News Sentiment (requires NewsAPI key)", value=False)
news_api_key = os.getenv("NEWSAPI_KEY", "") if use_news else ""
if use_news and not news_api_key and NEWSAPI_AVAILABLE:
    news_api_key = st.sidebar.text_input("Enter NewsAPI Key (or set NEWSAPI_KEY env var)", value="")
st.sidebar.markdown("---")
st.sidebar.markdown("Model & Data")
model_path = st.sidebar.text_input("Trained model file", value="Stock Predictions Model.keras")
st.sidebar.info("Place your trained Keras model file in the same folder (e.g., Stock Predictions Model.keras)")

# -------------------- Load Model (resource cached) --------------------
try:
    model = load_trained_model(model_path)
except Exception as e:
    st.error(f"Failed to load model from '{model_path}'. Error: {e}")
    st.stop()

# -------------------- Load Market Data --------------------
with st.spinner(f"Fetching {stock_symbol} data..."):
    try:
        data = load_market_data(stock_symbol, start=start_date)
    except Exception as e:
        st.error("Error fetching data: " + str(e))
        st.stop()

if data.empty:
    st.error("No data returned for symbol. Double-check the ticker (e.g., use AMZN or TCS.NS).")
    st.stop()

currency_name, currency_symbol = detect_currency(stock_symbol)

# -------------------- Header / Summary --------------------
last_date = data.index[-1].date()
st.markdown(
    f"<div style='background:#123243;padding:12px;border-radius:8px;color:white'>"
    f"<h2 style='margin:4px 0'>🔎 {selected_company}</h2>"
    f"<div style='color:#d0e7ff'>Data range: <b>{data.index[0].date()}</b> → <b>{last_date}</b> • Currency: <b>{currency_name}</b></div>"
    f"</div>",
    unsafe_allow_html=True
)

# -------------------- Technical Indicators --------------------
data_ind = compute_technical_indicators(data)

# Tabs for layout
tab_predict, tab_trend, tab_future, tab_data, tab_about = st.tabs(
    ["🔮 Prediction", "📈 Trend Charts", "📅 Future Forecast", "📚 Raw Data", "ℹ️ About / References"]
)

# -------------------- Prepare train/test and scaling --------------------
with st.spinner("Preparing data and running model..."):
    # split
    data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    past_100_days = data_train.tail(100)
    data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)

    # Fit scaler on training-like data for consistent scaling
    scaler.fit(data_train.values.reshape(-1,1))
    data_test_scaled = scaler.transform(data_test_combined.values.reshape(-1,1))

    # Create sequences for evaluation plotting
    X_test = []
    y_test = []
    for i in range(100, data_test_scaled.shape[0]):
        X_test.append(data_test_scaled[i-100:i])
        y_test.append(data_test_scaled[i, 0])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Predict (may be empty if not enough data)
    if X_test.size == 0:
        st.warning("Not enough data to form test sequences (need at least 100 training days + some test days). Showing what we have.")
        predictions = np.array([])
    else:
        predictions_scaled = model.predict(X_test)
        predictions = inverse_scale(scaler, predictions_scaled)
        y_true = inverse_scale(scaler, y_test.reshape(-1,1))

# -------------------- Next day and multi-day forecast --------------------
with st.spinner("Generating next-day and multi-day forecasts..."):
    # Next day (single-step) using last 100 actual Close values
    last_100 = data['Close'][-100:].values.reshape(-1,1)
    # Use scaler fitted earlier (on training data)
    last_100_scaled = scaler.transform(last_100).reshape(1,100,1)
    next_day_scaled = model.predict(last_100_scaled)
    next_day_price = inverse_scale(scaler, next_day_scaled)[0,0]

    # Multi-day iterative forecast (7 days default)
    future_days = 7
    current_input = last_100_scaled.copy()
    future_preds_scaled = []
    for _ in range(future_days):
        p = model.predict(current_input)[0,0]
        future_preds_scaled.append(p)
        # append and slide window
        current_input = np.append(current_input[:,1:,:], [[[p]]], axis=1)
    future_prices = inverse_scale(scaler, np.array(future_preds_scaled).reshape(-1,1)).flatten()

    # Build future dates skipping weekends
    future_dates = []
    next_dt = data.index[-1].date()
    i = 0
    while len(future_dates) < future_days:
        next_dt = next_dt + pd.Timedelta(days=1)
        if next_dt.weekday() < 5:
            future_dates.append(next_dt)
        i += 1

# -------------------- Tab: Prediction --------------------
with tab_predict:
    st.subheader("🔮 Next Day Prediction")
    st.markdown(
        f"""
        <div style='background:#f6fbff;padding:18px;border-radius:12px;text-align:center'>
          <h3 style='color:#1f77b4;margin:6px 0'>Predicted Closing Price for <b>{future_dates[0].strftime('%b %d, %Y')}</b></h3>
          <h1 style='color:#2ecc71;margin:6px 0'>{currency_symbol}{next_day_price:,.2f}</h1>
          <p style='color:#555'>Based on last available data up to <b>{last_date}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Model performance metrics (if predictions were computed)
    if X_test.size != 0:
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:,.2f}")
        col2.metric("RMSE", f"{rmse:,.2f}")
        col3.metric("MSE", f"{mse:,.2f}")
        col4.metric("R²", f"{r2:.4f}")

        st.markdown("**Directional accuracy (predicted direction vs actual direction)**")
        # directional accuracy
        pred_dir = np.sign(np.diff(predictions.flatten()))
        true_dir = np.sign(np.diff(y_true.flatten()))
        dir_acc = (pred_dir == true_dir).sum() / len(true_dir) if len(true_dir) > 0 else np.nan
        st.write(f"Directional accuracy: **{dir_acc*100:.2f}%**")

    # CSV download for single-day + multi-day
    df_forecast = pd.DataFrame({
        "Date": [future_dates[0]] + future_dates,
        "Predicted_Price": [next_day_price] + list(future_prices)
    })
    csv = df_forecast.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download forecast CSV", csv, file_name=f"{stock_symbol}_forecast.csv", mime='text/csv')

# -------------------- Tab: Trend Charts --------------------
with tab_trend:
    st.subheader("📈 Price vs Moving Averages & Indicators")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.index, data['Close'], label='Close', color='#2ecc71')
    ax.plot(data.index, data_ind['MA50'], label='MA50', color='#e74c3c', linewidth=1)
    ax.plot(data.index, data_ind['MA100'], label='MA100', color='#2980b9', linewidth=1)
    ax.plot(data.index, data_ind['MA200'], label='MA200', color='#8e44ad', linewidth=1)
    ax.set_title(f"{selected_company} Price vs MA50/100/200")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Price ({currency_name})")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔧 Technical Indicators: RSI & MACD")
    fig2, (axr, axm) = plt.subplots(2,1, figsize=(12,8), gridspec_kw={'height_ratios':[1,1]})
    axr.plot(data_ind.index, data_ind['RSI'], color='purple', label='RSI')
    axr.axhline(70, color='r', linestyle='--', alpha=0.6)
    axr.axhline(30, color='g', linestyle='--', alpha=0.6)
    axr.set_title('RSI (14)')
    axr.legend()

    axm.plot(data_ind.index, data_ind['MACD'], color='blue', label='MACD')
    axm.plot(data_ind.index, data_ind['Signal'], color='orange', label='Signal')
    axm.set_title('MACD')
    axm.legend()
    st.pyplot(fig2)

    st.subheader("🔍 Model Predictions vs Actual (Test Set)")
    if X_test.size != 0:
        fig3, ax3 = plt.subplots(figsize=(12,5))
        ax3.plot(y_true, label='Actual', color='green')
        ax3.plot(predictions, label='Predicted', color='red')
        ax3.set_title("Model - Predicted vs Actual (test window)")
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.info("Not enough data to plot model vs actual (need at least 100 days + test samples).")

# -------------------- Tab: Future Forecast --------------------
with tab_future:
    st.subheader(f"📅 {future_days}-Day Forecast Table & Chart")
    future_df = pd.DataFrame({
        "Date": future_dates,
        f"Predicted Price ({currency_name})": [float(round(x,2)) for x in future_prices]
    })
    st.table(future_df)

    figf, axf = plt.subplots(figsize=(10,4))
    axf.plot(future_dates, future_prices, marker='o', linestyle='-', color='#1f77b4')
    axf.set_title(f"{selected_company} {future_days}-Day Forecast")
    axf.set_xlabel("Date")
    axf.set_ylabel(f"Price ({currency_name})")
    st.pyplot(figf)

# -------------------- Tab: Raw Data --------------------
with tab_data:
    st.subheader("📚 Raw Historical Data (last 200 rows)")
    st.dataframe(data.tail(200))

# -------------------- Tab: About / References --------------------
with tab_about:
    st.subheader("ℹ️ Project Info & References")
    st.markdown("""
    **Model:** LSTM (Keras/TensorFlow) trained on historical closing prices (sequence length = 100).  
    **Data source:** Yahoo Finance via `yfinance`.  
    **Indicators:** MA50 / MA100 / MA200, RSI, MACD.  
    **Forecast methods:** One-step LSTM with iterative multi-day forecasting.  

    **References :**
    - Fischer, T. & Krauss, C., “Deep learning with LSTM networks for financial market predictions,” *Eur. J. Oper. Res.*, 2018.  
    - Nelson, D. M. Q., Pereira, A. C. M., & de Oliveira, R. A., “Stock market's next-day price direction prediction using LSTM,” *Expert Systems with Applications*, 2017.  
    - Patel, J., Shah, S., Thakkar, P., & Kotecha, K., “Predicting stock and stock price index movement using hybrid ML methods,” *Expert Systems with Applications*, 2015.
    """)

    st.markdown("**Developer:** Nitesh Kumar — Final Year Project / Demo Dashboard")
    st.markdown("---")
    st.markdown("**Optional features**: News sentiment and SHAP explainability are available if configured (NewsAPI key & shap installed).")

# -------------------- Optional: News Sentiment --------------------
if use_news and NEWSAPI_AVAILABLE and (news_api_key or os.getenv("NEWSAPI_KEY")):
    with st.expander("📰 News Sentiment (optional)"):
        key = news_api_key if news_api_key else os.getenv("NEWSAPI_KEY")
        try:
            newsapi = NewsApiClient(api_key=key)
            q = selected_company.split('(')[0].strip()
            res = newsapi.get_everything(q=q, language='en', page_size=10)
            headlines = [a['title'] for a in res.get('articles', [])]
            if headlines:
                sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
                avg_sent = np.mean(sentiments)
                if avg_sent > 0.05:
                    st.success(f"Overall News Sentiment: Positive ({avg_sent:.2f})")
                elif avg_sent < -0.05:
                    st.error(f"Overall News Sentiment: Negative ({avg_sent:.2f})")
                else:
                    st.info(f"Overall News Sentiment: Neutral ({avg_sent:.2f})")
                st.write("Top headlines:")
                for h in headlines:
                    st.write("• " + h)
            else:
                st.info("No recent headlines found.")
        except Exception as e:
            st.error("News / sentiment fetch error: " + str(e))

# -------------------- Optional: SHAP Explainability --------------------
# -------------------- Optional: SHAP Explainability --------------------
if SHAP_AVAILABLE:
    with st.expander("🧠 SHAP Explainability (optional)"):
        st.write("Explaining model predictions using SHAP values (for transparency).")

        if X_test.size == 0:
            st.info("📄 Not enough test samples to compute SHAP values.")
        else:
            try:
                explainer = shap.Explainer(model, X_test[:100])
                shap_vals = explainer(X_test[-1:])

                # Waterfall plot
                st.write("Feature contribution for the last prediction:")
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(shap_vals[0], show=False)
                st.pyplot(fig)

                # Bar summary (for last 100 samples)
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                shap.summary_plot(shap_vals, X_test[-100:], plot_type="bar", show=False)
                st.pyplot(fig2)

            except Exception as e:
                st.info("ℹ️ SHAP explanation unavailable for this model type. (No worries, model still works fine.)")


# -------------------- Market Overview --------------------
with st.sidebar.expander("🌍 Market Overview (Quick)"):
    st.write("📊 Live index snapshots (updated every few minutes)")

    indices = {
        "NIFTY 50 🇮🇳": "^NSEI",
        "SENSEX 🇮🇳": "^BSESN",
        "NASDAQ 🇺🇸": "^IXIC",
        "S&P 500 🇺🇸": "^GSPC"
    }

    for name, sym in indices.items():
        try:
            info = yf.Ticker(sym)
            hist = info.history(period="1mo", interval="1d", auto_adjust=True)

            if hist.empty:
                st.info(f"ℹ️ {name} data currently unavailable.")
                continue

            latest_close = hist['Close'][-1]
            prev_close = hist['Close'][-2] if len(hist) > 1 else latest_close
            delta = latest_close - prev_close
            pct_change = (delta / prev_close) * 100 if prev_close != 0 else 0

            # Choose color for delta
            delta_sign = "🔺" if delta > 0 else "🔻" if delta < 0 else "⚪"
            color = "green" if delta > 0 else "red" if delta < 0 else "gray"

            st.metric(
                label=name,
                value=f"{latest_close:,.2f}",
                delta=f"{delta_sign} {delta:,.2f} ({pct_change:.2f}%)"
            )

        except Exception as e:
            st.info(f"⚪ {name}: temporarily unavailable.")


# -------------------- Footer --------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:gray'>Made with ❤️ by Nitesh Kumar • Use responsibly — for educational/demo purposes only</div>", unsafe_allow_html=True)
