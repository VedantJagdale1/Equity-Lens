from hashlib import sha256
from supabase import create_client, Client
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit.runtime.scriptrunner import add_script_run_ctx
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from google import genai

import requests
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential


st.set_page_config(page_title="EquityLens", page_icon="logo.png",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    :root{
      --bg1:#071427;--bg2:#0f2a3a;--bg3:#133a4f;
      --bull:#00c853;--bear:#ff5252;--accent:#ffd166;
      --muted:rgba(255,255,255,0.94);--card:rgba(255,255,255,0.03);
    }
    html,body,.streamlit-container{background:linear-gradient(90deg,var(--bg1),var(--bg2),var(--bg3))}
    .stApp{color:var(--muted);min-height:100vh;font-family:Roboto,Segoe UI,system-ui,-apple-system,"Helvetica Neue",Arial}
    .header-wrap{display:flex;align-items:center;justify-content:center;gap:28px;padding:8px 0}
    .title{font-size:44px;font-weight:800;letter-spacing:-0.6px;margin:0;text-align:center}
    .subtitle{font-size:13px;color:rgba(255,255,255,0.72);text-align:center;margin-top:2px}
    .option-menu .nav-link{border-radius:12px;padding:8px 14px;margin:0 6px;color:rgba(255,255,255,0.95)!important;font-weight:700}
    .option-menu .nav-link:hover{background:rgba(255,255,255,0.05)}
    .option-menu .nav-link.active{background:linear-gradient(90deg,rgba(0,200,83,0.12),rgba(255,209,102,0.04));border:1px solid rgba(255,209,102,0.12);color:#fff!important}
    .kpi-row{display:flex;gap:18px;justify-content:center;align-items:stretch}
    .kpi-card{background:var(--card);padding:14px;border-radius:12px;min-width:170px;flex:1;text-align:center;box-shadow:0 6px 20px rgba(0,0,0,0.35)}
    .kpi-label{font-size:12px;color:rgba(255,255,255,0.7)} .kpi-value{font-size:20px;font-weight:800;margin-top:6px}
    @media(max-width:900px){.title{font-size:28px}.kpi-row{flex-direction:column;gap:10px}.option-menu .nav-link{padding:8px 10px;font-size:13px}}
    .main .block-container{padding:16px 16px 32px}
    </style>

    <div class="header-wrap">
      <div style="display:flex;flex-direction:column;align-items:center">
        <div class="title">EquityLens</div>
        <div class="subtitle">Stock-centric insights · Portfolio · Market news</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
selected = option_menu(
    menu_title=None,
    options=["Explore", "Market Status", "Portfolio Analysis", "News"],
    icons=["house", "graph-up-arrow", "bar-chart-line", "newspaper"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"padding": "6px 6px", "width": "100%"}, "nav-link": {
        "font-size": "14px"}, "icon": {"font-size": "16px", "margin": "0 6px 0 0"}},
)


@st.cache_data
def load_stock_list():
    df = pd.read_csv("stocks.csv")
    return df["symbol"].tolist()


stock_symbols = load_stock_list()

if selected == "Explore":

    st.markdown("---")

    st.markdown("<h2 style='text-align: center;'>Stock Fundamental & Technical values</h2>",
                unsafe_allow_html=True)

    def get_symbol(symbol):
        return f"{symbol}"

    def format_large_number(number):
        if number >= 1e9:
            return f"{number/1e9:.2f} b"
        elif number >= 1e6:
            return f"{number/1e6:.2f} m"
        else:
            return f"{number:,.0f}"

    def analyze_stock(stock_symbol):
        try:
            stock = yf.Ticker(get_symbol(stock_symbol))

            daily_data = stock.history(period="1y", interval="1d")
            info = stock.info

            st.markdown("---")

            if not daily_data.empty:

                st.success(
                    f"📊 {stock_symbol} - {info.get('longName', stock_symbol)} Analysis report")

                col1, col2, col3, col4 = st.columns(4)
                last_price = daily_data['Close'][-1]
                previous_price = daily_data['Close'][-2]
                change = ((last_price - previous_price) / previous_price) * 100

                with col1:
                    st.metric("LAST CLOSE",
                              f"{last_price:.2f}", f"{change:.2f}%")
                with col2:
                    if 'marketCap' in info:
                        st.metric("MARKET CAP", format_large_number(
                            info['marketCap']))
                with col3:
                    if 'volume' in info:
                        st.metric("VOLUME", format_large_number(
                            daily_data['Volume'][-1]))
                with col4:
                    if 'fiftyTwoWeekHigh' in info:
                        st.metric("52 WEEK HIGH",
                                  f"{info['fiftyTwoWeekHigh']:.2f}")

                st.markdown("---")

                st.markdown("### FUNDAMENTAL ANALYSIS METRICS")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(
                    4)

                with metrics_col1:
                    if 'trailingPE' in info:
                        st.metric("Trailing PE", f"{info['trailingPE']:.2f}")
                    if 'priceToBook' in info:
                        st.metric("Price to Book Ratio",
                                  f"{info['priceToBook']:.2f}")
                    if 'profitMargins' in info:
                        st.metric("Profit Margin",
                                  f"{info['profitMargins']*100:.2f}%")

                with metrics_col2:
                    if 'forwardPE' in info:
                        st.metric("Forward PE", f"{info['forwardPE']:.2f}")
                    if 'enterpriseToRevenue' in info:
                        st.metric("EV/R", f"{info['enterpriseToRevenue']:.2f}")
                    if 'returnOnEquity' in info:
                        st.metric("Return on Equity",
                                  f"{info['returnOnEquity']*100:.2f}%")

                with metrics_col3:
                    if 'enterpriseToEbitda' in info:
                        st.metric(
                            "EV/EBITDA", f"{info['enterpriseToEbitda']:.2f}")
                    if 'debtToEquity' in info:
                        st.metric("D/E", f"{info['debtToEquity']:.2f}")
                    if 'returnOnAssets' in info:
                        st.metric("ROA", f"{info['returnOnAssets']*100:.2f}%")

                with metrics_col4:
                    if 'dividendYield' in info and info['dividendYield'] is not None:
                        st.metric("Dividend Yield",
                                  f"{info['dividendYield']*100:.2f}%")
                    if 'payoutRatio' in info and info['payoutRatio'] is not None:
                        st.metric("Payout Ratio",
                                  f"{info['payoutRatio']*100:.2f}%")
                    if 'beta' in info:
                        st.metric("Beta", f"{info['beta']:.2f}")

                st.markdown("---")

                st.markdown("### TECHNICAL INDICATORS")

                daily_data['MA50'] = daily_data['Close'].rolling(
                    window=50).mean()
                daily_data['MA200'] = daily_data['Close'].rolling(
                    window=200).mean()
                daily_data['RSI'] = calculate_rsi(daily_data['Close'])

                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

                with tech_col1:
                    st.metric("50 Day Average",
                              f"{daily_data['MA50'][-1]:.2f}")
                with tech_col2:
                    st.metric("200 Day Average",
                              f"{daily_data['MA200'][-1]:.2f}")
                with tech_col3:
                    st.metric("RSI (14)", f"{daily_data['RSI'][-1]:.2f}")
                with tech_col4:
                    volatility = daily_data['Close'].pct_change(
                    ).std() * (252 ** 0.5) * 100
                    st.metric("Annual Volatility", f"{volatility:.2f}%")

                st.markdown("---")

                st.markdown("### PRICE AND VOLUME CHARTS")

                time_periods = {
                    "1 Day": ("90d", "1d"),
                    "1 Week": ("180d", "1d"),
                    "1 Month": ("730d", "1d"),
                    "1 Year": ("max", "1d")
                }
                selected_period = st.selectbox(
                    "Candle",
                    options=list(time_periods.keys()),
                    index=1,
                    key="chart_period"
                )

                period, interval = time_periods[selected_period]
                try:
                    chart_data = stock.history(
                        period=period, interval=interval)

                    if not chart_data.empty:

                        if selected_period == "4 hour":
                            chart_data = chart_data.resample('4H').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        elif selected_period == "1 week":
                            chart_data = chart_data.resample('W').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        elif selected_period == "1 month":
                            chart_data = chart_data.resample('M').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        elif selected_period == "1 year":
                            chart_data = chart_data.resample('Y').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()

                        if len(chart_data) > 50:
                            chart_data['MA50'] = chart_data['Close'].rolling(
                                window=50).mean()
                        if len(chart_data) > 200:
                            chart_data['MA200'] = chart_data['Close'].rolling(
                                window=200).mean()

                        fig = create_candlestick_chart(
                            chart_data, stock_symbol, selected_period)
                        st.plotly_chart(fig, use_container_width=True)

                        fig_volume = create_volume_chart(
                            chart_data, stock_symbol, selected_period)
                        st.plotly_chart(fig_volume, use_container_width=True)
                    else:
                        st.warning(
                            f"No data found for {selected_period}. Please try again.")
                except Exception as e:
                    st.warning(
                        f" Data could not be retrieved for {selected_period}. Please select another time period.")

                if 'longBusinessSummary' in info:
                    with st.expander("About Company", expanded=True):
                        try:

                            st.write(info['longBusinessSummary'])
                        except:
                            st.write("Information Not Found.")

                with st.expander("Technical Analysis Summary", expanded=True):
                    trend = "Uptrend" if daily_data['MA50'][-1] > daily_data['MA200'][-1] else "Downtrend"
                    st.write(
                        f"{'⬆️' if daily_data['MA50'][-1] > daily_data['MA200'][-1] else '⬇️'} **Trend Status:** {trend}")

                    if 'trailingPE' in info:
                        fk_status = "Low (Attractive)" if info[
                            'trailingPE'] < 10 else "High (Expensive)" if info['trailingPE'] > 20 else "Normal"
                        st.write(
                            f"{'✅' if info['trailingPE'] < 10 else '⚠️' if info['trailingPE'] > 20 else 'ℹ️'} **F/K Status:** {fk_status}")

                    rsi = daily_data['RSI'][-1]
                    rsi_status = "Oversold(Buying Opportunity)" if rsi < 30 else "Overbought(Selling Opportunity)" if rsi > 70 else "Normal"
                    st.write(
                        f"{'✅' if 40 < rsi < 60 else '⚠️'} **RSI Status:** {rsi_status}")

                    volume_change = ((daily_data['Volume'][-5:].mean() - daily_data['Volume']
                                     [-10:-5].mean()) / daily_data['Volume'][-10:-5].mean()) * 100
                    st.write(
                        f"{'✅' if volume_change > 0 else 'ℹ️'} **Volume Change (5 Days):** {volume_change:.2f}%")

                with st.expander("Raw Data", expanded=True):
                    show_dataframe(daily_data)

            else:
                st.error("❌ Data not found. please enter a valid stock symbol")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Please enter valid stock code")

        st.markdown("---")

    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_figure_layout(title):
        return dict(
            title=title,
            template="plotly",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=1000,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )

    def create_candlestick_chart(hist, stock_symbol, period_text):
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="OHLC"
        ))

        if 'MA50' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA50'],
                name='50 Day Average.',
                line=dict(width=2)
            ))

        if 'MA200' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA200'],
                name='200 Day Average.',
                line=dict(width=2)
            ))

        layout = create_figure_layout(
            f"{stock_symbol} {period_text} Price Chart")
        layout.update(
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day",
                             stepmode="backward"),
                        dict(count=7, label="1W", step="day",
                             stepmode="backward"),
                        dict(count=1, label="1M", step="month",
                             stepmode="backward"),
                        dict(count=3, label="3M", step="month",
                             stepmode="backward"),
                        dict(count=6, label="6M", step="month",
                             stepmode="backward"),
                        dict(count=1, label="1Y", step="year",
                             stepmode="backward"),
                        dict(step="all", label="all")
                    ])
                )
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                title=dict(
                    text="Price",
                    standoff=10
                )
            ),
            dragmode='zoom',
            showlegend=True,
            hovermode='x unified'
        )
        fig.update_layout(layout)

        return fig

    def create_volume_chart(hist, stock_symbol, period_text):
        fig_volume = go.Figure()

        fig_volume.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name="Volume",
            marker_color='#001A6E'
        ))

        layout = create_figure_layout(
            f"{stock_symbol} {period_text} Trading Volume")
        layout.update(
            xaxis=dict(
                type='date',
                autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day",
                             stepmode="backward"),
                        dict(count=7, label="1W", step="day",
                             stepmode="backward"),
                        dict(count=1, label="1M", step="month",
                             stepmode="backward"),
                        dict(count=3, label="3M", step="month",
                             stepmode="backward"),
                        dict(count=6, label="6M", step="month",
                             stepmode="backward"),
                        dict(count=1, label="1Y", step="year",
                             stepmode="backward"),
                        dict(step="all", label="all")
                    ]),

                )
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                title=dict(
                    text="Transaction Volume",
                    standoff=10
                )
            ),
            dragmode='zoom',
            showlegend=True
        )
        fig_volume.update_layout(layout)

        return fig_volume

    def show_dataframe(df):
        st.dataframe(
            df,
            height=300,
            use_container_width=True,
        )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        stock_symbol = st.selectbox("Stock analysis and screening tool", [
                                    ""] + stock_symbols, help="Search for a company").upper()

    if stock_symbol:
        analyze_stock(stock_symbol)
    user_input = stock_symbol

    @st.cache_data(ttl=60*60)
    def fetch_stock_data(symbol: str, period: str = "5y", interval: str = "1d"):
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
        return df

    # fetch historical once
    try:
        chart_df = fetch_stock_data(stock_symbol, period="5y", interval="1d")
    except Exception as e:
        chart_df = pd.DataFrame()

    # safe check and plotting
    else:
        st.markdown("### 📈 Historical Close Price (raw)")
        st.line_chart(chart_df['Close'])

        # add moving averages if enough data
        if len(chart_df) > 50:
            chart_df['MA50'] = chart_df['Close'].rolling(50).mean()
        if len(chart_df) > 200:
            chart_df['MA200'] = chart_df['Close'].rolling(200).mean()

        # try to show candlestick (use your helper function)
        try:
            fig = create_candlestick_chart(
                chart_df, stock_symbol, "Historical")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # fallback simple OHLC chart if helper fails
            fig = go.Figure(data=[go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name="OHLC"
            )])
            fig.update_layout(
                title=f"{stock_symbol} OHLC", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("🧠 Deep Learning Price Prediction")
            

    def create_sequences(data, window=10):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    @st.cache_resource
    def train_dl_model(symbol, window=10):
        df = yf.Ticker(symbol).history(period="5y", interval="1d")

        if df.empty or len(df) < window + 10:
            return None, None, None

        close_prices = df["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        close_scaled = scaler.fit_transform(close_prices)

        X, y = create_sequences(close_scaled, window)
        X = X.reshape(X.shape[0], X.shape[1])  # Dense NN input

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            Dense(64, activation="relu", input_shape=(X.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            verbose=0
        )

        return model, scaler, close_scaled
    def predict_next_n_days(model, scaler, close_scaled, window=10, days=22):
        """
        Recursive multi-step forecasting
        """
        preds_scaled = []
        current_window = close_scaled[-window:].reshape(1, -1)

        for _ in range(days):
            next_pred = model.predict(current_window, verbose=0)[0][0]
            preds_scaled.append(next_pred)

            # slide window
            current_window = np.append(
                current_window[:, 1:], [[next_pred]], axis=1
            )

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).flatten()

        return preds

    

    if stock_symbol:
        with st.spinner("Training Deep Learning model..."):
            model, scaler, close_scaled = train_dl_model(stock_symbol)

        if model is None:
            st.warning("Not enough data for DL prediction.")
        else:
            window = 10
            last_window = close_scaled[-window:]
            last_window = last_window.reshape(1, -1)

            pred_scaled = model.predict(last_window, verbose=0)
            predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

            last_price = float(
                yf.Ticker(stock_symbol).history(period="2d")["Close"].iloc[-1]
            )

            diff = predicted_price - last_price
            pct_change = (diff / last_price) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Last Close Price", f"₹{last_price:.2f}")

            with col2:
                st.metric("Predicted Next Close", f"₹{predicted_price:.2f}")

            with col3:
                st.metric(
                    "Expected Change",
                    f"{pct_change:.2f}%",
                    delta=f"₹{diff:.2f}"
                )
        st.markdown("---")
        st.subheader("📈 1-Month Price Prediction (Deep Learning)")
        future_days = 22  # ~1 trading month

        future_prices = predict_next_n_days(
            model=model,
            scaler=scaler,
            close_scaled=close_scaled,
            window=10,
            days=future_days
        )
        # Historical prices (last 3 months for clarity)
        hist_df = yf.Ticker(stock_symbol).history(period="3mo")
        hist_prices = hist_df["Close"]

        # Future dates (business days)
        last_date = hist_prices.index[-1]
        future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days
        )   

        pred_df = pd.Series(future_prices, index=future_dates)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_prices.index,
            y=hist_prices.values,
            mode="lines",
            name="Historical Price"
        ))

        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df.values,
            mode="lines",
            name="Predicted (Next 1 Month)",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            title=f"{stock_symbol} – 1 Month Price Prediction (DL)",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("""
            **Model:** Deep Learning (Dense Neural Network)  
            **Input:** Last 10 days closing prices  
            **Output:** Next day closing price  
            **Loss Function:** Mean Squared Error (MSE)
            """)


if selected == "Market Status":
    st.markdown("""
        <style>
        .main .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }
        .stApp {
            margin: 0 !important;
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    def fetch_symbol_data(symbol):
        try:
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            info = stock.info

            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown Sector')
            market_cap = info.get('marketCap', 0)
            market_cap_cr = round(max(market_cap / 1e7, 0.01), 2)
            current_price = info.get(
                'currentPrice', info.get('previousClose', 0))
            previous_close = info.get('previousClose', 0)
            p_change = ((current_price - previous_close) /
                        previous_close * 100) if previous_close else 0

            return {
                "Symbol": symbol,
                "Name": company_name,
                "Sector": sector,
                "MarketCap": market_cap_cr,
                "PriceChange": round(p_change, 2),
                "LastPrice": current_price
            }
        except Exception as e:
            st.warning(f"Error fetching {symbol} from Yahoo Finance: {e}")
            return create_error_data(symbol)

    def create_error_data(symbol):
        return {
            "Symbol": symbol,
            "Name": symbol,
            "Sector": "Error",
            "MarketCap": 0.01,
            "PriceChange": 0,
            "LastPrice": 0
        }

    @st.cache_data(ttl=300)
    def fetch_yfinance_data(symbols):
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for symbol in symbols:
                future = executor.submit(fetch_symbol_data, symbol)
                add_script_run_ctx(future) # <--- This attaches the Streamlit context
                futures.append(future)
        
            for future in futures:
                results.append(future.result())
              
        df = pd.DataFrame(results)

        fetched_symbols = set(df['Symbol'])
        input_symbols = set(symbols)
        missing = input_symbols - fetched_symbols
        if missing:
            st.warning(f"Missing data for: {', '.join(missing)}")

        return df

    def main():
        st.markdown("---")

        st.markdown(
            "<h2 style='text-align: center;'>Markets at a glance (NIFTY-50 heatmap)</h2>", unsafe_allow_html=True)

        default_symbols = ['TATASTEEL', 'NTPC', 'WIPRO', 'ITC', 'RELIANCE', 'SHRIRAMFIN', 'ONGC',
                           'COALINDIA', 'BHARTIARTL', 'INDUSINDBK', 'HINDALCO', 'KOTAKBANK',
                           'TATACONSUM', 'HDFCLIFE', 'LT', 'TCS', 'CIPLA', 'TRENT', 'ADANIENT',
                           'BAJAJFINSV', 'BRITANNIA', 'BAJFINANCE', 'TITAN', 'NESTLEIND',
                           'ULTRACEMCO', 'HEROMOTOCO', 'BAJAJ-AUTO', 'MARUTI', 'APOLLOHOSP',
                           'HDFCBANK', 'ICICIBANK', 'INFY', 'SBIN', 'HINDUNILVR', 'HCLTECH',
                           'SUNPHARMA', 'M&M', 'AXISBANK', 'POWERGRID', 'TATAMOTORS',
                           'ADANIPORTS', 'JSWSTEEL', 'ASIANPAINT', 'BEL', 'TECHM', 'GRASIM',
                           'EICHERMOT', 'BPCL', 'DRREDDY']

        with st.spinner("Loading heatmap..."):
            df = fetch_yfinance_data(tuple(default_symbols))
            valid_df = df[df['Sector'] != 'Error']
            valid_df = valid_df[valid_df['MarketCap'] > 0]

            st.write(f"")

        with st.sidebar:
            st.header("Filters for NIFTY-50 Heatmap")
            sector_options = valid_df["Sector"].unique()
            selected_sectors = st.multiselect(
                "Select Sectors",
                options=sector_options,
                default=sector_options,
                key="nifty_filter_v2"
            )

        filtered_df = valid_df[valid_df["Sector"].isin(selected_sectors)]

        if not filtered_df.empty:
            fig = px.treemap(
                filtered_df,
                path=[px.Constant("NSE Stocks"), 'Sector', 'Name'],
                values='MarketCap',
                color='PriceChange',
                color_continuous_scale=[[0, "red"], [
                    0.4, "red"], [.6, "green"], [1, "green"]],
                color_continuous_midpoint=0,
                hover_data=['MarketCap', 'PriceChange', 'LastPrice'],
                width=None,
                height=800
            )

            fig.update_traces(
                texttemplate="%{label}<br>M Cap: ₹%{customdata[0]:,.1f} Cr<br>Change: %{customdata[1]:+.2f}%<br>Price: ₹%{customdata[2]:,.1f}",
                hovertemplate="%{label}<br>Market Cap: ₹%{customdata[0]:,.1f} Cr<br>Price Change: %{customdata[1]:+.2f}%<br>Last Price: ₹%{customdata[2]:,.1f}<extra></extra>",
                textfont_size=18,
                hoverlabel=dict(font_size=18),
                marker=dict(line=dict(color="black", width=1))
            )

            fig.update_layout(
                margin=dict(t=50, l=0, r=0, b=0),
                coloraxis_colorbar=dict(
                    title="Price Change (%)",
                    tickprefix="",
                    ticksuffix="%"
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                autosize=True
            )

            with st.container():
                st.plotly_chart(fig, use_container_width=True,
                                config={'responsive': True})

        else:
            st.warning("No valid data available!")

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Sensex at a Glance (Heatmap)</h2>",
                unsafe_allow_html=True)

    default_symbols = [
        'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV',
        'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'HCLTECH',
        'HDFCBANK', 'HDFC', 'ICICIBANK', 'INDUSINDBK', 'ITC', 'KOTAKBANK',
        'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID',
        'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMOTORS', 'TATASTEEL', 'TECHM',
        'ULTRACEMCO'
    ]

    with st.spinner("Loading heatmap..."):
        df = fetch_yfinance_data(tuple(default_symbols))
        valid_df = df[df['Sector'] != 'Error']
        valid_df = valid_df[valid_df['MarketCap'] > 0]

    with st.sidebar:
        st.header("Filters for SENSEX Heatmap")
        sector_options = valid_df["Sector"].unique()
        selected_sectors = st.multiselect(
            "Select Sectors",
            options=sector_options,
            default=sector_options,
            key="sensex_filter_v2"
        )

    filtered_df = valid_df[valid_df["Sector"].isin(selected_sectors)]

    if not filtered_df.empty:
        fig = px.treemap(
            filtered_df,
            path=[px.Constant("Sensex Stocks"), 'Sector', 'Name'],
            values='MarketCap',
            color='PriceChange',
            color_continuous_scale=[[0, "red"], [
                0.4, "red"], [0.6, "green"], [1, "green"]],
            color_continuous_midpoint=0,
            hover_data=['MarketCap', 'PriceChange', 'LastPrice'],
            width=None,
            height=800
        )        

        fig.update_traces(
            texttemplate="%{label}<br>M Cap: ₹%{customdata[0]:,.1f} Cr<br>Change: %{customdata[1]:+.2f}%<br>Price: ₹%{customdata[2]:,.1f}",
            hovertemplate="%{label}<br>Market Cap: ₹%{customdata[0]:,.1f} Cr<br>Price Change: %{customdata[1]:+.2f}%<br>Last Price: ₹%{customdata[2]:,.1f}<extra></extra>",
            textfont_size=18,
            hoverlabel=dict(font_size=18),
            marker=dict(line=dict(color="black", width=1))
        )

        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            coloraxis_colorbar=dict(
                title="Price Change (%)",
                tickprefix="",
                ticksuffix="%"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True
        )

        with st.container():
            st.plotly_chart(fig, use_container_width=True,
                            config={'responsive': True})
    else:
        st.warning("No valid data available!")

    if __name__ == "__main__":
        main()

if selected == "Portfolio Analysis":
    GEMINI_API_KEY = "AIzaSyCYfiS4opMkULGVADhWIR4s3fIU4Uh06qU"
    client = genai.Client(api_key=GEMINI_API_KEY)

    SUPABASE_URL = "https://nucydrfkpbwqmiovcwdj.supabase.co"
    # 🔹 Use anon key (public) or service role key (admin access)
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im51Y3lkcmZrcGJ3cW1pb3Zjd2RqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE1NDQzNzUsImV4cCI6MjA3NzEyMDM3NX0.ljJXFMVdrD8Fkxai93sv0Qjm43qo4wcAhNFpN8035w0"

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def init_db():
        try:
            supabase.table("users").select("username").limit(1).execute()
        except Exception:
            supabase.table("users").insert(
                {"username": "dummy", "password": "dummy"}).execute()
            supabase.table("users").delete().eq("username", "dummy").execute()

        try:
            supabase.table("portfolios").select("username").limit(1).execute()
        except Exception:
            supabase.table("portfolios").insert(
                {"username": "dummy", "ticker": "dummy", "shares": 0, "buy_price": 0}).execute()
            supabase.table("portfolios").delete().eq(
                "username", "dummy").execute()
        init_db()
    def hash_password(password):
        return sha256(password.encode()).hexdigest()

    def register_user(username, password):
        try:
            supabase.table("users").insert(
                {"username": username, "password": hash_password(password)}).execute()
            return True
        except Exception as e:
            if "duplicate key" in str(e).lower():
                return False
            raise e

    def login_user(username, password):
        response = supabase.table("users").select(
            "password").eq("username", username).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]["password"] == hash_password(password)
        return False

    def save_portfolio(username, portfolio):
        supabase.table("portfolios").delete().eq(
            "username", username).execute()
        for stock in portfolio:
            supabase.table("portfolios").insert({
                "username": username,
                "ticker": stock["Ticker"],
                "shares": stock["Shares"],
                "buy_price": stock["Buy Price"]
            }).execute()

    def load_portfolio(username):
        response = supabase.table("portfolios").select(
            "*").eq("username", username).execute()
        return [{"Ticker": row["ticker"], "Shares": row["shares"], "Buy Price": row["buy_price"]} for row in response.data]

    def get_gemini_portfolio_analysis(portfolio_df):
        prompt = f"""
        You are a financial expert AI.

        Analyze the following stock portfolio and provide:
        1. Portfolio overview
        2. Key risks
        3. Strengths and opportunities

        Portfolio Data:
    {portfolio_df.to_string(index=False)}
    """
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error fetching Gemini AI analysis: {e}"


    

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.portfolio = []

    if not st.session_state.logged_in:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.header("Login / Register")
        auth_choice = st.radio("Choose an option", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if auth_choice == "Login":
            if st.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.portfolio = load_portfolio(username)
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            if st.button("Register"):
                if register_user(username, password):
                    st.success("Registered successfully! Please login.")
                else:
                    st.error("Username already exists")
        st.markdown("</div>", unsafe_allow_html=True)
    else:

        st.markdown("""
            <div class="header">
                <h2>Welcome to Your Stock Portfolio</h2>
                <p class="header-text">Track, analyze, and stay updated with your Indian stock investments—all in one place.</p>
            </div>
            <div>Add stocks in the sidebar to see your portfolio analysis!</div>
        """, unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=["Portfolio Analysis"],
            icons=["briefcase"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"font-size": "18px", "width": "50%"},
                "nav-link": {"font-size": "18px", "padding": "4px 4px", "font-weight": "bold"},
                "icon": {"font-size": "18px"},
            }
        )

        TIMEFRAME_OPTIONS = {
            "1 Month": "30d",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "Year to Date": "ytd"
        }

        with st.sidebar:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.portfolio = []
                st.rerun()

            st.header(f"Portfolio for {st.session_state.username}")

            ticker = st.selectbox("Stock Ticker (e.g., RELIANCE.NS)", [
                                  ""] + stock_symbols, help="Select a stock symbol from the list")

            shares = st.number_input("Shares", min_value=1, value=1)
            buy_price = st.number_input(
                "Buy Price (INR)", min_value=0.0, value=0.0)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add Stock"):
                    if ticker and shares and buy_price:
                        ticker_upper = ticker.upper()
                        existing_stock = next(
                            (stock for stock in st.session_state.portfolio if stock["Ticker"] == ticker_upper), None)
                        if existing_stock:
                            old_shares = existing_stock["Shares"]
                            old_cost = old_shares * existing_stock["Buy Price"]
                            new_cost = shares * buy_price
                            total_shares = old_shares + shares
                            new_avg_buy_price = (
                                old_cost + new_cost) / total_shares
                            existing_stock["Shares"] = total_shares
                            existing_stock["Buy Price"] = new_avg_buy_price
                            st.success(f"Updated {ticker_upper}!")
                        else:
                            st.session_state.portfolio.append({
                                "Ticker": ticker_upper,
                                "Shares": shares,
                                "Buy Price": buy_price
                            })
                            st.success(f"Added {ticker_upper}!")
                        save_portfolio(st.session_state.username,
                                       st.session_state.portfolio)
            with col2:
                if st.button("Clear Portfolio", key="clear"):
                    st.session_state.portfolio = []
                    save_portfolio(st.session_state.username,
                                   st.session_state.portfolio)
                    st.success("Portfolio cleared!")

            st.markdown("---")
            st.subheader("Remove Stock")
            remove_ticker = st.selectbox("Select Stock to Remove", [
                                         ""] + [stock["Ticker"] for stock in st.session_state.portfolio], key="remove_ticker")
            remove_shares = st.number_input(
                "Shares to Remove", min_value=1, value=1, key="remove_shares")

            if st.button("Remove Stock", key="remove_stock"):
                if remove_ticker:
                    existing_stock = next(
                        (stock for stock in st.session_state.portfolio if stock["Ticker"] == remove_ticker), None)
                    if existing_stock:
                        if existing_stock["Shares"] >= remove_shares:
                            existing_stock["Shares"] -= remove_shares
                            if existing_stock["Shares"] == 0:
                                st.session_state.portfolio = [
                                    stock for stock in st.session_state.portfolio if stock["Ticker"] != remove_ticker]
                            save_portfolio(st.session_state.username,
                                           st.session_state.portfolio)
                            st.success(
                                f"Removed {remove_shares} shares of {remove_ticker}!")
                        else:
                            st.error(
                                f"Not enough shares! You only have {existing_stock['Shares']} shares of {remove_ticker}.")
                    else:
                        st.error(
                            f"Stock {remove_ticker} not found in your portfolio!")
                else:
                    st.error("Please select a stock to remove!")

            st.markdown("---")
            st.markdown("""
                ### How to Use
                - Select a stock symbol from the list.
                - Add shares and buy price.
                - Click 'Add Stock' or 'Clear Portfolio'.
            """, unsafe_allow_html=True)

        if selected == "Portfolio Analysis":
            if not st.session_state.portfolio:
                a=1
            else:
                st.markdown("---")
                st.header("Portfolio Analysis")
                selected_timeframe = st.selectbox(
                    "Select Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=0)

                nifty_data = yf.Ticker("^NSEI").history(
                    period=TIMEFRAME_OPTIONS[selected_timeframe])
                nifty_close = nifty_data["Close"]
                nifty_return = (nifty_close[-1] / nifty_close[0] - 1) * 100

                risk_free_rate = 0.06
                market_return = nifty_return / 100
                normalized_data = pd.DataFrame()
                portfolio_value_history = pd.Series(0, index=nifty_close.index)
                portfolio_data = []

                for stock in st.session_state.portfolio:
                    try:
                        ticker_obj = yf.Ticker(stock["Ticker"])
                        history_1d = ticker_obj.history(period="5d")
                        history_timeframe = ticker_obj.history(
                            period=TIMEFRAME_OPTIONS[selected_timeframe])
                        if history_1d.empty or history_timeframe.empty:
                            st.warning(
                                f"No data available for {stock['Ticker']}")
                            continue
                        current_price = history_1d["Close"].dropna().iloc[-1]
                        total_value = current_price * stock["Shares"]
                        profit_loss = (current_price -
                                       stock["Buy Price"]) * stock["Shares"]
                        info = ticker_obj.info

                        beta = info.get("beta", 0)
                        capm_expected_return = risk_free_rate + \
                            beta * (market_return - risk_free_rate)
                        div_yield = info.get(
                            "dividendYield", 0) * 100 if info.get("dividendYield") else 0
                        intraday_vol = (
                            (history_1d["High"].iloc[-1] - history_1d["Low"].iloc[-1]) / current_price) * 100
                        stock_return_timeframe = (
                            history_timeframe["Close"].iloc[-1] / history_timeframe["Close"].iloc[0] - 1) * 100
                        rel_strength = stock_return_timeframe - nifty_return
                        days_to_breakeven = abs(
                            profit_loss / (history_timeframe["Close"].diff().mean() * stock["Shares"])) if profit_loss < 0 else 0

                        sector = {"RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking",
                                  "INFY.NS": "IT", "SBIN.NS": "Banking"}.get(stock["Ticker"], "Unknown")

                        portfolio_data.append({
                            "Ticker": stock["Ticker"],
                            "Shares": stock["Shares"],
                            "Buy Price (INR)": round(stock["Buy Price"], 2),
                            "Current Price (INR)": round(current_price, 2),
                            "Total Value (INR)": round(total_value, 2),
                            "Profit/Loss (INR)": round(profit_loss, 2),
                            "Beta": round(beta, 2),
                            "CAPM Exp Return (%)": round(capm_expected_return * 100, 2),
                            "Div Yield (%)": round(div_yield, 2),
                            "Intraday Vol (%)": round(intraday_vol, 2),
                            "Rel Strength vs Nifty (%)": round(rel_strength, 2),
                            "Days to Breakeven": int(days_to_breakeven) if days_to_breakeven > 0 else "N/A",
                            "Sector": sector
                        })

                        normalized_prices = (
                            history_timeframe["Close"] / history_timeframe["Close"].iloc[0]) * 100
                        normalized_data[stock["Ticker"]] = normalized_prices
                        portfolio_value_history += history_timeframe["Close"] * \
                            stock["Shares"]

                    except Exception as e:
                        st.error(f"Error with {stock['Ticker']}: {e}")

                df = pd.DataFrame(portfolio_data)
                st.dataframe(df.style.format({
                    "Buy Price (INR)": "₹{:.2f}", "Current Price (INR)": "₹{:.2f}",
                    "Total Value (INR)": "₹{:.2f}", "Profit/Loss (INR)": "₹{:.2f}",
                    "Beta": "{:.2f}", "CAPM Exp Return (%)": "{:.2f}",
                    "Div Yield (%)": "{:.2f}", "Intraday Vol (%)": "{:.2f}",
                    "Rel Strength vs Nifty (%)": "{:.2f}", "Days to Breakeven": "{}"
                }), use_container_width=True)

                st.markdown("---")
                st.subheader(
                    "AI-Powered Portfolio Insights (Powered by Gemini)")
                gemini_analysis = get_gemini_portfolio_analysis(df)
                st.markdown(gemini_analysis)

                st.markdown("---")
                st.subheader("Portfolio Summary")
                total_investment = sum(
                    stock["Shares"] * stock["Buy Price"] for stock in st.session_state.portfolio)
                total_value_sum = df["Total Value (INR)"].sum()
                total_pl_sum = df["Profit/Loss (INR)"].sum()
                percent_pl = (total_pl_sum / total_investment) * \
                    100 if total_investment > 0 else 0
                portfolio_beta = np.average(
                    df["Beta"], weights=df["Total Value (INR)"]) if total_value_sum > 0 else 0
                portfolio_capm = np.average(
                    df["CAPM Exp Return (%)"], weights=df["Total Value (INR)"]) if total_value_sum > 0 else 0
                div_yield_contrib = sum(
                    df["Div Yield (%)"] * df["Total Value (INR)"]) / total_value_sum if total_value_sum > 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"<div class='metric-box'><b>Total Investment</b><br>₹{total_investment:,.2f}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(
                        f"<div class='metric-box'><b>Total Value</b><br>₹{total_value_sum:,.2f}</div>", unsafe_allow_html=True)
                with col3:
                    pl_class = "gain" if total_pl_sum >= 0 else "loss"
                    st.markdown(
                        f"<div class='metric-box'><b>Profit/Loss</b><br><span class='{pl_class}'>₹{total_pl_sum:,.2f} ({percent_pl:.2f}%)</span></div>", unsafe_allow_html=True)

                col4, col5, col6 = st.columns(3)
                with col4:
                    st.markdown(
                        f"<div class='metric-box'><b>Portfolio Beta</b><br>{portfolio_beta:.2f}</div>", unsafe_allow_html=True)
                with col5:
                    st.markdown(
                        f"<div class='metric-box'><b>Portfolio CAPM Return</b><br>{portfolio_capm:.2f}%</div>", unsafe_allow_html=True)
                with col6:
                    st.markdown(
                        f"<div class='metric-box'><b>Div Yield Contribution</b><br>{div_yield_contrib:.2f}%</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Portfolio vs Nifty 50 Comparison")
                portfolio_normalized = (
                    portfolio_value_history / portfolio_value_history.iloc[0]) * 100
                nifty_normalized = (nifty_close / nifty_close.iloc[0]) * 100
                comparison_data = pd.DataFrame(
                    {"Portfolio": portfolio_normalized, "Nifty 50": nifty_normalized})
                st.line_chart(comparison_data, use_container_width=True)

                portfolio_returns = portfolio_value_history.pct_change().dropna()
                nifty_returns = nifty_close.pct_change().dropna()
                portfolio_total_return = (
                    portfolio_value_history[-1] / portfolio_value_history[0] - 1) * 100
                tracking_error = (portfolio_returns -
                                  nifty_returns).std() * 100
                correlation = portfolio_returns.corr(nifty_returns)

                col1, col2, col3 = st.columns(3)
                with col1:
                    return_class = "gain" if portfolio_total_return >= nifty_return else "loss"
                    st.markdown(
                        f"<div class='metric-box'><b>Portfolio Return</b><br><span class='{return_class}'>{portfolio_total_return:.2f}%</span></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(
                        f"<div class='metric-box'><b>Tracking Error</b><br>{tracking_error:.2f}%</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(
                        f"<div class='metric-box'><b>Correlation with Nifty</b><br>{correlation:.2f}</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Normalized Price Performance")
                st.line_chart(normalized_data, use_container_width=True)

                st.markdown("---")
                st.subheader("Profit/Loss by Stock")
                st.bar_chart(df.set_index("Ticker")[
                             "Profit/Loss (INR)"], use_container_width=True)

if selected == "News":

    st.markdown("""
    <style>
    .news-header {
        text-align: center;
        margin-bottom: 25px;
    }

    .news-header h1 {
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(90deg, #00c853, #64ffda);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .news-subtitle {
        color: #9e9e9e;
        font-size: 15px;
    }

    .news-card {
        background: rgba(30, 30, 30, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        transition: all 0.25s ease-in-out;
    }

    .news-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 18px 40px rgba(0,0,0,0.55);
        border-left: 4px solid #00c853;
    }

    .news-title {
        font-size: 17px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .news-title a {
        text-decoration: none;
        color: #ffffff;
    }

    .news-title a:hover {
        color: #64ffda;
    }

    .news-meta {
        font-size: 13px;
        color: #b0b0b0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .news-chip {
        background-color: rgba(0, 200, 83, 0.15);
        color: #00c853;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }

    .news-empty {
        text-align: center;
        padding: 40px;
        color: #9e9e9e;
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="news-header">
        <h1>Market News</h1>
        <div class="news-subtitle">
            Latest Indian stock market headlines powered by Google News
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stock selector
    query = st.selectbox(
        "🔍 Select Stock",
        [""] + stock_symbols,
        help="Choose a stock to view related market news"
    )

    def fetch_google_news_rss(query):
        url = f"https://news.google.com/rss/search?q={query}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        news_list = []

        for item in root.findall(".//item")[:6]:
            title = item.find("title").text
            link = item.find("link").text
            source = item.find("source").text if item.find(
                "source") is not None else "Google News"
            news_list.append((title, link, source))

        return news_list

    if st.button("📰 Fetch Latest News"):
        if not query:
            st.warning("Please select a stock first.")
        else:
            with st.spinner("Fetching market headlines..."):
                news = fetch_google_news_rss(query)

                if news:
                    for title, link, source in news:
                        st.markdown(f"""
                        <div class="news-card">
                            <div class="news-title">
                                <a href="{link}" target="_blank">🟢 {title}</a>
                            </div>
                            <div class="news-meta">
                                <span class="news-chip">{query}</span>
                                <span>🗞 {source}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="news-empty">
                        😕 No recent news found for this stock.<br>
                        Try another company.
                    </div>
                    """, unsafe_allow_html=True)


st.markdown("<br/><div style='margin-top:10px;color:rgba(255,255,255,0.6);font-size:13px;text-align:center'>EquityLens • Built with Streamlit</div>", unsafe_allow_html=True)









