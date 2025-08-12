import streamlit as st
import yfinance as yf
import requests
from requests.exceptions import RequestException
import time
import pandas as pd
import feedparser
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import statsmodels.api as sm
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import google.generativeai as genai
from supabase import create_client, Client
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET


st.set_page_config(page_title="EquityLens", page_icon="logo.png", layout="wide",initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    /* Apply a gradient background */
    .stApp {
        background: linear-gradient(to right, #0F2027, #203A43, #2C5364);
        color: white;
    }

    /* Make header transparent */
    header {
        background: rgba(0, 0, 0, 0) !important;  /* Fully transparent */
    }

    /* Adjust header title color */
    header .stMarkdown {
        color: white !important;  /* Keep text readable */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stButton>button {
        background: rgba(0, 0, 0, 0) !important;
        color: white !important;
    }
    .stSidebar, .stApp {
        background-color: #0F2027!important; /* Change sidebar and main background */
        color:white !important;
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
        .container {
            display: flex;
            flex-wrap: wrap; /* Wrap buttons on small screens */
            justify-content: center; /* Center the buttons */
            align-items: center; /* Align vertically */
            gap: 10px; /* Space between buttons */
            margin-top: 20px; /* Add space above buttons */
        }
        .btn {
            flex: 1; /* Makes buttons take equal width */
            min-width: 150px; /* Minimum button width */
            max-width: 200px; /* Maximum button width */
            padding: 15px;
            font-size: 16px;
            text-align: center;
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: 0.3s;
        }
        .btn:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])  


with col1:
  st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap" rel="stylesheet">
    <h1 style="font-family: 'Roboto', sans-serif; font-weight: 700; color: #FFFFFF;padding-top:35px; text-align: left;font-size: 55px;">
        EquityLens
    </h1>
    """,
    unsafe_allow_html=True
    )
with col2:
    
    st.markdown('<div class="container">', unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    selected = option_menu(
    menu_title=None,
    options=["Explore","Market Status","Portfolio Analysis","News"],
    icons=["house","graph-up-arrow","bar-chart-line","newspaper"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"font-size": "18px", "width": "100%","border-radius": "initial"},
        "nav-link": {"font-size": "15px", "padding": "4px 8px", "margin": "0 4px", "font-weight": "bold"},
        "icon": {"font-size": "18px"},
    }

)

    st.markdown('</div>', unsafe_allow_html=True)
@st.cache_data
def load_stock_list():
    df = pd.read_csv("stocks.csv")  # Expects column: symbol
    return df["symbol"].tolist()  # Return list of symbols
    
stock_symbols = load_stock_list()

if selected == "Explore":
    # Home Page Content

    st.markdown("---")
    
    # Header section
    st.markdown("<h2 style='text-align: center;'>Stock Fundamental & Technical values</h2>", unsafe_allow_html=True)

    


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
            # Daily data for general metrics
            daily_data = stock.history(period="1y", interval="1d")
            info = stock.info

            st.markdown("---")
            
            if not daily_data.empty:
                # header card
                st.success(f"📊 {stock_symbol} - {info.get('longName', stock_symbol)} Analysis report")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                last_price = daily_data['Close'][-1]
                previous_price = daily_data['Close'][-2]
                change = ((last_price - previous_price) / previous_price) * 100
                
                with col1:
                    st.metric("LAST CLOSE", f"{last_price:.2f}", f"{change:.2f}%")
                with col2:
                    if 'marketCap' in info:
                        st.metric("MARKET CAP", format_large_number(info['marketCap']))
                with col3:
                    if 'volume' in info:
                        st.metric("VOLUME", format_large_number(daily_data['Volume'][-1]))
                with col4:
                    if 'fiftyTwoWeekHigh' in info:
                        st.metric("52 WEEK HIGH", f"{info['fiftyTwoWeekHigh']:.2f}")

                st.markdown("---")
                
                # FUNDAMENTAL ANALYSIS METRICS
                st.markdown("### FUNDAMENTAL ANALYSIS METRICS")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    if 'trailingPE' in info:
                        st.metric("Trailing PE", f"{info['trailingPE']:.2f}")
                    if 'priceToBook' in info:
                        st.metric("Price to Book Ratio", f"{info['priceToBook']:.2f}")
                    if 'profitMargins' in info:
                        st.metric("Profit Margin", f"{info['profitMargins']*100:.2f}%")
                
                with metrics_col2:
                    if 'forwardPE' in info:
                        st.metric("Forward PE", f"{info['forwardPE']:.2f}")
                    if 'enterpriseToRevenue' in info:
                        st.metric("EV/R", f"{info['enterpriseToRevenue']:.2f}")
                    if 'returnOnEquity' in info:
                        st.metric("Return on Equity", f"{info['returnOnEquity']*100:.2f}%")
                
                with metrics_col3:
                    if 'enterpriseToEbitda' in info:
                        st.metric("EV/EBITDA", f"{info['enterpriseToEbitda']:.2f}")
                    if 'debtToEquity' in info:
                        st.metric("D/E", f"{info['debtToEquity']:.2f}")
                    if 'returnOnAssets' in info:
                        st.metric("ROA", f"{info['returnOnAssets']*100:.2f}%")
                
                with metrics_col4:
                    if 'dividendYield' in info and info['dividendYield'] is not None:
                        st.metric("Dividend Yield", f"{info['dividendYield']*100:.2f}%")
                    if 'payoutRatio' in info and info['payoutRatio'] is not None:
                        st.metric("Payout Ratio", f"{info['payoutRatio']*100:.2f}%")
                    if 'beta' in info:
                        st.metric("Beta", f"{info['beta']:.2f}")

                st.markdown("---")
                
                # TECHNICAL INDICATORS
                st.markdown("### TECHNICAL INDICATORS")
                
                # MOVING AVERAGES
                daily_data['MA50'] = daily_data['Close'].rolling(window=50).mean()
                daily_data['MA200'] = daily_data['Close'].rolling(window=200).mean()
                daily_data['RSI'] = calculate_rsi(daily_data['Close'])
                
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                with tech_col1:
                    st.metric("50 Day Average", f"{daily_data['MA50'][-1]:.2f}")
                with tech_col2:
                    st.metric("200 Day Average", f"{daily_data['MA200'][-1]:.2f}")
                with tech_col3:
                    st.metric("RSI (14)", f"{daily_data['RSI'][-1]:.2f}")
                with tech_col4:
                    volatility = daily_data['Close'].pct_change().std() * (252 ** 0.5) * 100
                    st.metric("Annual Volatility", f"{volatility:.2f}%")

                st.markdown("---")
                
                # Graphics
                st.markdown("### PRICE AND VOLUME CHARTS")
                
                # Dropdown for timezone selection
                time_periods = {
                    "1 Day": ("90d", "1d"), 
                    "1 Week": ("180d", "1d"),
                    "1 Month": ("730d", "1d"),
                    "1 Year": ("max", "1d")
                }
                selected_period = st.selectbox(
                    "Candle",
                    options=list(time_periods.keys()),
                    index=1,  # 1 Day is selected by default
                    key="chart_period"
                )
                
                # Import data only for charts
                period, interval = time_periods[selected_period]
                try:
                    chart_data = stock.history(period=period, interval=interval)
                    
                    if not chart_data.empty:
                        # resample data according to selected time period
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

                        # Moving Averages
                        if len(chart_data) > 50:
                            chart_data['MA50'] = chart_data['Close'].rolling(window=50).mean()
                        if len(chart_data) > 200:
                            chart_data['MA200'] = chart_data['Close'].rolling(window=200).mean()
                        
                        # Candlestick chart
                        fig = create_candlestick_chart(chart_data, stock_symbol, selected_period)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume chart
                        fig_volume = create_volume_chart(chart_data, stock_symbol, selected_period)
                        st.plotly_chart(fig_volume, use_container_width=True)
                    else:
                        st.warning(f"No data found for {selected_period}. Please try again.")
                except Exception as e:
                    st.warning(f" Data could not be retrieved for {selected_period}. Please select another time period.")

                # Comapany Information
                if 'longBusinessSummary' in info:
                    with st.expander("About Company",expanded=True):
                        try:
                            
                            st.write(info['longBusinessSummary'])
                        except:
                            st.write("Information Not Found.")

                # Technical Analysis Summary
                with st.expander("Technical Analysis Summary",expanded=True):
                    trend = "Uptrend" if daily_data['MA50'][-1] > daily_data['MA200'][-1] else "Downtrend"
                    st.write(f"{'⬆️' if daily_data['MA50'][-1] > daily_data['MA200'][-1] else '⬇️'} **Trend Status:** {trend}")
                    
                    if 'trailingPE' in info:
                        fk_status = "Low (Attractive)" if info['trailingPE'] < 10 else "High (Expensive)" if info['trailingPE'] > 20 else "Normal"
                        st.write(f"{'✅' if info['trailingPE'] < 10 else '⚠️' if info['trailingPE'] > 20 else 'ℹ️'} **F/K Status:** {fk_status}")
                    
                    rsi = daily_data['RSI'][-1]
                    rsi_status = "Oversold(Buying Opportunity)" if rsi < 30 else "Overbought(Selling Opportunity)" if rsi > 70 else "Normal"
                    st.write(f"{'✅' if 40 < rsi < 60 else '⚠️'} **RSI Status:** {rsi_status}")
                    
                    volume_change = ((daily_data['Volume'][-5:].mean() - daily_data['Volume'][-10:-5].mean()) / daily_data['Volume'][-10:-5].mean()) * 100
                    st.write(f"{'✅' if volume_change > 0 else 'ℹ️'} **Volume Change (5 Days):** {volume_change:.2f}%")

                # Raw Data
                with st.expander("Raw Data",expanded=True):
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

    # Update Graphic Styles
    def create_figure_layout(title):
        return dict(
            title=title,
            template="plotly",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=1000,  # Fixed width
            height=600,  # Fixed height
            margin=dict(l=50, r=50, t=50, b=50)  # Margin
        )

    # Update candlestick chart creation function
    def create_candlestick_chart(hist, stock_symbol, period_text):
        fig = go.Figure()
        
        # candlesticks
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="OHLC"
        ))
        
        # Moving Averages (if any)
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
        
        layout = create_figure_layout(f"{stock_symbol} {period_text} Price Chart")
        layout.update(
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
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

    # update volume chart creation function
    def create_volume_chart(hist, stock_symbol, period_text):
        fig_volume = go.Figure()
        
        # volume bars (dark blue colour)
        fig_volume.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name="Volume",
            marker_color='#001A6E'  # dark blue
        ))
        
        layout = create_figure_layout(f"{stock_symbol} {period_text} Trading Volume")
        layout.update(
            xaxis=dict(
                type='date',
                autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="all")
                    ]),
                    
                )
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                title=dict(
                    text="Transaction Volume",
                    standoff=10  # distance between y axis title and axis
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

    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        stock_symbol = st.selectbox("Stock analysis and screening tool",[""] + stock_symbols,help="Search for a company").upper()

    if stock_symbol:
        analyze_stock(stock_symbol)
    
    user_input=stock_symbol
    
    while user_input is not None and user_input.strip() != "":
           def fetch_stock_data(stock_symbol, period="5y"):
               stock = yf.Ticker(stock_symbol)
               df = stock.history(period=period)
               return df[['Close']]
 
# Function to prepare data for LSTM
# Function to prepare data for LSTM
           def prepare_data(df, time_steps=60):
                scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Ensure df is converted to NumPy array before fitting scaler
                scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

                X, y = [], []
                for i in range(len(scaled_data) - time_steps):
                    X.append(scaled_data[i:i+time_steps])
                    y.append(scaled_data[i+time_steps])
 
                return np.array(X), np.array(y), scaler


# Function to build LSTM model
           def build_lstm_model(input_shape):
               model = Sequential([
                   LSTM(50, return_sequences=True, input_shape=input_shape),
                   Dropout(0.2),
                   LSTM(50, return_sequences=False),
                   Dropout(0.2),
                   Dense(25),
                   Dense(1)
               ])
               model.compile(optimizer='adam', loss='mean_squared_error')
               return model

# ✅ Fixed Function to predict next 30 days
           def predict_next_30_days(model, last_60_days, scaler):
               predictions = []
               current_input = last_60_days.reshape(1, -1, 1)

               for _ in range(30):
                   pred = model.predict(current_input, verbose=0)[0][0]  # Extract single value
                   predictions.append(pred)

        # Append prediction correctly by reshaping
                   pred_reshaped = np.array([[pred]]).reshape(1, 1, 1)
                   current_input = np.append(current_input[:, 1:, :], pred_reshaped, axis=1)

               return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
           z=1
           if (z>0):# Streamlit UI
              
              df = fetch_stock_data(stock_symbol)
              st.write("### Historical Stock Prices")
              st.line_chart(df)

    # Prepare data
              time_steps = 60
              X, y, scaler = prepare_data(df, time_steps)

    # Train model
              model = build_lstm_model((time_steps, 1))
              model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Predict next 30 days
    # Ensure last 60 days are transformed correctly
              last_60_days = scaler.transform(df[-60:].values.reshape(-1, 1))
              predictions = predict_next_30_days(model, last_60_days, scaler)

    # Visualize predictions
              st.write("### Predicted Next 30 Days Prices")
              fig, ax = plt.subplots(figsize=(10, 5))
              ax.plot(range(1, 31), predictions, label="Predicted Prices", marker="o")
              ax.set_xlabel("Days Ahead")
              ax.set_ylabel("Stock Price")
              ax.legend()
              st.pyplot(fig)
           else:
               print("")
           break
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
            current_price = info.get('currentPrice', info.get('previousClose', 0))
            previous_close = info.get('previousClose', 0)
            p_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0

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
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_symbol_data, symbols))
        df = pd.DataFrame(results)
        
        fetched_symbols = set(df['Symbol'])
        input_symbols = set(symbols)
        missing = input_symbols - fetched_symbols
        if missing:
            st.warning(f"Missing data for: {', '.join(missing)}")
        
        return df

    def main():
        st.markdown("---")
        
        st.markdown("<h2 style='text-align: center;'>Markets at a glance (NIFTY-50 heatmap)</h2>", unsafe_allow_html=True)
        
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
                default=sector_options
            )

        filtered_df = valid_df[valid_df["Sector"].isin(selected_sectors)]

        if not filtered_df.empty:
            fig = px.treemap(
                filtered_df,
                path=[px.Constant("NSE Stocks"), 'Sector', 'Name'],
                values='MarketCap',
                color='PriceChange',
                color_continuous_scale=[[0, "red"],[0.4, "red"],[.6, "green"],[1, "green"]],
                color_continuous_midpoint=0,
                hover_data=['MarketCap', 'PriceChange', 'LastPrice'],
                width=None,
                height=800
            )

            # After creating your treemap figure with px.treemap(...)
            fig.update_traces(
            texttemplate="%{label}<br>M Cap: ₹%{customdata[0]:,.1f} Cr<br>Change: %{customdata[1]:+.2f}%<br>Price: ₹%{customdata[2]:,.1f}",
            hovertemplate="%{label}<br>Market Cap: ₹%{customdata[0]:,.1f} Cr<br>Price Change: %{customdata[1]:+.2f}%<br>Last Price: ₹%{customdata[2]:,.1f}<extra></extra>",
            textfont_size=18,
            hoverlabel=dict(font_size=18),
             marker=dict(line=dict(color="black", width=1))  # This adds a black border of 1px width around each component
            )
  # Increases the font size of the hover text
            
            

            fig.update_layout(
                margin=dict(t=50, l=0, r=0, b=0),
                coloraxis_colorbar=dict(
                    title="Price Change (%)",
                    tickprefix="",
                    ticksuffix="%"
                ),
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background outside the plot area
                plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area background
                autosize=True
            )



            # Use a full-width container to display the chart
            with st.container():
                st.plotly_chart(fig, use_container_width=True, config={'responsive': True})

        else:
            st.warning("No valid data available!")


    
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Sensex at a Glance (Heatmap)</h2>", unsafe_allow_html=True)
    
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
            default=sector_options
        )
    
    filtered_df = valid_df[valid_df["Sector"].isin(selected_sectors)]

    if not filtered_df.empty:
        fig = px.treemap(
            filtered_df,
            path=[px.Constant("Sensex Stocks"), 'Sector', 'Name'],
            values='MarketCap',
            color='PriceChange',
            color_continuous_scale=[[0, "red"], [0.4, "red"], [0.6, "green"], [1, "green"]],
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
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    else:
        st.warning("No valid data available!")

    if __name__ == "__main__":
       main()
if selected == "Portfolio Analysis":
    
    

    # Configure Gemini API
    GEMINI_API_KEY = "AIzaSyCEOyIc73JMWwtigUBovrRoBatTT9O-Bqw"
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Configure Supabase (credentials in secrets.toml)
    SUPABASE_URL = "https://lqilulmhbcxpshbefaie.supabase.co"  # 🔹 Found in Supabase Dashboard > Project Settings > API
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxxaWx1bG1oYmN4cHNoYmVmYWllIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ5NzY0MDUsImV4cCI6MjA3MDU1MjQwNX0.b2jMddO63Q3EKuaqeAaP5P55nNMdyCn1PxOZeOkaa-s"  # 🔹 Use anon key (public) or service role key (admin access)

# ✅ Create Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Database setup
    def init_db():
        # Create tables if they don’t exist (Supabase allows manual table creation via dashboard, but we’ll ensure here)
        try:
            supabase.table("users").select("username").limit(1).execute()  # Check if table exists
        except Exception:
            supabase.table("users").insert({"username": "dummy", "password": "dummy"}).execute()  # Create table
            supabase.table("users").delete().eq("username", "dummy").execute()  # Clean up dummy entry
            

        
        try:
            supabase.table("portfolios").select("username").limit(1).execute()
        except Exception:
            supabase.table("portfolios").insert({"username": "dummy", "ticker": "dummy", "shares": 0, "buy_price": 0}).execute()
            supabase.table("portfolios").delete().eq("username", "dummy").execute()
    
    # User authentication functions
    def hash_password(password):
        return sha256(password.encode()).hexdigest()
    
    def register_user(username, password):
        try:
            supabase.table("users").insert({"username": username, "password": hash_password(password)}).execute()
            return True
        except Exception as e:
            if "duplicate key" in str(e).lower():  # Check for unique constraint violation
                return False
            raise e
    
    def login_user(username, password):
        response = supabase.table("users").select("password").eq("username", username).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]["password"] == hash_password(password)
        return False
    
    def save_portfolio(username, portfolio):
        supabase.table("portfolios").delete().eq("username", username).execute()  # Clear existing portfolio
        for stock in portfolio:
            supabase.table("portfolios").insert({
                "username": username,
                "ticker": stock["Ticker"],
                "shares": stock["Shares"],
                "buy_price": stock["Buy Price"]
            }).execute()
    
    def load_portfolio(username):
        response = supabase.table("portfolios").select("*").eq("username", username).execute()
        return [{"Ticker": row["ticker"], "Shares": row["shares"], "Buy Price": row["buy_price"]} for row in response.data]
    
    # Gemini AI portfolio analysis function
    def get_gemini_portfolio_analysis(portfolio_df):
        model = genai.GenerativeModel('gemini-1.5-flash')
        portfolio_summary = portfolio_df.to_string()
        prompt = f"""
        You are a financial expert AI. Analyze the following stock portfolio and provide:
        1. A detailed description of the portfolio.
        2. Key risks associated with the portfolio.
        3. Positive aspects and potential benefits of the portfolio.
        
        Portfolio Data:
        {portfolio_summary}
        """
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error fetching Gemini AI analysis: {str(e)}"
    
    # Initialize database
    init_db()
    
    # Enhanced Dark Mode Custom CSS
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Arial', sans-serif;
        }
        .header {
            background-color: #1e1e1e;
            color: #ffffff;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        .header h2 {
            margin-bottom: 10px;
            font-weight: bold;
            color: #4CAF50;
        }
        .header-text {
            color: #b0b0b0;
            font-size: 0.9em;
        }
        .metric-box {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 15px;
            color: #e0e0e0;
        }
        .metric-box .gain { color: #4CAF50; }
        .metric-box .loss { color: #F44336; }
        .st-emotion-cache-1aumxhk {
            background-color: #1a1a1a;
            border-right: 1px solid #333;
        }
        .stTextInput input, .stNumberInput input {
            background-color: #2a2a2a !important;
            color: #e0e0e0 !important;
            border: 1px solid #444 !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stDataFrame, .stLineChart, .stBarChart {
            background-color: #1e1e1e;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            padding: 15px;
            border: 1px solid #333;
        }
        footer {
            background-color: #1e1e1e;
            color: #b0b0b0;
            text-align: center;
            padding: 10px;
            position: fixed;
            bottom: 0;
            width: 100%;
            left: 0;
            border-top: 1px solid #333;
        }
        </style>
    """, unsafe_allow_html=True)
    
    
    # Authentication
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
        # Header
        st.markdown("""
            <div class="header">
                <h2>Welcome to Your Stock Portfolio</h2>
                <p class="header-text">Track, analyze, and stay updated with your Indian stock investments—all in one place.</p>
            </div>
        """, unsafe_allow_html=True)
    
        # Main navigation
        selected = option_menu(
            menu_title=None,
            options=["Portfolio Analysis"],
            icons=["briefcase","newspaper"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"font-size": "18px", "width": "70%"},
                "nav-link": {"font-size": "18px", "padding": "4px 8px", "margin": "0 4px", "font-weight": "bold"},
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
    
        
        
        # Sidebar with simplified symbol selection
        with st.sidebar:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.portfolio = []
                st.rerun()
    
            st.header(f"Portfolio for {st.session_state.username}")
            
            # Single selectbox for stock symbols
            ticker = st.selectbox("Stock Ticker (e.g., RELIANCE.NS)", [""] + stock_symbols,help="Select a stock symbol from the list")
            
            shares = st.number_input("Shares", min_value=1, value=1)
            buy_price = st.number_input("Buy Price (INR)", min_value=0.0, value=0.0)
    
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add Stock"):
                    if ticker and shares and buy_price:
                        ticker_upper = ticker.upper()
                        existing_stock = next((stock for stock in st.session_state.portfolio if stock["Ticker"] == ticker_upper), None)
                        if existing_stock:
                            old_shares = existing_stock["Shares"]
                            old_cost = old_shares * existing_stock["Buy Price"]
                            new_cost = shares * buy_price
                            total_shares = old_shares + shares
                            new_avg_buy_price = (old_cost + new_cost) / total_shares
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
                        save_portfolio(st.session_state.username, st.session_state.portfolio)
            with col2:
                if st.button("Clear Portfolio", key="clear"):
                    st.session_state.portfolio = []
                    save_portfolio(st.session_state.username, st.session_state.portfolio)
                    st.success("Portfolio cleared!")

            st.markdown("---")
            st.subheader("Remove Stock")
            remove_ticker = st.selectbox("Select Stock to Remove", [""] + [stock["Ticker"] for stock in st.session_state.portfolio], key="remove_ticker")
            remove_shares = st.number_input("Shares to Remove", min_value=1, value=1, key="remove_shares")
            
            if st.button("Remove Stock", key="remove_stock"):
                if remove_ticker:
                    existing_stock = next((stock for stock in st.session_state.portfolio if stock["Ticker"] == remove_ticker), None)
                    if existing_stock:
                        if existing_stock["Shares"] >= remove_shares:
                            existing_stock["Shares"] -= remove_shares
                            if existing_stock["Shares"] == 0:
                                st.session_state.portfolio = [stock for stock in st.session_state.portfolio if stock["Ticker"] != remove_ticker]
                            save_portfolio(st.session_state.username, st.session_state.portfolio)
                            st.success(f"Removed {remove_shares} shares of {remove_ticker}!")
                        else:
                            st.error(f"Not enough shares! You only have {existing_stock['Shares']} shares of {remove_ticker}.")
                    else:
                        st.error(f"Stock {remove_ticker} not found in your portfolio!")
                else:
                    st.error("Please select a stock to remove!")
            
            
    
            st.markdown("---")
            st.markdown("""
                ### How to Use
                - Select a stock symbol from the list.
                - Add shares and buy price.
                - Click 'Add Stock' or 'Clear Portfolio'.
            """, unsafe_allow_html=True)
        
                       
        
    
        # Portfolio Analysis
        if selected == "Portfolio Analysis":
            if not st.session_state.portfolio:
                st.markdown("<div>Add stocks in the sidebar to see your portfolio analysis!</div>", unsafe_allow_html=True)
            else:
                st.markdown("---")
                st.header("Portfolio Analysis")
                selected_timeframe = st.selectbox("Select Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=0)
                
                nifty_data = yf.Ticker("^NSEI").history(period=TIMEFRAME_OPTIONS[selected_timeframe])
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
                        history_1d = ticker_obj.history(period="1d")
                        history_timeframe = ticker_obj.history(period=TIMEFRAME_OPTIONS[selected_timeframe])
                        current_price = history_1d["Close"].iloc[-1]
                        total_value = current_price * stock["Shares"]
                        profit_loss = (current_price - stock["Buy Price"]) * stock["Shares"]
                        info = ticker_obj.info
    
                        beta = info.get("beta", 0)
                        capm_expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
                        div_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
                        intraday_vol = ((history_1d["High"].iloc[-1] - history_1d["Low"].iloc[-1]) / current_price) * 100
                        stock_return_timeframe = (history_timeframe["Close"].iloc[-1] / history_timeframe["Close"].iloc[0] - 1) * 100
                        rel_strength = stock_return_timeframe - nifty_return
                        days_to_breakeven = abs(profit_loss / (history_timeframe["Close"].diff().mean() * stock["Shares"])) if profit_loss < 0 else 0
    
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
    
                        normalized_prices = (history_timeframe["Close"] / history_timeframe["Close"].iloc[0]) * 100
                        normalized_data[stock["Ticker"]] = normalized_prices
                        portfolio_value_history += history_timeframe["Close"] * stock["Shares"]
    
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
    
                # Gemini AI Analysis
                st.markdown("---")
                st.subheader("AI-Powered Portfolio Insights (Powered by Gemini)")
                gemini_analysis = get_gemini_portfolio_analysis(df)
                st.markdown(gemini_analysis)
    
                st.markdown("---")
                st.subheader("Portfolio Summary")
                total_investment = sum(stock["Shares"] * stock["Buy Price"] for stock in st.session_state.portfolio)
                total_value_sum = df["Total Value (INR)"].sum()
                total_pl_sum = df["Profit/Loss (INR)"].sum()
                percent_pl = (total_pl_sum / total_investment) * 100 if total_investment > 0 else 0
                portfolio_beta = np.average(df["Beta"], weights=df["Total Value (INR)"]) if total_value_sum > 0 else 0
                portfolio_capm = np.average(df["CAPM Exp Return (%)"], weights=df["Total Value (INR)"]) if total_value_sum > 0 else 0
                div_yield_contrib = sum(df["Div Yield (%)"] * df["Total Value (INR)"]) / total_value_sum if total_value_sum > 0 else 0
    
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='metric-box'><b>Total Investment</b><br>₹{total_investment:,.2f}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><b>Total Value</b><br>₹{total_value_sum:,.2f}</div>", unsafe_allow_html=True)
                with col3:
                    pl_class = "gain" if total_pl_sum >= 0 else "loss"
                    st.markdown(f"<div class='metric-box'><b>Profit/Loss</b><br><span class='{pl_class}'>₹{total_pl_sum:,.2f} ({percent_pl:.2f}%)</span></div>", unsafe_allow_html=True)
    
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.markdown(f"<div class='metric-box'><b>Portfolio Beta</b><br>{portfolio_beta:.2f}</div>", unsafe_allow_html=True)
                with col5:
                    st.markdown(f"<div class='metric-box'><b>Portfolio CAPM Return</b><br>{portfolio_capm:.2f}%</div>", unsafe_allow_html=True)
                with col6:
                    st.markdown(f"<div class='metric-box'><b>Div Yield Contribution</b><br>{div_yield_contrib:.2f}%</div>", unsafe_allow_html=True)
    
                st.markdown("---")
                st.subheader("Portfolio vs Nifty 50 Comparison")
                portfolio_normalized = (portfolio_value_history / portfolio_value_history.iloc[0]) * 100
                nifty_normalized = (nifty_close / nifty_close.iloc[0]) * 100
                comparison_data = pd.DataFrame({"Portfolio": portfolio_normalized, "Nifty 50": nifty_normalized})
                st.line_chart(comparison_data, use_container_width=True)
    
                portfolio_returns = portfolio_value_history.pct_change().dropna()
                nifty_returns = nifty_close.pct_change().dropna()
                portfolio_total_return = (portfolio_value_history[-1] / portfolio_value_history[0] - 1) * 100
                tracking_error = (portfolio_returns - nifty_returns).std() * 100
                correlation = portfolio_returns.corr(nifty_returns)
    
                col1, col2, col3 = st.columns(3)
                with col1:
                    return_class = "gain" if portfolio_total_return >= nifty_return else "loss"
                    st.markdown(f"<div class='metric-box'><b>Portfolio Return</b><br><span class='{return_class}'>{portfolio_total_return:.2f}%</span></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><b>Tracking Error</b><br>{tracking_error:.2f}%</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-box'><b>Correlation with Nifty</b><br>{correlation:.2f}</div>", unsafe_allow_html=True)
    
                st.markdown("---")
                st.subheader("Normalized Price Performance")
                st.line_chart(normalized_data, use_container_width=True)
    
                st.markdown("---")
                st.subheader("Profit/Loss by Stock")
                st.bar_chart(df.set_index("Ticker")["Profit/Loss (INR)"], use_container_width=True)
                
if selected == "News":
    st.markdown("""
    <style>
    .news-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .news-title {
        font-size: 18px;
        font-weight: bold;
        color: #0078ff;
    }
    .news-source {
        font-size: 14px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI Header ---
    st.markdown("<h1 style='text-align: center;'>📈 Indian Stock Market News</h1>", unsafe_allow_html=True)
    st.write("Get the latest stock market news powered by **Google News RSS Feed**.")

# --- Stock Symbol Input ---
    query = st.selectbox("🔍 Enter Stock Name (e.g., Reliance, Infosys, TCS):",[""] + stock_symbols,help="Select a stock symbol from the list")

# --- Fetch stock logo from Yahoo Finance ---
    def get_stock_logo(symbol):
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}.NS"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "quoteResponse" in data and "result" in data["quoteResponse"]:
                result = data["quoteResponse"]["result"]
                if result and "logo_url" in result[0]:
                   return result[0]["logo_url"]
        return None  # Return None if logo not found

# --- Fetch news from Google News RSS ---
    def fetch_google_news_rss(query):
        url = f"https://news.google.com/rss/search?q={query}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url)
        if response.status_code != 200:
            return None

        news_list = []
        root = ET.fromstring(response.content)

        for item in root.findall(".//item")[:5]:  # Get top 5 news
            title = item.find("title").text
            link = item.find("link").text
            news_list.append((title, link))

        return news_list

    if st.button("🔍 Get Latest News"):
       with st.spinner("Fetching latest news..."):
            news = fetch_google_news_rss(query)
            if news:
               st.markdown("### 📰 Latest News")
               for title, link in news:
                st.markdown(
                    f"""
                    <div class='news-card'>
                        <p class='news-title'>🔹 <a href='{link}' target='_blank'>{title}</a></p>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.error("❌ Failed to fetch news. Try a different stock name.")

