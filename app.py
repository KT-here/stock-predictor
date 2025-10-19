import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 50px !important;
        color: #1f77b4;
        text-align: center;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">📈 Stock Price Predictor</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Stock Selection")

# Popular Indian stocks
popular_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS", 
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS"
}

selected_stock = st.sidebar.selectbox(
    "Choose a stock:",
    list(popular_stocks.keys())
)

symbol = popular_stocks[selected_stock]

# Prediction settings
st.sidebar.header("Prediction Settings")
prediction_days = st.sidebar.slider("Days to Predict", 1, 7, 3)

# Main function
def main():
    if st.sidebar.button("Analyze Stock", type="primary"):
        with st.spinner("Downloading data and analyzing..."):
            try:
                # Download stock data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    st.error("No data found for this stock!")
                    return
                
                # Display current price
                current_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"₹{current_price:.2f}",
                        f"{change_pct:+.2f}%"
                    )
                
                with col2:
                    st.metric("52W High", f"₹{df['High'].max():.2f}")
                
                with col3:
                    st.metric("52W Low", f"₹{df['Low'].min():.2f}")
                
                # Price chart
                st.subheader("Price Chart")
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))
                
                fig.update_layout(
                    height=500,
                    title=f"{selected_stock} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Simple prediction (moving average based)
                st.subheader("Price Prediction")
                
                # Calculate moving averages
                ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
                ma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
                
                # Simple prediction logic
                trend = "BULLISH" if ma_20 > ma_50 else "BEARISH"
                momentum = (ma_20 - ma_50) / ma_50
                
                # Generate predictions
                predictions = []
                last_price = current_price
                
                for day in range(1, prediction_days + 1):
                    # Simple prediction based on momentum
                    predicted_change = momentum * last_price * np.random.uniform(0.8, 1.2)
                    predicted_price = last_price + predicted_change
                    predictions.append({
                        'Day': day,
                        'Date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                        'Predicted Price': predicted_price,
                        'Change %': ((predicted_price - current_price) / current_price) * 100
                    })
                    last_price = predicted_price
                
                prediction_df = pd.DataFrame(predictions)
                
                # Display predictions
                st.dataframe(
                    prediction_df.style.format({
                        'Predicted Price': '₹{:.2f}',
                        'Change %': '{:+.2f}%'
                    }),
                    use_container_width=True
                )
                
                # Trading recommendation
                total_change = predictions[-1]['Change %']
                
                st.markdown("### 💡 Trading Recommendation")
                
                if total_change > 3:
                    recommendation = "STRONG BUY 🟢"
                    reasoning = "Strong upward momentum expected"
                elif total_change > 1:
                    recommendation = "BUY 🟢"
                    reasoning = "Moderate growth expected"
                elif total_change < -3:
                    recommendation = "STRONG SELL 🔴"
                    reasoning = "Significant decline expected"
                elif total_change < -1:
                    recommendation = "SELL 🔴"
                    reasoning = "Moderate decline expected"
                else:
                    recommendation = "HOLD 🟡"
                    reasoning = "Limited movement expected"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>{recommendation}</h3>
                    <p><strong>Reason:</strong> {reasoning}</p>
                    <p><strong>Expected Return:</strong> {total_change:+.2f}% in {prediction_days} days</p>
                    <p><strong>Market Trend:</strong> {trend}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Prediction chart
                st.subheader("Prediction Chart")
                
                fig_pred = go.Figure()
                
                # Historical data (last 30 days)
                historical = df.tail(30)
                fig_pred.add_trace(go.Scatter(
                    x=historical.index,
                    y=historical['Close'],
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color='blue', width=2)
                ))
                
                # Prediction data
                pred_dates = [datetime.now() + timedelta(days=i) for i in range(1, prediction_days + 1)]
                pred_prices = [p['Predicted Price'] for p in predictions]
                
                fig_pred.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='lines+markers',
                    name='Predicted Prices',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_pred.update_layout(
                    title="Price Predictions",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again or check your internet connection.")
    
    else:
        # Welcome message
        st.info("👈 Select a stock and click 'Analyze Stock' to get started!")
        
        st.markdown("""
        ### 🎯 How to Use This App
        
        1. **Select a stock** from the sidebar
        2. **Choose prediction days** (1-7 days)
        3. **Click "Analyze Stock"** to generate predictions
        4. **Review** the analysis and recommendations
        
        ### 📊 Features
        - Real-time stock data
        - Interactive price charts
        - Price predictions
        - Trading recommendations
        - Technical analysis
        
        ### ⚠️ Disclaimer
        This app is for educational purposes only. 
        Stock predictions are not financial advice.
        Always do your own research before investing.
        """)

if __name__ == "__main__":
    main()
