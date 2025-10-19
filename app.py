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
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .positive {
        color: green;
        font-weight: bold;
    }
    .negative {
        color: red;
        font-weight: bold;
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
    "Bajaj Finance": "BAJFINANCE.NS",
    "Wipro": "WIPRO.NS",
    "LT": "LT.NS"
}

selected_stock = st.sidebar.selectbox(
    "Choose a stock:",
    list(popular_stocks.keys())
)

symbol = popular_stocks[selected_stock]

# Prediction settings
st.sidebar.header("Prediction Settings")
prediction_days = st.sidebar.slider("Days to Predict", 1, 7, 3)

# Cache data to improve performance
@st.cache_data(ttl=3600)
def download_stock_data(symbol, period="1y"):
    """Download stock data with error handling"""
    try:
        stock_data = yf.download(symbol, period=period, progress=False)
        return stock_data
    except Exception as e:
        return None

def simple_prediction_model(df, days=3):
    """Simple prediction using moving averages"""
    try:
        current_price = df['Close'].iloc[-1]
        
        # Calculate trends
        short_ma = df['Close'].tail(20).mean()
        long_ma = df['Close'].tail(50).mean()
        trend = "BULLISH" if short_ma > long_ma else "BEARISH"
        
        # Simple momentum calculation
        momentum = (short_ma - long_ma) / long_ma
        
        predictions = []
        last_price = current_price
        
        for day in range(1, days + 1):
            # Conservative prediction based on momentum
            daily_change = momentum * 0.3  # Reduced impact for stability
            predicted_price = last_price * (1 + daily_change)
            
            change_pct = ((predicted_price - current_price) / current_price) * 100
            
            predictions.append({
                'Day': day,
                'Date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'Predicted Price': predicted_price,
                'Change %': change_pct
            })
            last_price = predicted_price
        
        return predictions, trend, momentum
        
    except Exception as e:
        return [], "NEUTRAL", 0

def create_display_table(predictions):
    """Create a display-friendly table without pandas styling"""
    display_data = []
    
    for pred in predictions:
        display_data.append({
            'Day': pred['Day'],
            'Date': pred['Date'],
            'Predicted Price': f"₹{pred['Predicted Price']:.2f}",
            'Change %': f"{pred['Change %']:+.2f}%"
        })
    
    return pd.DataFrame(display_data)

# Main function
def main():
    if st.sidebar.button("Analyze Stock", type="primary"):
        with st.spinner("Downloading data and analyzing..."):
            try:
                # Download stock data
                df = download_stock_data(symbol)
                
                if df is None or df.empty:
                    st.error("❌ No data found for this stock. Please try another symbol.")
                    return
                
                if len(df) < 50:
                    st.warning("⚠️ Limited data available. Analysis may be less accurate.")
                
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
                st.subheader("📊 Price Chart")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(window=20).mean(),
                    mode='lines',
                    name='MA 20',
                    line=dict(color='orange', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(window=50).mean(),
                    mode='lines',
                    name='MA 50',
                    line=dict(color='red', width=1)
                ))
                
                fig.update_layout(
                    height=500,
                    title=f"{selected_stock} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate predictions
                st.subheader("🔮 Price Prediction")
                
                predictions, trend, momentum = simple_prediction_model(df, prediction_days)
                
                if predictions:
                    # Create display table without any pandas styling
                    display_df = create_display_table(predictions)
                    
                    # Display predictions
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Trading recommendation
                    total_change = predictions[-1]['Change %']
                    
                    st.markdown("### 💡 Trading Recommendation")
                    
                    if total_change > 3:
                        recommendation = "STRONG BUY 🟢"
                        reasoning = "Strong upward momentum expected"
                        confidence = "High"
                    elif total_change > 1:
                        recommendation = "BUY 🟢"
                        reasoning = "Moderate growth expected"
                        confidence = "Medium"
                    elif total_change < -3:
                        recommendation = "STRONG SELL 🔴"
                        reasoning = "Significant decline expected"
                        confidence = "High"
                    elif total_change < -1:
                        recommendation = "SELL 🔴"
                        reasoning = "Moderate decline expected"
                        confidence = "Medium"
                    else:
                        recommendation = "HOLD 🟡"
                        reasoning = "Limited movement expected"
                        confidence = "Low"
                    
                    # Determine color for change percentage
                    change_color = "positive" if total_change > 0 else "negative"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>{recommendation}</h3>
                        <p><strong>Reason:</strong> {reasoning}</p>
                        <p><strong>Confidence:</strong> {confidence}</p>
                        <p><strong>Expected Return:</strong> <span class="{change_color}">{total_change:+.2f}%</span> in {prediction_days} days</p>
                        <p><strong>Market Trend:</strong> {trend}</p>
                        <p><strong>Data Points Analyzed:</strong> {len(df)} trading days</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction chart
                    st.subheader("📈 Prediction Chart")
                    
                    fig_pred = go.Figure()
                    
                    # Historical data (last 60 days)
                    historical = df.tail(60)
                    fig_pred.add_trace(go.Scatter(
                        x=historical.index,
                        y=historical['Close'],
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Prediction data
                    pred_dates = [datetime.now() + timedelta(days=i) for i in range(1, prediction_days + 1)]
                    pred_prices = [p['Predicted Price'] for p in predictions]
                    
                    fig_pred.add_trace(go.Scatter(
                        x=pred_dates,
                        y=pred_prices,
                        mode='lines+markers',
                        name='Predicted Prices',
                        line=dict(color='red', width=3, dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title="Price Predictions",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                else:
                    st.error("❌ Could not generate predictions. Please try again.")
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("🔧 Please try the following:")
                st.info("1. Check your internet connection")
                st.info("2. Try a different stock symbol")
                st.info("3. Wait a moment and try again")
    
    else:
        # Welcome message
        st.info("👈 **Select a stock and click 'Analyze Stock' to get started!**")
        
        st.markdown("""
        ### 🎯 How to Use This App
        
        1. **Select a stock** from the sidebar
        2. **Choose prediction days** (1-7 days)
        3. **Click "Analyze Stock"** to generate predictions
        4. **Review** the analysis and recommendations
        
        ### 📊 Features
        - Real-time stock data from Yahoo Finance
        - Interactive price charts
        - Simple price predictions
        - Trading recommendations
        - Technical indicators (Moving Averages)
        
        ### ⚠️ Important Notes
        - Predictions are based on technical analysis
        - This is for **educational purposes only**
        - **Not financial advice** - always do your own research
        - Data may have slight delays
        """)

if __name__ == "__main__":
    main()
