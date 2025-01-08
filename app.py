import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import base64
import matplotlib.pyplot as plt

#loading data from yahoofinance
model = load_model(r'C:\Users\amish\OneDrive\Desktop\python_programs\Bitcoin_price_prediction.keras')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Use background image
img_base64 = get_base64_of_bin_file(r'C:\Users\amish\OneDrive\Desktop\python_programs\pexels-davidmcbee-730547.jpg')
# Set up the background image with CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        animation: moveBackground 20s linear infinite;
    }}
    @keyframes moveBackground {{
        0% {{ background-position: center top; }}
        100% {{ background-position: center bottom; }}
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
        background-color: black;
        padding: 20px;
        border-radius: 8px;
        font-size: 3em !important;
        font-weight: bold !important;
        text-align: center;
        width: 100%;
        margin: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


#data from dataset
st.header('Bitcoin Price Prediction Model')
st.subheader('Bitcoin Price Prediction')
data = pd.DataFrame(yf.download('BTC-USD', '2014-09-17','2024-09-08'))
data = data.reset_index()
st.write(data)

#line chart of bitcoin data
st.subheader('Bitcoin Line Chart')
data.drop(columns=['Date','Open','High','Low','Adj Close','Volume'], inplace=True)
st.line_chart(data)

train_data = data[:-100]
test_data = data[-200:]

scaler = MinMaxScaler(feature_range=(0,1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)
base_days = 100
x = []
y = []
for i in range(base_days, test_data_scale.shape[0]):
  x.append(test_data_scale[i-base_days:i])
  y.append(test_data_scale[i,0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0],x.shape[1],1))

st.subheader('Predicted vs Original prices')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1,1)
ys = scaler.inverse_transform(y.reshape(-1,1))
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys),axis=1)
st.write(chart_data)
st.subheader('Predicted vs Original prices')
st.line_chart(chart_data)

m = y
z = []
future_days = 10
for i in range (base_days, len(m)+future_days):
  m = m.reshape(-1,1)
  inter = [m[-base_days:,0]]
  inter = np.array(inter)
  inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
  pred = model.predict(inter)
  m = np.append(m, pred)
  z = np.append(z, pred)

future_days = st.number_input('', min_value=1, max_value=365, value=10)
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1, 1))

pred_dates = pd.date_range(start='2024-09-08', periods=future_days, freq='D')
future_predictions = pd.DataFrame(z[-future_days:], columns=['Predicted Price'], index=pred_dates)

#scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(future_predictions.index, future_predictions['Predicted Price'], color='orange')
plt.title("Scatter Plot of Predicted Bitcoin Prices", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Price (USD)", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(plt)

#area chart
plt.figure(figsize=(10, 6))
plt.fill_between(future_predictions.index, future_predictions['Predicted Price'], color='skyblue', alpha=0.4)
plt.plot(future_predictions.index, future_predictions['Predicted Price'], color='blue')
plt.title("Area Chart of Predicted Bitcoin Prices", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Price (USD)", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(plt)

st.markdown("""
    <style>
    /* Styling for Key Statistics container */
    .key-stats-container {
        background-color: #f8f9fa; /* Light grey background */
        padding: 20px;             /* Padding around content */
        border-radius: 8px;        /* Rounded corners */
        margin-top: 20px;          /* Space above */
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
    
    /* Styling for each statistic */
    .metric {
        display: inline-block;
        text-align: center;
        width: 30%;
        background-color: #ffffff; /* White background for contrast */
        padding: 10px;
        margin: 10px;
        border-radius: 8px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Title styling inside Key Statistics */
    .key-stats-container h2 {
        text-align: center;        /* Centered title */
        color: #007bff;            /* Blue color for title */
        font-weight: bold;
        margin-bottom: 15px;
    }

    /* Statistic title and value styling */
    .metric h3 {
        color: #333;               /* Dark color for titles */
        margin: 0;
    }
    .metric p {
        font-size: 1.5em;          /* Larger font for values */
        color: #007bff;            /* Blue color for values */
        margin: 5px 0 0 0;
    }
    </style>
""", unsafe_allow_html=True)

# Display Key Statistics section
st.markdown('<div class="key-stats-container"><h2>Key Statistics</h2>', unsafe_allow_html=True)

# Metrics with styled containers
st.markdown(f"""
    <div class="metric">
        <h3>Max Price</h3>
        <p>${data['Close'].max():.2f}</p>
    </div>
    <div class="metric">
        <h3>Min Price</h3>
        <p>${data['Close'].min():.2f}</p>
    </div>
    <div class="metric">
        <h3>Avg Price</h3>
        <p>${data['Close'].mean():.2f}</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

#future prediction line chart
st.line_chart(future_predictions)
st.write(future_predictions)
st.subheader('Future Days Bitcoin Price Predicted')
z= np.array(z)
z = scaler.inverse_transform(z.reshape(-1,1))
st.line_chart(z)

#pie chart
price_bins = ['<30k', '50k-60k', '60k+']
price_ranges = [0, 50000, 60000, np.inf]
price_categories = pd.cut(future_predictions['Predicted Price'], bins=price_ranges, labels=price_bins)

pie_chart_data = price_categories.value_counts()
fig_pie = plt.figure(figsize=(8, 8))  # Increase figure size
plt.pie(pie_chart_data, labels=pie_chart_data.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85, 
        wedgeprops={'edgecolor': 'black'})  # Better spacing and edge color for clarity
plt.title("Price Distribution of Bitcoin Prices", fontsize=16)
st.pyplot(fig_pie)

#bar chart
price_bins = ['<30k', '30k-40k', '40k-50k', '50k-60k', '60k+']
price_ranges = [0, 30000, 40000, 50000, 60000, np.inf]
price_categories = pd.cut(future_predictions['Predicted Price'], bins=price_ranges, labels=price_bins)
# Counting the number of predicted prices in each bin
price_counts = price_categories.value_counts()
# Plotting the Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(price_counts.index, price_counts.values, color='skyblue', edgecolor='black')
plt.title("Predicted Bitcoin Prices", fontsize=16)
plt.xlabel("Price Range", fontsize=12)
plt.ylabel("Count", fontsize=12)
st.pyplot(plt)

#stacked bar chart
time_series = future_predictions.copy()
time_series['Price Range'] = pd.cut(time_series['Predicted Price'], bins=price_ranges, labels=price_bins)

# Create a pivot table for stacked bars
pivot_data = time_series.groupby(['Price Range', time_series.index.month]).size().unstack()

pivot_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3')
plt.title("Monthly Distribution of Predicted Bitcoin Prices", fontsize=16)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Count", fontsize=12)
st.pyplot(plt)