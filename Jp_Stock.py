# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 01:11:37 2021

@author: karat
"""

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#追加分
import pandas_datareader as web


START="2015-01-01"
TODAY=date.today().strftime("%Y-%m-%d")

st.title("当たるといいね！機械学習で株価の予測")

code=st.text_input('株のコードを入力 (日本の株は最後に.jpをつける)', '7974.JP')

#stocks=("AAPL","GOOG","MSFT","GME")
selected_stock=code

n_years=st.slider("Years of prediction:",1,4)
period=n_years*365

@st.cache
def load_data(ticker):
    #data=yf.download(ticker,START,TODAY)
    #作業中
    data=web.DataReader(code,'stooq')
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data...")
data=load_data(code)
data_load_state.text("Loading data...done!")

st.subheader("5年分のデータ")
st.write(data.tail())

    
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

#Forecasting
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
    
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.subheader('予測データ')
st.write(forecast.tail())

st.write('forcast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast componets')
fig2=m.plot_components(forecast)
st.write(fig2)
    
    
    
    
    
    
    
    
    
    
    
    