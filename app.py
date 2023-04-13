import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import streamlit as st
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2023, 1, 31)

st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker','TSLA')

df = web.DataReader(user_input, 'tiingo', start, end,api_key='f16e807c3ff8cdadadc1b5828bfac4d9acce61e7')
print(df.head())

#Describing data
st.subheader('Data from 2015-2023')
st.write(df.describe())


df=df.reset_index()
df=df.drop(['date','adjClose','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor','symbol'],axis=1)

#Visualization
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100=df.close.rolling(100).mean()
ma200=df.close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.close,'b')
st.pyplot(fig)

#spilitting data into Traninig and Testing
data_training=pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#load my model
model=load_model('keras.model.h5')

#Testing part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing, ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
#prediction
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted * scale_factor
y_test=y_test * scale_factor

#final graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)