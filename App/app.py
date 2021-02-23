import json
import pandas_datareader as web #Get data
import numpy as np #Matrix operation
import pandas as pd #Dataframes
from sklearn.preprocessing import MinMaxScaler #Normalization
import matplotlib.pyplot as plt #Graphs
import datetime #Today's date
from sklearn.metrics import mean_squared_error #RMSE ERROR
from tensorflow.keras.models import load_model #Load Keras Model
import streamlit as st
import altair as alt





#read Json file and choose one model
with open('data.json') as json_file:
    data = json.load(json_file)


st.title('Stock Prices Prediction Using LSTM Models')

st.markdown(
    "This is an example of predict stock market prices using deep learning. In particular the **LSTM** model. The model use the past 60 days to predict the actual price.")


#Select Corp
corp = st.sidebar.selectbox('Select stock to predict:', ['Apple','Netflix','Amazon','Facebook','Ecopetrol','Bancolombia'])
stock = [model for model in data['models'] if model['corp'] == corp][0]
#Testing the model in real time data
#Get the data of Corp choosed. Using yahoo finance as source and last 5 years
today=datetime.datetime.now().strftime("%Y-%m-%d")
df=web.DataReader(stock['name'],
                  data_source='yahoo',
                  start=stock['startTraining'],
                  end=today)

#load Keras Model
model = load_model(f"models/{stock['model']}")
#predict prices after training an test data (realtime)
df2=df.reset_index()
n_rows=df2.loc[df2['Date']==stock['endTraining']].index[0]
data=df[n_rows-60:].filter(['Close'])
#Y values (real values)
y=data[60:]
y=np.array(y)
#MinMax Scaler for x matrix
scaler=MinMaxScaler(feature_range=(0,1))
scaled_x = scaler.fit_transform(data)
#x values (last 60 days)
x=[]
for i in range(60,len(scaled_x)):
    x.append(scaled_x[i-60:i,0])
x = np.array(x)
#Reshape
x=np.reshape(x,(x.shape[0],x.shape[1],1))
#Get predicted scaled prices
pred_price=model.predict(x)
predictions=scaler.inverse_transform(pred_price)
_df=df[n_rows:].filter(['Close'])
_df['Predictions']=predictions

#RMSE using ScikitLearn
rmse=mean_squared_error(_df['Close'],_df['Predictions'],squared=False)
#Predict Tomorrow price
xt=scaled_x[-60:]
xt=np.reshape(xt,(1,60,1))
yt1 = scaler.inverse_transform(model.predict(xt))[0]

#Plot
_x=df[['Close']].rename(columns={'Close':'Price $USD'})
_x['Symbol']='Close Price'
_y=_df[['Predictions']].rename(columns={'Predictions':'Price $USD'})
_y['Symbol']='Predicted Price'
_x = _x.append(_y).reset_index()

fig=alt.Chart(_x,title=f"Model Prediction for {corp}").mark_line().encode(
    x='Date',
    y='Price $USD',
    color='Symbol',
    strokeDash='Symbol',
).properties(
    width="container",
    height=300
).configure(
    legend=alt.LegendConfig(direction='horizontal',orient="bottom-left")
)




#Print in screen
color = None
diff = yt1-_df.iloc[-1]['Close']
diff=diff[0]
if diff > 0:
    color = "green"
else:
    color="red"

st.header(f"{corp}")

st.markdown(f'<div style="display: flex;border-radius:10px;flex-direction: row;justify-content: center;align-items: center;border: 1px solid lightgray;padding: 20px; margin-bottom:30px":margin-top:20px> <p style="margin:0;padding-right:10px">Tomorrow Close Price Predicted: </p> <p style="margin:0;font-weight: 500;color:{color}">{round(float(yt1),2)} USD</p> </div>',
 unsafe_allow_html=True)


st.altair_chart(fig, use_container_width=True)
#Datafame

_df['Error']=_df['Predictions']-_df['Close']
st.dataframe(_df.tail(10).sort_index(ascending=False),600)

st.markdown(
    f'<p style="text-align:center"> This model was trained using data from {stock["startTraining"]} to {stock["endTraining"]}. </p>',
    unsafe_allow_html=True)


st.markdown(f'<div style="width:100%;text-align:center"><a href="https://github.com/boring-programmer7">Follow me on Github and get the code</a></div>',
unsafe_allow_html=True)

st.markdown(f'<div style="width:100%;text-align:center"><a href="https://www.linkedin.com/in/andr%C3%A9s-felipe-acevedo-rodr%C3%ADguez-5a2453158/">Follow me on Linkedin</a></div>',
unsafe_allow_html=True)