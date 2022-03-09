import streamlit as st

def app():
    import math
    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from keras.models import load_model
    from tensorflow.keras.layers import Dense,Dropout,LSTM
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    plt.style.use('fivethirtyeight')


    # In[2]:
    
    


    #!pip install nsepy


    # In[3]:

    import datetime
    from nsepy import get_history
    from datetime import date

         

    st.title("Stock Trend Prediction")
    st.image('./bulb.png')
    user_input = st.text_input('Enter stock ticker', 'SBIN')
    start_date = datetime.date.today() - datetime.timedelta(days=1277)

    data=get_history(symbol=user_input,start=start_date,end = datetime.date.today(),index=0)

    end_date = datetime.date.today() + datetime.timedelta(days=1)
    start_date = datetime.date.today() - datetime.timedelta(days=120)
    nifty_quote = get_history(symbol=user_input, start=start_date, end=end_date,index=0)
    stock_data = get_history(symbol=user_input, start=date(2022,3,7), end=date(2022,3,8),index=0)

    st.write(stock_data.Open[1:])
    st.write(stock_data.Close[1:])
    st.write(stock_data.High[1:])
    st.write(stock_data.Low[1:])
    # In[4]:

    #Visualizations
    st.subheader('Closing price vs Time')
    fig = plt.figure(figsize = (12,6))
    plt.xlabel("Date",fontsize=18)
    plt.ylabel("Close Price of Stock",fontsize=18)
    plt.plot(data.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 50MA & 200MA')
    ma_50 = data.Close.rolling(50).mean()
    ma_200 = data.Close.rolling(200).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma_50,color ='tab:green')
    plt.plot(ma_200,color ='tab:red')
    plt.plot(data.Close)
    plt.legend(['50MA',"200MA"],loc="upper left")
    st.pyplot(fig)


    # In[5]:
    #Training and testing
    close=data.filter(["Close"])
    dataset = close.values
    training_data_len=math.ceil(len(dataset)*0.60)

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)



    #Load my model
    model = load_model('models.h5')
    #X_test.append(last_60_days_scaled)
    test_data=scaled_data[training_data_len-60:,:]
    x_test=[]
    y_test=dataset[training_data_len:,:]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])
    x_test=np.array(x_test)
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        

    new_df=nifty_quote.filter(['Close'])
    last_60_days=new_df[-60:].values
    last_60_days_scaled=scaler.transform(last_60_days)
    X_test=[]
    X_test.append(last_60_days_scaled)
    X_test=np.array(X_test)


    pred_price=model.predict(X_test)
    pred_price=scaler.inverse_transform(pred_price)
    st.write("Predicted Price for", end_date ," is ",pred_price)
