# -*- coding: utf-8 -*-
# """
# Created on Thu Aug 13 11:52:56 2020

# @author: Rial


import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers import Dropout
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#get GHI data
df=df=pd.read_excel(r'Data_Preparation.xlsx')
df=df[['date','GHI_Average']]
# df['date'] = pd.to_datetime(df['date'])
# df['hour'] = df['date'].dt.hour
# df=df[(df.hour >= 7) & (df.hour <= 19)]
df=df[['date','GHI_Average']]
df=df.set_index('date')
df=df[:8760]
#number rows and columns
shape=df.shape

#plot graph
plt.figure(figsize=(16,8))
plt.title('GHI_2018')
plt.plot(df['GHI_Average'])
plt.xlabel('date', fontsize=16)
plt.ylabel('GHI_2018')

#convert data to numpy
data=df.values

#number of rows to train (80%)
training_data_len=math.ceil(len(data)*.8)


#scaled the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data)

#training data set
#scaled training data set
train_data=scaled_data[0:training_data_len, :]
#split the data into x_train and y_train data sets
x_train=[]
y_train=[]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()
     
#convert x train y train to numpy
x_train, y_train=np.array(x_train), np.array(y_train)

#reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

#build the LSTM model
# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, activation='relu',return_sequences = True, input_shape = (x_train.shape[1], 1)))
# model.add(Dropout(0.2))

# # Adding a second LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))

# # Adding a third LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
# model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 25))    
model.add(Dense(units = 1))     

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train,y_train,batch_size=10, epochs=100)

#create the testing data set
#create a new array containing scaled values from index 10424 to 10484
test_data=scaled_data[training_data_len-60:,:]

#create the data sets x_tes and y_test
x_test=[]
y_test=data[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
#convert data to numpy
x_test=np.array(x_test)

#reshape
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#get the model predicted ghi
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

#get the rmse
rmse=np.sqrt(np.mean(predictions-y_test)**2)

#plot the ddata
train=df[:training_data_len]
valid=df[training_data_len:]
valid['Predictions']=predictions
train_data2=df[:training_data_len]
train_data2['Predictions']=np.nan
all_data=train_data2.append(valid)

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('date',fontsize=14)
plt.ylabel('GHI',fontsize=14)
plt.plot(train['GHI_Average'])
plt.plot(all_data[['GHI_Average','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.scatter(valid['GHI_Average'],valid['Predictions'],marker='o')


from sklearn.metrics import r2_score

coefficient_of_dermination = r2_score(valid['GHI_Average'],valid['Predictions'])


# #predict future
# from keras.preprocessing.sequence import TimeseriesGenerator

# train=df
# scaler.fit(train)
# train=scaler.transform(train)

# n_input=60
# n_features=1

# generator=TimeseriesGenerator(train,train,length=n_input, batch_size=6)
# model.fit_generator(generator, epochs=10)

# pred_list=[]
# batch=train[-n_input:].reshape((1,n_input,n_features))

# for i in range(n_input):
#     pred_list.append(model.predict(batch)[0])
#     batch=np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
    
# from pandas.tseries.offsets import DateOffset
# add_dates=[df.index[-1]+DateOffset(hours=x) for x in range(0,61)]
# future_dates=pd.DataFrame(index=add_dates[1:],columns=df.columns)


# df_predict=pd.DataFrame(scaler.inverse_transform(pred_list),
#                         index=future_dates[-n_input:].index,columns=['Prediction'])

# df_proj=pd.concat([df,df_predict],axis=1)

# plt.figure(figsize=(10,4))
# plt.plot(df_proj.index,df_proj['GHI_Average'])
# plt.plot(df_proj.index,df_proj['Prediction'], color='r')
# plt.legend(loc='best',fontsize='large')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()



    
