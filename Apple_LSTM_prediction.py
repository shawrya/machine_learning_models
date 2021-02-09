

#importing libraries
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt

#printing dataset
df=pd.read_excel("C:\python_files\stock\data\AAPL.xlsx")
print(df)

#visualizing close values
plt.figure(figsize=(20,8))
plt.plot(df.close)
plt.show()

#filtering close values
data=df.filter(['close'])
dataset = data.values

#scaling the values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

training_data_len = math.ceil(len(dataset)*0.8)
print(training_data_len)

#dividing the dataset into testing and training dataset

#training dataset
train_data = scaled_data[0:training_data_len,:]
x_train = []
y_train = []
for i in range(70,len(train_data)):
    x_train.append(train_data[i-70:i,0])
    y_train.append(train_data[i,0])
    if i<=70:
        print(x_train)
        print(y_train)

#testing dataset
test_data = scaled_data[training_data_len-70:,:]
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(70,len(test_data)):
    x_test.append(test_data[i-70:i,0])

#converting arrays to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

#reshaping data into three dimensional data because LSTM model expects three dimensional data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

#initializing the model
model =Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(1))

#compiling the model
model.compile(optimizer="adam",loss="mean_squared_error")

#fitting the model according to the dataset
model.fit(x_train,y_train,batch_size=1,epochs=1)

#converting x_test into numpy format
x_test = np.array(x_test)
print(x_test.shape)

#reshaping data into three dimensional data because LSTM model expects three dimensional data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
print(x_test.shape)

#get model's predicted prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#evaluating root mean squared error
evel = np.sqrt(np.mean(predictions-y_test)**2)
print(evel)


#plotting the final data
train = data[:training_data_len]
valid = data[training_data_len:]
valid["prediction"] = predictions
plt.figure(figsize=(20,8))
plt.plot(train.close)
plt.plot(valid[["close","prediction"]])
plt.legend(["train","val","prediction"])
plt.show()
