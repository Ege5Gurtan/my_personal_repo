from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
                 na_values=['NA','?'])

cars = df['name']
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

x = df[['cylinders','displacement','horsepower','weight','acceleration','year','origin']].values
y = df['mpg'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

model =Sequential()
model.add(Dense(25,input_dim=x.shape[1],activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

monitor = EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=5,verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=1)

pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(score)









