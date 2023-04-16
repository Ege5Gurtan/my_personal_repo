from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics


df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
                 na_values=['NA','?'])

cars = df['name']
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

x = df[['cylinders','displacement','horsepower','weight','acceleration','year','origin']].values
y = df['mpg'].values

model =Sequential()
model.add(Dense(25,input_dim=x.shape[1],activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x,y,verbose=2,epochs=100)

pred = model.predict(x)
score = np.sqrt(metrics.mean_squared_error(pred,y))
print(score)

model_json = model.to_json()
with open(os.path.join("network.json"),"w") as json_file:
    json_file.write(model_json)

# model_yaml = model.to_yaml()
# with open(os.path.join("network.yaml"),"w") as yaml_file:
#     yaml_file.write(model_yaml)

model.save(os.path.join("network.h5"))









