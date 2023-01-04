#!/usr/bin/env python
# coding: utf-8
import keras.losses
# #### 패키지 Import

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense


# #### 전역변수 설정

# In[24]:


trainGraphTitle = "Train Data"
testGraphTitle = "Test Data"

timeStep = 10 # input_length 10, 20, 40
hiddenState = 32 # units: 32, 64, 256
activation = "tanh" # tanh, sigmoid
epochs = 100 # 50 100 200
batchSize = 32 # 32 64 256
dataSeYear = 5 # 5, 10, 40
loss = "mse" # mse, rmse
optimizer = "adam" # adam, sgd


# #### Pandas Setting

# In[25]:


pd.set_option('display.max_rows', None) # row 생략 없이 출력
pd.set_option('display.max_columns', None) # col 생략 없이 출력


# #### Data Load

# In[26]:


apple = pd.read_csv("Apple_5Y.csv")

if dataSeYear == 10:
    apple = pd.read_csv("Apple_10Y.csv")
elif dataSeYear == 40:
    apple = pd.read_csv("Apple_Whole_Period.csv")


# #### Describe 확인

# In[27]:


print(apple.describe())


# #### trainData, testData 가공하는 함수

# In[28]:


def transformData(data: [[float]]) -> ([[[float]]], [[float]], [[[float]]], [[float]]):
    # 날짜 제외
    data = data.drop(columns=["Date"])

    # 데이터 0~1로 정규화
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(data)
    scaledData = sc.transform(data)

    # 데이터 분리 test: 200개와 나머지
    trainSet = scaledData[ : -200]
    testSet = scaledData[-200 : ]

    # trainX, trainY, testX, testY 분리
    def parsingData(dataSet: [[float]]) -> ([[[float]]], [[float]]):
        dataX, dataY = [], []
        for index in range(len(dataSet) - timeStep):
            temp = []
            for step in range(timeStep):
                temp.append(dataSet[index + step])
            dataX.append(temp)
            dataY.append(dataSet[index + timeStep])

        return np.array(dataX), np.array(dataY)

    trainDataX, trainDataY = parsingData(trainSet)
    testDataX, testDataY = parsingData(testSet)

    return trainDataX, trainDataY, testDataX, testDataY


# #### Data Parsing

# In[29]:


trainX, trainY, testX, testY = transformData(apple)


# #### Model Implementation

# In[30]:


model = Sequential()
model.add(
    GRU(
        units = hiddenState,
        input_length = trainX.shape[1],
        input_dim = trainX.shape[2],
        activation = activation
    )
)
model.add(Dense(6))
model.summary()


# #### Model Complie

# In[31]:

model.compile(
    loss = loss,
    optimizer = optimizer,
    metrics = ["mae"]
)


# #### Model Training

# In[48]:


fitStartTime = time.time()
history = model.fit(
    trainX,
    trainY,
    epochs = 100,
    batch_size = 32
)
fitEndTime = time.time()


# #### 시간 및 평가 기록

# In[43]:


fitTime = fitEndTime - fitStartTime
score = model.evaluate(testX, testY)

f = open(f"Training_Result.txt", "w")
f.write(f"모델 학습 시간: {fitTime:.3} sec\n평가 손실: {score[0]}")
f.close()


# In[44]:


f"모델 학습 시간: {fitTime:.3} sec"


# In[45]:


f"평가 손실: {score[0]}"


# #### 예측

# In[46]:


trainPrediction = model.predict(trainX)
testPrediction = model.predict(testX)


# #### 그래프

# #### Loss

# In[47]:


loss = history.history["loss"]
plt.title("Loss")
plt.plot(loss, label="loss")
plt.grid(True)
plt.legend()
plt.show()


# Train Data Graph

# In[37]:


plt.title(trainGraphTitle)
plt.plot(trainY[:, 4], label="Train ADJ.Close")
plt.plot(trainPrediction[:, 4], label="Train ADJ.Close")
plt.grid(True)
plt.legend()
plt.show()


# Test Data Graph

# In[38]:


plt.title(testGraphTitle)
plt.plot(testY[:, 4], label="Test ADJ.Close")
plt.plot(testPrediction[:, 4], label="Predict ADJ.Close")
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




