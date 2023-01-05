#!/usr/bin/env python
# coding: utf-8

# #### 패키지 Import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


# #### 변수 설정

# In[2]:


trainGraphTitle = "Train Data"
testGraphTitle = "Test Data"
checkpointPath = "./weightCheckpoint/bestParam.hdf5"
loss = "mse" 

title = "ETC"
detail = "Test"
resultComment = f"{title} - {detail}"
fileSavePath = f"./result/{title}/{detail}"

depth = "SingleGRU" # Default
# depth = "DoubleGRU" 
# depth = "TripleGRU"

# 변수 "" 가 Default
hiddenState = 32 # units: 16 "32" 64
timeStep = 20 # input_length 10 "20" 40
activation = "tanh" # "tanh" sigmoid
epochs = 100 # 50 "100" 200
batchSize = 64 # 32 "64" 256
dataSize = 10 # 5 "10" 40
optimizer = "adam" # "adam" sgd
patience = 30 # 10 "30" 50


# #### Pandas Setting

# In[3]:


pd.set_option('display.max_rows', None) # row 생략 없이 출력
pd.set_option('display.max_columns', None) # col 생략 없이 출력


# #### Data Load

# In[4]:


apple = pd.read_csv("Apple_5Y.csv")

if dataSize == 10:
    apple = pd.read_csv("Apple_10Y.csv")
elif dataSize == 40:
    apple = pd.read_csv("Apple_Whole_Period.csv")


# #### Describe 확인

# In[5]:


apple.describe()


# #### trainData, testData 가공

# In[6]:


def transformData(data: [[float]]):
    # 날짜 제외
    data = data.drop(columns=["Date"])

    # 데이터 분리 test: 200개와 나머지
    trainSet = data[ : -200]
    testSet = data[-200 : ]
    
    # 데이터 0~1로 정규화
    sc = MinMaxScaler(feature_range=(0, 1)) 
    sc.fit(trainSet)
    trainSet = sc.transform(trainSet)
    testSet = sc.transform(testSet)
    
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
    
    
    return trainDataX, trainDataY, testDataX[:-30], testDataY, sc


# #### Data Parsing

# In[7]:


trainX, trainY, testX, testY, scaler = transformData(apple)


# #### Model Implementation

# In[8]:


model = Sequential()

if depth == "SingleGRU":
    model.add(
    GRU(
        units = hiddenState,
        input_length = trainX.shape[1],
        input_dim = trainX.shape[2],
        activation = activation
        )
    )
    model.add(Dense(6))
    
    
    
    
elif depth == "DoubleGRU":
    model.add(
        GRU(
            units = hiddenState,
            input_length = trainX.shape[1],
            input_dim = trainX.shape[2],
            activation = activation,
            return_sequences = True
        )
    )
    model.add(
        GRU(
            units = hiddenState,
            activation = activation
        )
    )
    
    model.add(Dense(6))
    
elif depth == "TripleGRU":
    model.add(
        GRU(
            units = hiddenState,
            input_length = trainX.shape[1],
            input_dim = trainX.shape[2],
            activation = activation,
            return_sequences = True
        )
    )
    model.add(
        GRU(
            units = hiddenState,
            activation = activation,
            return_sequences = True
        )
    )
    model.add(
        GRU(
            units = hiddenState,
            activation = activation
        )
    )
    
    model.add(Dense(6))

model.summary()


# #### Model Complie

# In[9]:


model.compile(
    loss = loss,
    optimizer = optimizer,
    metrics = ["mae"]
)


# #### Model Training

# In[10]:


earlyStop = EarlyStopping(
    monitor = 'loss',
    min_delta = 0.0001,
    patience = patience,
    verbose = 1
)
saveBest = ModelCheckpoint(
    filepath = checkpointPath,
    monitor = "loss",
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    mode = "auto",
    save_freq = "epoch"
)

fitStartTime = time.time()
history = model.fit(
    trainX,
    trainY,
    epochs = epochs,
    batch_size = batchSize,
    callbacks = [earlyStop, saveBest]
)
fitEndTime = time.time()


model.load_weights(checkpointPath)


# #### 이후 30일 예측

# In[11]:


thirtyPredict = testX

for _ in range(30):
    currentPredict = model.predict(thirtyPredict)
    recentPredict = np.reshape(currentPredict[-1], (1, currentPredict[-1].shape[0]))
    newPredictStep = thirtyPredict[-1, 1:]
    nextPredict = np.append(newPredictStep, recentPredict, axis = 0)
    nextPredict = np.reshape(nextPredict, (1, nextPredict.shape[0], nextPredict.shape[1]))
    thirtyPredict = np.append(thirtyPredict, nextPredict, axis = 0)


# #### 결과 기록

# In[12]:


fitTime = fitEndTime - fitStartTime
score = model.evaluate(testX, testY[:-30])
predictScore = model.evaluate(thirtyPredict[-30:], testY[-30:])
parameters = f"""
layer = {depth}
hiddenState = {hiddenState}
timeStep = {timeStep}
activation = {activation}
epochs = {epochs}
batchSize = {batchSize}
dataSetYear = {dataSize}
optimizer = {optimizer}
patience = {patience}
"""

# f = open("result.txt", "a")
# f.write(f"{resultComment}\n모델 학습 시간: {fitTime:.3} sec\n평가 손실: {score[0]}\n30일 예측 손실: {predictScore[0]}\n{parameters}\n--------------------------------------\n\n")
# f.close()


# In[13]:


print(f"모델 학습 시간: {fitTime:.3} sec")


# In[14]:


print(f"평가 손실: {score[0]}")


# In[15]:


print(f"30일 예측 손실: {predictScore[0]}")


# #### Test Data 예측 및 역정규화

# In[16]:


trainPrediction = scaler.inverse_transform(model.predict(trainX))
testPrediction = scaler.inverse_transform(model.predict(testX))

inversingTrainY = scaler.inverse_transform(trainY)
inversingTestY = scaler.inverse_transform(testY)


# ### 그래프

# #### Loss

# In[17]:


loss = history.history["loss"]
plt.title("Loss")
plt.plot(loss, label="loss")
plt.grid(True)
# plt.savefig(f"{fileSavePath}/Loss.png")
plt.legend()
plt.show()


# #### Train Data Graph

# In[18]:


plt.title(trainGraphTitle)
plt.plot(inversingTrainY[:, 4], label="Train ADJ.Close")
plt.plot(trainPrediction[:, 4], label="Predict ADJ.Close")
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('Price')
# plt.savefig(f"{fileSavePath}/Train.png")
plt.legend()
plt.show()


# #### Test Data Graph

# In[19]:


# plt.title(testGraphTitle)
# plt.plot(inversingTestY[:, 4], label="Test ADJ.Close")
# plt.plot(testPrediction[:, 4], label="Predict ADJ.Close")
# plt.grid(True)
# plt.xlabel('Day')
# plt.ylabel('Price')
# plt.axvline(x=len(testY)-30, color='green', linestyle='-', linewidth=1)
# plt.legend()
# plt.savefig(f"{fileSavePath}/Test.png")
# plt.show()


# #### Test Data + 30일 예측 그래프

# In[20]:


thirtyDaysAfterpredict = scaler.inverse_transform(model.predict(thirtyPredict))

plt.title("30 Days After")
plt.plot(inversingTestY[:, 4], label="Test ADJ.Close")
plt.plot(thirtyDaysAfterpredict[:, 4], label="30 Days predict ADJ.Close")
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('Price')
plt.axvline(x=len(testY)-30, color='green', linestyle='-', linewidth=1)
plt.legend()
# plt.savefig(f"{fileSavePath}/30Predict.png")
plt.show()

