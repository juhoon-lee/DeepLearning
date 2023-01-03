import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense

# 전역변수 설정
timeStep = 10 # 10, 20, 40
hiddenState = 32 # 32, 64, 256
graphTitle = "Default test"

# Pandas Setting
pd.set_option('display.max_rows', None) # row 생략 없이 출력
pd.set_option('display.max_columns', None) # col 생략 없이 출력

# 데이터 불러오기
apple = pd.read_csv("Apple_5Y.csv")
# apple = pd.read_csv("Apple_10Y.csv")
# apple = pd.read_csv("Apple_Whole_Period.csv")

# Describe 출력
# print(apple.describe())

# trainData, testData 가공
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

# 데이터 분리
trainX, trainY, testX, testY = transformData(apple)

# 모델 구현
model = Sequential()
model.add(
    GRU(
        units = 32,
        input_length = trainX.shape[1],
        input_dim = trainX.shape[2],
        activation = "tanh"
    )
)
model.add(Dense(6))
model.summary()

# 모델 컴파일
model.compile(
    loss = 'mse',
    optimizer = 'adam',
    metrics = ["mae"]
)

# 모델 훈련
fitStartTime = time.time()
model.fit(
    trainX,
    trainY,
    epochs = 100,
    batch_size = 32
)
fitEndTime = time.time()

# 시간 및 평가 기록
fitTime = fitEndTime - fitStartTime
score = model.evaluate(testX, testY)

f = open(f"Training_Result.txt", "w")
f.write(f"모델 학습 시간: {fitTime:.3} sec\n평가 손실: {score[0]}")
f.close()

# 예측
prediction = model.predict(testX)
adjClose = prediction[:, 4]

# 그래프
plt.title(graphTitle)
plt.plot(testY[:, 4], label="Test ADJ.Close")
plt.plot(adjClose, label="Predict ADJ.Close")
plt.grid(True)
plt.legend()
plt.show()

# multiStep 예측