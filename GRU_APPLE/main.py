import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense

# 전역변수 설정
timeStep = 10 # 10, 20, 40
hiddenState = 32 # 32, 64, 256

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
def transformData(data: [[float]]) -> (([float], [float]), ([float], [float])):
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

        return dataX, dataY

    (trainX, trainY) = parsingData(trainSet)
    (testX, testY) = parsingData(testSet)



transformData(apple)