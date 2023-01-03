import numpy as np
import pandas as pd

class stockDataSet:

    def samsungStockData(self) -> (([int], [int]), ([int], [int])):
        data = pd.read_csv("samsung.csv")  # 종가 index: 4
        stockData = data.values[:, 4]  # 주가 데이터
        stockData = self.__minMaxNormalize(stockData) # 정규화
        # stockData = self.__zScoreNormalize(stockData) # 표준화

        return self.__convertDataToTuple(stockData)

    def __convertDataToTuple(self, data: [int]) -> (([int], [int]), ([int], [int])):
        step = 5 # 5개씩 잘라 다음 6번째 주가 예측
        boundaryIndex: int = int(len(data)/3*2)
        trainData: [int] = data[:boundaryIndex]
        testData: [int] = data[boundaryIndex:]

        trainX, trainY = self.__parsingData(trainData, 5)
        testX, testY = self.__parsingData(testData, 5)

        return ((np.array(trainX), np.array(trainY)), (np.array(testX), testY))

        # return ((trainX, trainY), (testX, testY))

    # [1 2 3 4 5 6]이 있으면 [1,2,3,4,5]가 X, 6이 Y로 들어가고 Index 1씩 증가하면서 반복
    def __parsingData(self, data,  step) -> ([int], [int]):
        dataX = []
        dataY = []
        for index in range(len(data)-step):
            temp= []
            for step in range(5):
                temp.append(data[index+step])
            dataX.append(temp)
            dataY.append(data[index+5])

        return (dataX, dataY)

    def __minMaxNormalize(self, list):
        normalized = []
        print("최대 최소 값")
        print(max(list))
        print(min(list))
        for value in list:
            normalizedNum = (value - min(list)) / (max(list) - min(list))
            normalized.append(normalizedNum)

        return normalized

    def __zScoreNormalize(self, list):
        normalized = []
        for value in list:
            normalizedNum = (value - np.mean(list)) / np.std(list)
            normalized.append(normalizedNum)
        return normalized