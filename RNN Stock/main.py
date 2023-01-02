import dataStorage as ds
import graph as gp
import RNNModel as rm

import time
import numpy as np

# 삼성전자 주가 데이터 받아오기
dataStorage = ds.stockDataSet()
(trainX, trainY), (testX, testY) = dataStorage.samsungStockData()

# 3D Tensor로 변환
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Model 구현
def runningModel(model):
    model.compile(loss='mse', optimizer='adam', metrics=["mae"])

    layerCount = len(model.layers)
    caseString = "SimpleRNN" if layerCount == 2 else "DeepRNN"

    # 학습 및 시간 기록
    fitStartTime = time.time()
    history = model.fit(trainX, trainY, epochs=100, batch_size=32)
    fitEndTime = time.time()

    f = open(f"./{caseString}/Model_Training_Time.txt", "w")
    f.write(f"{caseString} 모델 학습 시간: {fitEndTime - fitStartTime:.3} sec\n")
    f.close()

    # 그래프 출력 및 저장

    # Loss 그래프
    gp.showAndSaveGraph(
        title=f"{caseString} Loss",
        data=[(history.history["loss"], "loss")],
        fileName=f"./{caseString}/LossData.png"
    )

    # Train Data 그래프
    prediction = model.predict(trainX)
    gp.showAndSaveGraph(
        title=f"{caseString} Train Data",
        data=[(trainY, "Train Y"), (prediction, "Train Prediction")],
        fileName=f"./{caseString}/TrainData.png"
    )

    # Test Data 그래프
    prediction = model.predict(testX)
    gp.showAndSaveGraph(
        title=f"{caseString} Test Data",
        data=[(testY, "Test Y"), (prediction, "Test Prediction")],
        fileName=f"./{caseString}/TestData.png"
    )


models = [rm.simpleRNN(), rm.deepRNN()]

for model in models:
    runningModel(model)
