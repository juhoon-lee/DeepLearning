### sigmoid를 사용하면 상당히 느리다.

> exp연산을 사용함으로 느림  
> 또한 함수의 중심이 0이 아님으로 학습이 느려지는 것! -> 가중치의 미분 값이 모두 같은 부호를 가져 같은 방향으로의 업데이트는 학습을 지그재그 형태로 만든다. [참조 링크](https://stats.stackexchange.com/questions/237169/why-are-non-zero-centered-activation-functions-a-problem-in-backpropagation)
>
> 때문에 실제로는 거의 사용하지 않음

### Additional

- 30일 예측
- 역 정규화
- Early Stopping
- Best Parameter

## 가설 및 검증

### Layer

가설: Layer가 많아지면 더 정확한 결과를 내거나 과적합 양상을 보일 것이라 생각했고 Double까지는 적절하게 나오지만 Triple은 과적합의 양상을 띄지 않을까 생
검증: Single Layer가 테스트데이터에 대해 가장 정확하게 나왔다. 하지만 예측값에 대해선 Triple이 가장 높았다.
