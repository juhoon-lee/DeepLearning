### sigmoid를 사용하면 상당히 느리다.

> exp연산을 사용함으로 느림  
> 또한 함수의 중심이 0이 아님으로 학습이 느려지는 것! -> 가중치의 미분 값이 모두 같은 부호를 가져 같은 방향으로의 업데이트는 학습을 지그재그 형태로 만든다. [참조 링크](https://stats.stackexchange.com/questions/237169/why-are-non-zero-centered-activation-functions-a-problem-in-backpropagation)
>
> 때문에 실제로는 거의 사용하지 않음

### 가장 좋은 결과들을 가져온 파라미터들을 대입

### 수정한 것들

- 30일 예측
- 역 정규화
- Early Stopping
- Best Parameter
