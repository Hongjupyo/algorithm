import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터  # 두개이상은 리스트 
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]]).T
# x = x.T  # Transpose
# x = x.transpose
print(x.shape) # (10,3)
# exit()

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [9,8,7,6,5,4,3,2,1,0]]).T 
print(y.shape) # (10,2)

# 2. 모델구성    # y=ax+b  # 노드와 파라미터 레이어 추가
model = Sequential()
model.add(Dense(6, input_shape=(3, )))  # 열 = 컬럼 = 피처 = 특성 = 속성 = 어트리뷰트
model.add(Dense(10))
model.add(Dense(27))
model.add(Dense(51))
model.add(Dense(333))
model.add(Dense(55))
model.add(Dense(11))
model.add(Dense(5))
model.add(Dense(2))

# 3. 컴파일, 훈련  # 훈련할때 batch 개수 끊어서
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=4)

# 4. 평가, 예측   # 최소의 Loss의 값을 구하기 위해  # 최적의 W 값을 구한다
loss = model.evaluate(x, y)
print("loss = ", loss)
result = model.predict(np.array([[10, 1.3, 0]]))
print("10, 1.3, 0의 예측값 : ", result)