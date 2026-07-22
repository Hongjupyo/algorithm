import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터  # 두개이상은 리스트 
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

# 2. 모델구성    # y=ax+b  # 노드와 파라미터 레이어 추가
model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(10))               # keras03_deep1에서 있던 히든부분 input_dim 생략 
model.add(Dense(27))
model.add(Dense(501))
model.add(Dense(104))
model.add(Dense(333))
model.add(Dense(55))
model.add(Dense(11))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

# 4. 평가, 예측
result = model.predict(np.array([7]))
print("7의 예측값 : ", result)