import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]
             [9,8,7,6,5,4,3,2,1,0]).T
# x = x.T

y = np.array([1,2,3,4,5,6,7,8,9,10])

# 모델
model = Sequential()
model.add(Dense(6, input_shape=3))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(101))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=100, batch_size=5)

# 평가 예측
loss = model.evaluate(x,y)
print("loss= ", loss)

result = model.predict(np.array([[10, 1.3, 0]]))
print("10, 1.3, 0의 예측값 : ", result)