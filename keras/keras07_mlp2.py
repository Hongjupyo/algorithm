import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터  # 두개이상은 리스트 
x = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]]) # (2,6)
# x = x.T  # Transpose
x = x.transpose
# x = np.array([[1,7],[2,8],[3,9],[4,10],[5,11],[6,12]]) # (6,2)
print(x.shape)  # (6,2)
# exit()
y = np.array([1,2,3,4,5,6]) # (6,)

# 2. 모델구성    # y=ax+b  # 노드와 파라미터 레이어 추가
model = Sequential()
# model.add(Dense(6, input_dim=2))  # 열 = 컬럼 = 피처 = 특성 = 속성 = 어트리뷰트
model.add(Dense(6, input_shape=(2, )))  # 열 = 컬럼 = 피처 = 특성 = 속성 = 어트리뷰트
model.add(Dense(10))
model.add(Dense(27))
model.add(Dense(51))
model.add(Dense(333))
model.add(Dense(55))
model.add(Dense(11))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련  # 훈련할때 batch 개수 끊어서
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

# 4. 평가, 예측   # 최소의 Loss의 값을 구하기 위해  # 최적의 W 값을 구한다
loss = model.evaluate(x, y)
print("loss = ", loss)
result = model.predict(np.array([[7, 13]]))
print("7의 예측값 : ", result)