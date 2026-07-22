import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터  # 두개이상은 리스트 
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,     # train_size, test_size를 명시하지 않을 경우 7.5:2.5가 디폴트 값 
                                                    test_size=0.2,
                                                    # shuffle=False,    # shuffle은 기본 값이 True
                                                    random_state=371,                 
                                                    )

print(x_train, x_test)
print(y_train, y_test)

print(x_train.shape, x_test.shape) # (7,) (3,)
print(y_train.shape, y_test.shape) # (7,) (3,)

# exit()

# 2. 모델구성    # y=ax+b  # 노드와 파라미터 레이어 추가
model = Sequential()
model.add(Dense(6, input_shape=(1, )))  # (None, 1) # 열 = 컬럼 = 피처 = 특성 = 속성 = 어트리뷰트
model.add(Dense(10))
model.add(Dense(27))
model.add(Dense(51))
model.add(Dense(333))
model.add(Dense(55))
model.add(Dense(11))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련  # 훈련할때 batch 개수 끊어서 # 인공지능은 1. 회귀모델(현재까지 이용) 2. 분류모델
model.compile(loss='mse', optimizer='adam')     # loss = 오차 = 에러 = cost 동일    # mse = Mean Squared Error  제곱의 문제를 해결하기 위해 Rmse (root)
model.fit(x_train, y_train, epochs=200, batch_size=5)

# 4. 평가, 예측   # 최소의 Loss의 값을 구하기 위해  # 최적의 W 값을 구한다
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)      # loss =  31.398176193237305(random_state=8)

y_predict = model.predict([[x_test]])   # 100번째 w를 이용해서 예측값=wx+b
print("y_test의 원값 : ", y_test)
print("x_test의 예측값 : ", y_predict)

from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)

rmse = root_mean_squared_error(y_test, y_predict)
print("rmse : ", rmse)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)