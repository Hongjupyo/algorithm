# 26_1 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

# 1. 데이터
datasets = load_diabetes()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1',
# 's2', 's3', 's4', 's5', 's6']


x = datasets.data
y = datasets.target
# print(x)
# print(y)

print(x.shape, y.shape) # (442, 10) (442,)


x_train, x_test, y_train, y_test = train_test_split(
                                            x, y, 
                                            train_size=0.9,
                                            random_state=31,
                                            shuffle=True,
                                                    )
print(x_train.shape, x_test.shape)  # (331, 10) (111, 10)
print(y_train.shape, y_test.shape)  # (331,) (111,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

# exit()
# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_shape=(10, )))
model.add(Dense(18))
model.add(Dense(30))
model.add(Dense(120))
model.add(Dense(310))
model.add(Dense(111))
model.add(Dense(5050))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=6)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)

"""
train_size=0.9,
random_state=31,

model.add(Dense(3, input_shape=(10, )))
model.add(Dense(18))
model.add(Dense(30))
model.add(Dense(120))
model.add(Dense(310))
model.add(Dense(111))
model.add(Dense(5050))
model.add(Dense(7))
model.add(Dense(1))


epochs=100, batch_size=6

r2_score :  0.6267417964184047
"""