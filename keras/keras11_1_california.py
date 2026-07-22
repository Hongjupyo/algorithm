from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import fetch_california_housing

# 1. 데이터
datasets = fetch_california_housing()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
# 'Population', 'AveOccup', 'Latitude', 'Longitude']

x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
                                            x, y, 
                                            train_size=0.85,
                                            random_state=27,
                                            shuffle=True,
                                                    )
print(x_train.shape, x_test.shape)  # (15480, 8) (5160, 8)
print(y_train.shape, y_test.shape)  # (15480,) (5160,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_shape=(8, )))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(101))
model.add(Dense(333))
model.add(Dense(101))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=16)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)

"""

"""
