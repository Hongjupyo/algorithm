# 11_2를 es 적용해서 성능향상해볼것
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
import time
from tensorflow.keras.callbacks import EarlyStopping

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

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, activation='relu', input_shape=(10, )))
model.add(Dense(18, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(310, activation='relu'))
model.add(Dense(111, activation='relu'))
model.add(Dense(5050, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=10000000000000000,
          batch_size=6, validation_split=0.2,
          callbacks=[es]    # EarlyStopping 적용
          )  # validation_split=0.2 훈련한 데이터의 20%
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)

print("걸린시간 : ", round(end_time - start_time, 2), "초")  # 모델 학습 시간에 걸린시간 측정

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

r2_score :  0.6216994506842788
"""
