# 16 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_breast_cancer, load_iris
import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# 1. 데이터
datasets = load_iris()
# print(datasets)
# exit()
# print(datasets.DESCR)
# exit()
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# exit()
x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape, y.shape) # (150, 4) (150,)

# exit()
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50]))

########## onehot 01 판다스 이용 ##########
# y = pd.get_dummies(y)
# print(y)

########## onehot 02 Tensorflow꺼 이용 ##########
from tensorflow.keras.utils  import to_categorical
y = to_categorical(y)
print(y)

# exit()
x_train, x_test, y_train, y_test = train_test_split(
                                            x, y, 
                                            train_size=0.75,
                                            random_state=273,
                                            shuffle=True,
                                                    )
print(x_train.shape, x_test.shape)  # (112, 4) (38, 4)
print(y_train.shape, y_test.shape)  # (112,) (38,)
print(np.unique(y_train, return_counts=True))   # (array([0, 1, 2]), array([40, 33, 39]))
# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, activation='relu',input_shape=(4, )))
model.add(Dense(17, activation='relu'))
model.add(Dense(55, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(333, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(505, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='softmax')) 
# 회귀 최종 activation='liner'/ 분류 이진 최종 activation='sigmoid'/ 분류 다중 최종 activation='softmax'

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min', # 'auto'로 mode 주면 자동으로 min max 잡아줌
    patience=100,
    restore_best_weights=True,
)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'],
              ) # mse...등은 회귀 /분류의 이진에선 binary_crossentropy 유일/ 분류의 다중에선 categorical_crossentropy
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1,
          validation_split=0.1,
          callbacks=[es],
          )
end_time = time.time()

# verbose = 0 : 침묵(훈련과정을 보이지 않는다)
# verbose = 1 : 기본(훈련과정이 다 보인다)
# verbose = 2 : 프로그래스바 제거
# verbose = 3....나머지 : Epoch만 출력

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # loss는 binary_crossentropy가 나옴
print("loss : ", loss)

# exit()
y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
# exit()

print(y_test)
y_test = np.argmax(y_test, axis=1)
print(y_test)
# exit()

# y_predict = np.round(y_predict)
print("y_test의 원값 :", y_test, y_test.shape)
print("x_test의 예측값 : ", y_predict, y_predict.shape)

print("걸린시간 :", round(end_time-start_time,2), "초")

# exit()
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score" ,acc_score)

"""

"""