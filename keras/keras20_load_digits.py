# 18 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import accuracy_score


# 1. 데이터
datasets = load_digits()
# print(datasets)
# print(datasets.DESCR)
# exit()
print(datasets.feature_names)
# ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 
# 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 
# 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 
# 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 
# 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 
# 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 
# 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 
# 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']

# exit()
x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape, y.shape) # (1797, 64) (1797,)

print(np.unique(y, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# exit()s

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
                                            train_size=0.8,
                                            random_state=615,
                                            shuffle=True,
                                                    )
print(x_train.shape, x_test.shape)  # (112, 4) (38, 4)
print(y_train.shape, y_test.shape)  # (112,) (38,)
print(np.unique(y_train, return_counts=True))   # (array([0, 1, 2]), array([40, 33, 39]))
# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, activation='relu',input_shape=(64, )))
model.add(Dense(11, activation='relu'))
model.add(Dense(55, activation='relu'))
model.add(Dense(222, activation='relu'))
model.add(Dense(333, activation='relu'))
model.add(Dense(1111, activation='relu'))
model.add(Dense(555, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(10, activation='softmax')) 
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
model.fit(x_train, y_train, epochs=100000, batch_size=16, verbose=2,
          validation_split=0.2,
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
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# exit()

# print(y_test)
y_test = np.argmax(y_test, axis=1)
# print(y_test)
# exit()

# print("y_test의 원값 :", y_test, y_test.shape)
# print("x_test의 예측값 : ", y_predict, y_predict.shape)

print("걸린시간 :", round(end_time-start_time,2), "초")

# exit()
acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score" ,acc_score)

"""

"""