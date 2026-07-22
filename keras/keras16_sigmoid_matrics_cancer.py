# 12 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_breast_cancer
import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

# exit()
x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape, y.shape) # (569, 30) (569,)

print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357]))

# exit()
x_train, x_test, y_train, y_test = train_test_split(
                                            x, y, 
                                            train_size=0.8,
                                            random_state=101,
                                            shuffle=True,
                                                    )
print(x_train.shape, x_test.shape)  # (426, 30) (143, 30)
print(y_train.shape, y_test.shape)  # (426,) (143,)

# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, activation='relu',input_shape=(30, )))
model.add(Dense(17, activation='relu'))
model.add(Dense(55, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(333, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(505, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 회귀 최종 activation='liner'/ 분류 이진 최종 activation='sigmoid'

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min', # 'auto'로 mode 주면 자동으로 min max 잡아줌
    patience=100,
    restore_best_weights=True,
)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'],
              ) # mse...등은 회귀 /분류의 이진에선 binary_crossentropy 유일
start_time = time.time()
model.fit(x_train, y_train, epochs=100000000, batch_size=16, verbose=1,
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

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
# print("y_test의 원값 :", y_test)
# print("x_test의 예측값 : ", y_predict)

print("걸린시간 :", round(end_time-start_time,2), "초")

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score" ,acc_score)

"""

"""