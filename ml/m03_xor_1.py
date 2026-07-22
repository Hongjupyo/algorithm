import numpy as np
from sklearn.svm import LinearSVC   # svm 단층 예전 모델
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])

print(x_data.shape, y_data.shape) # (4, 2) (4,)

# 2. 모델
# model = LinearSVC()
model = Perceptron()

# 3. 훈련
model.fit(x_data,y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data)

results = model.score(x_data,y_data)    # accuracy
print(results)

acc = accuracy_score(y_data, y_predict)
print("acc  : ", acc)