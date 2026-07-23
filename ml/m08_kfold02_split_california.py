from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    # stratify=y
)   # stratify = 데이터를 균형있게 잘라라

# print(y_train)
# print(y_test)

# exit()
kfold = KFold(n_splits=5, shuffle=True, random_state=123)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# exit()
# 2. 모델
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

# 3. 컴파일, 훈련
# 4. 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, n_jobs=-1)

print('ACC : ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores),4)
      )