from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

# 2. 모델
# model = DecisionTreeRegressor()
model = RandomForestRegressor(max_depth=5, max_leaf_nodes=3, n_estimators=100, max_features=33,)

# 3. 컴파일, 훈련
# 4. 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, n_jobs=-1)

print('ACC : ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores),4)
      )