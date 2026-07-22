import numpy as np
from sklearn.datasets import load_diabetes

x,y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (20640, 8) (20640,)

# 2. 모델구성
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression # LogisticRegression은 Regressor이지만 분류모델
from sklearn.tree import DecisionTreeRegressor  # Regressor는 회귀모델 / Classifier는 분류모델
from sklearn.ensemble import RandomForestRegressor

# model = LinearSVC()  # 0.14027149321266968
# model = LogisticRegression()  # 0.020361990950226245
# model = DecisionTreeRegressor()   # 1.0
model  = RandomForestRegressor()    # 0.9192669292039708

# 3. 컴파일, 훈력
model.fit(x,y)

# 4. 평가, 예측
results = model.score(x,y)
print(results)
