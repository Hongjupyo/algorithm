# 09_1 카피
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import r2_score
import time
from xgboost import XGBRegressor

# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    stratify=y
)   # stratify = 데이터를 균형있게 잘라라

# print(y_train)
# print(y_test)

# exit()
# kfold = KFold(n_splits=5, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# exit()
# 2. 모델
parmeters = [
    {"C":[1,10,100,1000],"kernel":['linear','sigmoid'], "degree":[3,4,5]},  #24
    {"C":[1,10,100],"kernel":['rbf'],'gamma':[0.001,0.0001]},    # 6
    {"C":[1,10,100,1000],"kernel":['sigmoid'], "gamma":[0.01,0.001,0.0001], "degree":[3,4,5]},    # 36    
]   # 24+6+16 = 66

model = RandomizedSearchCV(XGBRegressor(), parmeters, cv=kfold, verbose=1,)

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

# 4. 평가,예측
print('best_score : ', model.best_score_)
# best_score :  0.9397978544235229

print('model.score : ', model.score(x_test,y_test))
# model.score :  0.8496350646018982

y_predict = model.predict(x_test)
print("r2_score : ", r2_score(y_test, y_predict))
# r2_score :  0.8496350646018982

print("걸린시간 : ", round(end_time-start_time,2), '초')
# 걸린시간 :  0.8 초
