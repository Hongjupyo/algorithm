# 09_1 카피
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, r2_score
import time
from xgboost import XGBRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    # stratify=y
)   # stratify = 데이터를 균형있게 잘라라

# print(y_train)
# print(y_test)

# exit()
kfold = KFold(n_splits=5, shuffle=True, random_state=713)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# exit()
# 2. 모델
parmeters = [
    {'n_estimators':[100,200,300,400], "max_depth":[6,8,10], 'learning_rate':[0.1,0.01,0.001]},    # 18
    {'max_depth':[6,8,10], 'learning_rate':[0.1,0.01,0.001]},    # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1,0.01,0.001]},  # 12 
]   # 18+12+12 = 42

model = GridSearchCV(XGBRegressor(), parmeters, cv=kfold, verbose=1,)

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
            #  colsample_bylevel=None, colsample_bynode=None,
            #  colsample_bytree=None, device=None, early_stopping_rounds=None,
            #  enable_categorical=False, eval_metric=None, feature_types=None,
            #  feature_weights=None, gamma=None, grow_policy=None,
            #  importance_type=None, interaction_constraints=None,
            #  learning_rate=0.1, max_bin=None, max_cat_threshold=None,
            #  max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
            #  max_leaves=None, min_child_weight=None, missing=nan,
            #  monotone_constraints=None, multi_strategy=None, n_estimators=200,
            #  n_jobs=None, num_parallel_tree=None, ...)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}

# 4. 평가,예측
print('best_score : ', model.best_score_)
# best_score :  0.8444747963411523

print('model.score : ', model.score(x_test,y_test))
# model.score :  0.8410143781697635

y_predict = model.predict(x_test)
print("r2_score : ", r2_score(y_test, y_predict))
# r2_score :  0.8410143781697635

print("걸린시간 : ", round(end_time-start_time,2), '초')
# 걸린시간 :  105.92 초