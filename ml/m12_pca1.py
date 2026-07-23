# 11_1 카피
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris, load_breast_cancer
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import time
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

pca = PCA(n_components=10)  # 컬럼이 너무 많을 때 압축시켜서 훈련한다.
x = pca.fit_transform(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    stratify=y
)   # stratify = 데이터를 균형있게 잘라라

print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape,y_test.shape)    # (114, 30) (114,)


# 2. 모델
parmeters = {
    "learning_rate":0.9,
    "max_depth":50,
    "n_estimators":100,
}   # 딕셔너리는 키벨류 

model = XGBClassifier(**parmeters)   # **는 딕셔너리를 불러와라/ *는 리스트 불러와라

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# 4. 평가,예측
print('model.score : ', model.score(x_test,y_test))
# model.score :  0.8357082683009049

y_predict = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_predict))
# acc_score :  0.8357082683009049

print("걸린시간 : ", round(end_time-start_time,2), '초')
# 걸린시간 :  0.2 초