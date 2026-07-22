# 23_1 카피
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
import pandas as pd
import time 

# 1. 데이터
path = "./_data/"

train_csv = pd.read_csv(path+"train.csv", index_col=0)  # "./data/train.csv"
test_csv = pd.read_csv(path+"test.csv", index_col=0)   # "./data/test.csv"
submit_csv = pd.read_csv(path+"sampleSubmission.csv", index_col=0)   # "./data/sampleSubmission.csv"
# print(train_csv)
# print(train_csv.shape)  # (10886, 11)
# print(test_csv)
# print(test_csv.shape)   # (6493, 8)
# print(submit_csv)
# print(submit_csv.shape)   # (6493, 1)
print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
# 'humidity', 'windspeed', 'casual', 'registered', 'count'],

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)

y = train_csv['count']
print(y)
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
                                                x,y,
                                                train_size=0.8,
                                                random_state=731,
                                                shuffle=True,
                                                )
print(x_train.shape, x_test.shape)  # (8164, 8) (2722, 8)
print(y_train.shape, y_test.shape)  # (8164,) (2722,)
# exit()

# 2. 모델 구성


# 3. 컴파일, 훈련
#######################################################
model = load_model('./_save/keras23_mcp_1.keras')
#######################################################

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

rmse = root_mean_squared_error(y_test, y_predict)
print("rmse : ", rmse)  # rmse :  154.3572540283203

########## CSV 파일 만들기 ##########
y_submit = model.predict(test_csv)
# print(y_submit)
submit_csv['count'] = y_submit
# print(submit_csv)
submit_csv.to_csv(path + "submission_0717_1453.csv")    # CSV 파일로 생성

# print("걸린시간 : ", round(end_time - start_time, 2), "초")  # 모델 학습 시간에 걸린시간 측정
