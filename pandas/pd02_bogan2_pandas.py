import numpy as np
import pandas as pd

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10],
    [2, 4, np.nan, 8, np.nan],
    [2, 4, 6, 8, 10],
    [np.nan, 4, np.nan, 8, np.nan]
])
# print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data, data.shape) # (5, 4)

# 0. 결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

# 1. 결측치 삭제
print(data.dropna())
print(data.dropna(axis=0))
print(data.dropna(axis=1))

# 2. 특정갑값, 평균
means = data.mean() # 판다스에서는 알아서 자동으로 컬럼별로 평균해줘,
print(means)
data2 = data.fillna(means)
print(data2)

# 2-2. 특정값 - 중위값
med = data.median()
print(med)
data3 = data.fillna(med)
print(data3)

# 2-3. 특정값  - 0
data4 = data.fillna(0)
print(data4)

# 2-4. 특정값 - 777
data5 = data.fillna(777)
print(data5)

# 2-5. 특정값 ffil  시계열 잘먹혀, 첫번째가 없으면 nan
data6 = data.ffill()
print(data6)

# 2-6. 특정값 - bfill   시계열 잘먹혀, 마지막이 없으면 nan
data7 = data.bfill()
print(data7)

#############################################
med = data['x1'].median()
print(med)    # 6.5

mean = data['x4'].mean()
print(mean)    # 6.0

# 실습
"""
x1 : median
x2 : ffill
x3 : means
"""
data['x1'] = data['x1'].fillna(med)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(mean)
print(data)