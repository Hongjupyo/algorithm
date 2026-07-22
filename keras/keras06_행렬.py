import numpy as np     # 0차원 = 스칼라 / 1차원 = 백터 / 2차원 = 행렬 / 3차원 = 텐서 / 4차원 = 4차원텐서

# 열 = 컬럼 = 픽처 = 특성 = 속성 = 어트리뷰트
# 행무시 , 열우선

x1 = np.array([1,2,3])
print("x1 = ", x1.shape)  # x1 =  (3,)    # 백터

x2 = np.array([[1,2,3]])
print("x2 = ", x2.shape)  # x2 =  (1, 3)  # 행렬

x3 = np.array([[1,2,3], [4,5,6]])
print("x3 = ", x3.shape)  # x3 =  (2, 3)


x4 = np.array([[1,2],[3,4]])
print("x4 = ", x4.shape)  # x4 =  (2, 2)

x5 = np.array([[[[1]]],[[[2]]]])
print("x5 = ", x5.shape)  # x5 =  (2, 1, 1, 1)
