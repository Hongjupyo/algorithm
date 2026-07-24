import pandas as pd
print(pd.__version__)   # 3.0.3

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500']
]
index = ['031', '059', '033', '045', '023']
colums = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=colums)
print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

# print(data[0][1])   # 파이썬은 이렇게 (행렬순으로) 잘나오는데 판다스는......
# print(df[0])    # KeyError(key) from err
# print(df['031'])    # KeyError(key) from err
print(df['종목명']) # ★★★"판다스 열행"★★★ 컬럼,피처,열이 기준.

########## 아모레 시가 출력 ##########
# print(df[3,1])    # 에러
# print(df[3][1])   # 에러
# print(df['045','시가'])   # 에러
# print(df['045']['시가'])  # 에러
# print(df[['045']['시가']])    # 행렬 방식은 안된다!! 에러다!

# print(df['시가', '045'])  # 에러
print(df['시가']['045'])    # ★★★"판다스 열행"★★★
# print(df[1, 3])   # 에러

##############################################################
# loc : 인덱스 기준으로 행 데이터 추출
# iloc : 행번호 기준으로 행 데이터 추출
#        int loc로 외워라.
##############################################################
print(df)
print("=============================================")

print(df.iloc[3])
# print(df.iloc["045"])   # 에러
# print(df.loc[3])    # 에러
print(df.loc["045"])

print("네이버 뽑기")
print(df.iloc[4])
print(df.loc["023"])

print("아모레 시가뽑기")
print(df.iloc[3, 1])    # 3500
# print(df.iloc[3][1])    # 에러
# print(df.iloc[3, '시가'])   # 에러
print(df.iloc[3]['시가'])   # 3500
print(df.iloc[3].iloc[1])   # 3500
print(df.iloc[3].loc['시가'])   #3500

print(df.loc['045'].loc['시가'])    # 3500
print(df.loc['045'].iloc[1])    # 3500