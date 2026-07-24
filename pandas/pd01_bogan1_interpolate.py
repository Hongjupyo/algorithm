"""
결측치 처리
1. 삭제 - 행,, 또는 열.
2. 임의의값(특정값)
    - 0 : fillna
    - 평균 : mean (이상치 조심.) 
    - 중위: median
    - 앞값 : ffill
    - 뒷값 : bfill
    - 특정값 : 777 (조건 보고 넣음)
    - 기타 등등...
3. interpolate - 보간(알려진 데이터 점 접합의 범위 내에 새 데이터 점을 추가하는 기법)
4. 모델 : .predict (값을 예측해서), (가급적 다른모델 사용)
5. 부스팅 계열 모델 : 통상 이상치, 결측치에 대해 영향을 덜 받는다. (자유~! freedom~)
"""

import pandas as pd
import numpy as np

dates = [
    '24/7/2026', '25/7/2026', '26/7/2026',
    '27/7/2026', '28/7/2026', '29/7/2026',
]
dates = pd.to_datetime(dates)
print(dates)
# DatetimeIndex(['2026-07-24', '2026-07-25', '2026-07-26', '2026-07-27',
#                '2026-07-28', '2026-07-29'],
#               dtype='datetime64[us]', freq=None)
print("==========================================")
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan,], index=dates)
print(ts)
print("==========================================")
ts = ts.interpolate()
print(ts)