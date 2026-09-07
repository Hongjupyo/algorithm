# 모델 프로젝트 안내

## 1. 프로젝트 구조

~~~text
project
├─ README.md
├─ run_frontend.bat
├─ frontend
│  ├─ index.html
│  ├─ css
│  │  └─ styles.css
│  ├─ js
│  │  └─ app.js
│  └─ model
│     ├─ model.json
│     ├─ group1-shard1of1.bin
│     └─ preprocessing.json
├─ training
│  ├─ train_model.py
│  ├─ export_frontend_model.py
│  ├─ requirements.txt
│  ├─ data
│  └─ artifacts
│     ├─ models
│     ├─ preprocessors
│     └─ reports
└─ docs
   ├─ MODEL_GUIDE.md
   └─ FRONTEND_README.md
~~~

## 2. 폴더별 역할

### frontend

브라우저 테스트에 필요한 파일만 보관합니다.

- **index.html**: 입력 및 결과 화면
- **css/styles.css**: 화면 디자인
- **js/app.js**: 전처리, 모델 실행, 지도 표시
- **model**: TensorFlow.js 모델과 브라우저용 전처리 정보

### training

모델 학습에 필요한 파일을 보관합니다.

- **train_model.py**: 모델 학습
- **export_frontend_model.py**: 학습 모델을 브라우저용으로 변환
- **data**: X/Y 원본 CSV
- **artifacts/models**: Keras 모델
- **artifacts/preprocessors**: 숫자·범주·목표값 전처리기
- **artifacts/reports**: 학습 이력, 그래프, 테스트 예측

### docs

실행 방법과 모델 설명 문서를 보관합니다.

## 3. 모든 값을 입력해야 하나요?

아닙니다.

- 화면을 열면 테스트용 기본값이 이미 입력되어 있습니다.
- 아는 값만 수정하면 됩니다.
- 모르는 숫자 값은 비워두면 학습 데이터의 평균값을 사용합니다.
- 모르는 선택 값은 **모름 / 선택 안 함**을 선택하면 됩니다.
- 마지막 목격 위도와 경도를 비워두면 학습 데이터의 평균 좌표를 사용합니다.

실제 의미 있는 테스트를 위해서는 마지막 목격 위도와 경도는 가능하면 입력하는 것이 좋습니다.

### 실종 장소 입력

1. **실종 장소**에 공공장소나 지역명을 입력합니다.
2. **장소 검색**을 누릅니다.
3. 검색 결과에서 정확한 장소를 선택합니다.
4. 위도와 경도가 자동으로 설정됩니다.
5. 필요할 때만 **자동 설정된 좌표 확인 또는 직접 수정**을 펼칩니다.

예측 후에는 예상 위도·경도뿐 아니라 해당 좌표와 가까운 지역명도 함께 표시됩니다.

공개 주소 검색 서비스를 사용하므로 실제 개인 주소나 민감한 정보는 입력하지 마세요.

## 4. 프론트 실행

프로젝트 루트에 있는 **run_frontend.bat**를 더블클릭합니다.

브라우저 주소:

~~~text
http://127.0.0.1:5500
~~~

종료할 때는 함께 열린 검은색 창을 닫습니다.

## 5. 모델을 다시 학습한 후

1. **training/train_model.py**를 실행합니다.
2. 새 결과가 **training/artifacts**에 저장됩니다.
3. **training/export_frontend_model.py**를 실행합니다.
4. 변환된 모델이 **frontend/model**에 저장됩니다.
5. 프론트 화면을 새로고침합니다.

## 6. 모델 파일 구분

### 브라우저에서 사용하는 파일

- **frontend/model/model.json**
- **frontend/model/group1-shard1of1.bin**
- **frontend/model/preprocessing.json**

### 학습용 파일

- **training/artifacts/models/best_model.keras**
- **training/artifacts/models/final_model.keras**
- **training/artifacts/preprocessors/numeric_scaler.joblib**
- **training/artifacts/preprocessors/categorical_encoder.joblib**
- **training/artifacts/preprocessors/target_scaler.joblib**

### 확인용 파일

- **training/artifacts/reports/training_history.csv**
- **training/artifacts/reports/training_history.png**
- **training/artifacts/reports/test_predictions.csv**
- **training/artifacts/reports/actual_vs_predicted.png**

## 7. 주의

이 모델의 결과는 프론트 연결과 모델 동작을 확인하기 위한 참고용 예측입니다. 실제 수색 판단이나 위치를 보장하지 않습니다.
