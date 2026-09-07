# Missing Person Location Prediction

학습 코드, 데이터, 결과물, 프론트 화면을 역할별로 분리한 프로젝트입니다. 위치 예측 지도 아래에는 지역 기상과 경과 시간을 활용한 OpenAI 실종 위험도 참고 분석이 표시됩니다.

## 빠른 실행

프로젝트 루트의 **.env** 파일에서 `OPENAI_API_KEY=` 뒤에 발급받은 키를 입력하고 **run_frontend.bat**를 실행합니다. API 키를 설정하지 않아도 기존 위치 예측은 실행되지만 위험도 분석은 표시되지 않습니다.

~~~dotenv
OPENAI_API_KEY=sk-여기에_발급받은_키
~~~

~~~text
http://127.0.0.1:5500
~~~

5500 포트가 다른 서버에서 사용 중이면 5501 이후의 빈 포트로 자동 전환되고 해당 주소가 브라우저에서 열립니다.

## 폴더 구조

~~~text
project
├─ README.md
├─ run_frontend.bat
├─ server.py
├─ frontend
│  ├─ index.html
│  ├─ css
│  │  └─ styles.css
│  ├─ js
│  │  ├─ app.js
│  │  └─ map-view.js
│  └─ model
│     ├─ model.json
│     ├─ group1-shard1of1.bin
│     └─ preprocessing.json
├─ training
│  ├─ train_model.py
│  ├─ export_frontend_model.py
│  ├─ requirements.txt
│  ├─ data
│  │  ├─ 실종_AI_데이터셋_X_300000건.csv
│  │  └─ 실종_AI_데이터셋_Y_300000건.csv
│  └─ artifacts
│     ├─ models
│     ├─ preprocessors
│     └─ reports
└─ docs
   ├─ MODEL_GUIDE.md
   └─ FRONTEND_README.md
~~~

## 역할

- **frontend**: 브라우저 테스트 화면과 TensorFlow.js 위치 예측 모델
- **server.py**: 프런트 제공, Open-Meteo 기상 조회, 서버 측 OpenAI Responses API 호출
- **training**: Python 학습 코드, 원본 데이터, 학습 결과
- **docs**: 사용법과 모델 파일 설명
- **run_frontend.bat**: 로컬 서버와 웹 화면 실행

## 모델을 다시 학습할 때

1. **training/train_model.py**를 실행합니다.
2. 결과는 **training/artifacts** 아래 역할별 폴더에 저장됩니다.
3. **training/export_frontend_model.py**를 실행해 브라우저 모델을 갱신합니다.
4. **run_frontend.bat**로 화면을 확인합니다.
