# 타이타닉 생존 예측 Flask 애플리케이션

이 프로젝트는 타이타닉 승객의 생존 여부를 예측하는 머신러닝 모델을 Flask 웹 애플리케이션으로 제공합니다. MLflow를 사용하여 모델 실험 관리 및 배포를 구현했습니다.

## 프로젝트 구성
```
titanic_flask/
│
├── data/                          # 데이터 디렉토리
│   ├── titanic_train.csv          # 원본 타이타닉 데이터셋
│   ├── preprocessed_features.csv  # 전처리된 특성 데이터
│   └── preprocessed_target.csv    # 전처리된 타겟(생존 여부) 데이터
│
├── models/                        # 저장된 모델 디렉토리
│   └── random_forest_model/       # 최적의 랜덤 포레스트 모델
│
├── notebook/                      # 주피터 노트북
│   └── preprocessing_notebook.ipynb # 데이터 탐색 및 전처리 노트북
│
├── src/                           # 소스 코드
│   ├── app.py                     # Flask 웹 애플리케이션
│   └── train_mlflow.py            # MLflow 기반 모델 학습 스크립트
│
├── mlruns/                        # MLflow 실험 결과 저장 디렉토리
└── mlartifacts/                   # MLflow 모델 아티팩트 저장 디렉토리
```

## 주요 기능
1. 데이터 전처리: 노트북을 통해 타이타닉 데이터셋을 탐색하고 전처리합니다.

  - 결측치 처리 (나이, 승선 항구 등)
  - 특성 엔지니어링 (가족 크기, 갑판 정보, 호칭 등)
  - 범주형 변수 인코딩
  - 특성 스케일링

2. 모델 학습: 다양한 머신러닝 알고리즘으로 타이타닉 생존 예측 모델을 학습하고 MLflow로 추적합니다.
  - 로지스틱 회귀
  - 서포트 벡터 머신(SVM)
  - K-최근접 이웃(KNN)
  - 의사결정 트리
  - 랜덤 포레스트

3.모델 배포: 최고 성능의 모델을 Flask 웹 API로 배포합니다.
  - REST API 엔드포인트를 통한 예측 제공
  - 정확도, 정밀도, 재현율, F1 점수, ROC AUC와 같은 성능 지표 활용

##설치 및 실행 방법
필요 사항
- Python 3.8 이상
- pip

설치

```
# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

MLflow 서버 실행
```
mlflow server --host 127.0.0.1 --port 5000
```

모델학습
```
python src/train_mlflow.py
```

Flask 앱 실행
```
python src/app.py
```

## API 사용법
생존 예측 API
생존 여부를 예측하려면 PassengerId 파라미터를 사용하여 GET 요청을 보냅니다:
```
GET http://localhost:8000/predict?PassengerId=1
```

응답예시
```
{
  "PassengerId": 1,
  "Probability": 0.12,
  "Prediction": "사망",
  "Survived": 0
}
```

## 기술 스택
- Python: 주요 개발 언어
- Flask: 웹 API 프레임워크
- Pandas & NumPy: 데이터 처리 및 분석
- Scikit-learn: 머신러닝 모델 구현
- MLflow: 모델 관리 및 추적
- Matplotlib & Seaborn: 데이터 시각화

## 향후 개선 사항
- 원시 데이터를 직접 전처리하는 추가 API 엔드포인트 구현
- Docker를 사용한 컨테이너화
- 모델 성능 개선을 위한 하이퍼파라미터 튜닝
- 웹 프론트엔드 인터페이스 개발
