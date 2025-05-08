import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = Flask(__name__)

# MLflow 서버 설정
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# 데이터 로드 - 전처리된 데이터셋 사용
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
features = pd.read_csv(os.path.join(data_dir, 'preprocessed_features.csv'))
target = pd.read_csv(os.path.join(data_dir, 'preprocessed_target.csv'))
train_data = pd.concat([features, target], axis=1)

# MLflow에서 최고 성능의 모델 가져오기
def get_best_model():
    # 직접 학습된 모델 파일 경로
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "random_forest_model")
    
    # 학습된 모델이 있으면 해당 모델 사용
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return mlflow.sklearn.load_model(model_path)
    
    # 학습된 모델이 없으면 MLflow에서 최고 성능 모델 검색
    experiment = client.get_experiment_by_name("Titanic_Survival_Prediction")
    if experiment is None:
        raise Exception("No experiment found with name Titanic_Survival_Prediction")
    
    experiment_id = experiment.experiment_id
    
    # 실험에 대한 모든 실행 가져오기
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string="",
        order_by=["metrics.accuracy DESC"]  # 정확도 기준으로 정렬
    )
    
    if not runs:
        raise Exception("No runs found for experiment")
    
    best_run = runs[0]  # 정확도가 가장 높은 모델
    best_run_id = best_run.info.run_id
    
    print(f"Best model run_id: {best_run_id}, accuracy: {best_run.data.metrics.get('accuracy', 0.0)}")
    
    # 최고 성능의 모델 로드
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

# 데이터 전처리 함수
def preprocess_data(passenger_data):
    # 원본 데이터가 아닌 전처리된 데이터에서 가져온 경우
    if 'PassengerId' in passenger_data.columns:
        passenger_id = passenger_data['PassengerId'].values[0]
        # 전처리된 데이터에서 해당 승객 찾기
        preprocessed_data = features.iloc[passenger_id - 1:passenger_id]
        if not preprocessed_data.empty:
            return preprocessed_data
    
    # 원본 데이터를 전처리해야 하는 경우
    # 이 부분은 필요하다면 train_mlflow.py의 전처리 로직을 구현
    
    return passenger_data

# 모델 로드
model = get_best_model()

@app.route('/predict', methods=['GET'])
def predict():
    passenger_id = request.args.get('PassengerId', type=int)
    
    if not passenger_id:
        return jsonify({"error": "PassengerId parameter is required"}), 400
    
    try:
        # 전처리된 데이터에서 승객 검색 (인덱스는 0부터 시작하므로 -1)
        passenger_idx = passenger_id - 1
        if passenger_idx < 0 or passenger_idx >= len(features):
            return jsonify({"error": f"Passenger with ID {passenger_id} not found"}), 404
        
        # 예측에 사용할 특성 데이터
        X = features.iloc[passenger_idx:passenger_idx+1]
        
        # Deck 열이 있고 문자열 값이면 원-핫 인코딩 적용
        if 'Deck' in X.columns and X['Deck'].dtype == object:
            X['Deck'] = X['Deck'].fillna('')
            deck_dummies = pd.get_dummies(X['Deck'], prefix='Deck')
            X = pd.concat([X.drop('Deck', axis=1), deck_dummies], axis=1)
        
        # 예측 수행
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None
        
        # 결과 리턴
        result = {
            "PassengerId": int(passenger_id),
            "Survived": int(prediction),
            "Prediction": "생존" if prediction == 1 else "사망"
        }
        
        if probability is not None:
            result["Probability"] = float(probability)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)