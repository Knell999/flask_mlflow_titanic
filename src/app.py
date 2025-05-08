import os
import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = Flask(__name__)

# MLflow 서버 설정
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# 데이터 로드 (예측 시 필요한 feature 정보를 위해)
data_path = '../data/titanic_train.csv'
train_data = pd.read_csv(data_path)

# MLflow에서 최고 성능의 모델 가져오기
def get_best_model():
    # Titanic 실험 ID 가져오기
    experiment = client.get_experiment_by_name("Titanic_Survival_Prediction")
    if experiment is None:
        raise Exception("No experiment found with name Titanic_Survival_Prediction")
    
    experiment_id = experiment.experiment_id
    
    # 실험에 대한 모든 실행 가져오기
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string="",
        order_by=["metrics.test_accuracy DESC"]
    )
    
    if not runs:
        raise Exception("No runs found for experiment")
    
    best_run = runs[0]  # 정확도가 가장 높은 모델
    best_run_id = best_run.info.run_id
    
    print(f"Best model run_id: {best_run_id}, accuracy: {best_run.data.metrics.get('test_accuracy', 0.0)}")
    
    # 최고 성능의 모델 로드
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

# 모델 로드
model = get_best_model()

@app.route('/predict', methods=['GET'])
def predict():
    passenger_id = request.args.get('PassengerId', type=int)
    
    if not passenger_id:
        return jsonify({"error": "PassengerId parameter is required"}), 400
    
    try:
        # PassengerId로 데이터 검색
        passenger = train_data[train_data['PassengerId'] == passenger_id]
        
        if passenger.empty:
            return jsonify({"error": f"Passenger with ID {passenger_id} not found"}), 404
        
        # 예측에 필요한 특성 선택
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        X = passenger[features]
        
        # 예측 수행
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # class 1에 대한 확률
        
        # 결과 리턴
        result = {
            "PassengerId": int(passenger_id),
            "Name": passenger['Name'].values[0],
            "Survived": int(prediction),
            "Probability": float(probability),
            "Prediction": "생존" if prediction == 1 else "사망"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)