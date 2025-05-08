import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# MLflow 서버 설정
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 전처리된 데이터 로드 - 상대 경로 설정
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
X = pd.read_csv(os.path.join(data_dir, 'preprocessed_features.csv'))
y = pd.read_csv(os.path.join(data_dir, 'preprocessed_target.csv'))['Survived']

# 'Deck' 열은 문자열 값(A, B, C, D, E 등)을 포함하고 있어서 추가 전처리 필요
# NaN 값은 빈 문자열로 대체
X['Deck'] = X['Deck'].fillna('')

# Deck 열을 원-핫 인코딩으로 변환
deck_dummies = pd.get_dummies(X['Deck'], prefix='Deck')
X = pd.concat([X.drop('Deck', axis=1), deck_dummies], axis=1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 딕셔너리 정의
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10)
}

# MLflow 실험 설정
mlflow.set_experiment("Titanic_Survival_Prediction")

# 자동 로깅 설정
mlflow.sklearn.autolog()

# 모델 성능 추적을 위한 변수
best_accuracy = 0
best_model = None
best_model_name = None

# 각 모델 학습 및 평가
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 평가 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 평가 지표 로깅
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # ROC AUC 계산 및 로깅 (predict_proba를 지원하는 모델만)
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", roc_auc)
        
        # 모델 파라미터 로깅
        mlflow.log_param("model_class", name)
        
        print(f"Model: {name}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if y_proba is not None:
            print(f"  ROC AUC: {roc_auc:.4f}")
        
        # 모델 저장
        mlflow.sklearn.log_model(model, "model")
        
        # 최고 성능 모델 추적
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

# 최고 성능 모델 저장
if best_model is not None:
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    mlflow.sklearn.save_model(best_model, model_path)
    print(f"Best model saved to {model_path}")
else:
    print("No model was trained successfully.")