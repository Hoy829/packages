from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

class ModelEvaluator:
    """
    모델 평가 모듈
    - 회귀 모델 평가
    - 분류 모델 평가
    - 클러스터링 모델 평가
    - 시계열 모델 평가
    - 딥러닝 모델 평가 (PyTorch 지원)
    - 일반적인 머신러닝 모델 학습 및 평가 지원
    """
    
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """
        회귀 모델 평가 함수
        :param y_true: 실제 값
        :param y_pred: 예측 값
        :return: MSE, RMSE, MAE, R2 score 출력
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R2 Score: {r2:.4f}')
    
    @staticmethod
    def evaluate_classification(y_true, y_pred):
        """
        분류 모델 평가 함수
        :param y_true: 실제 라벨
        :param y_pred: 예측 라벨
        :return: Accuracy, Precision, Recall, F1-score 출력
        """
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f'Accuracy: {acc:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        print('\nClassification Report:\n', classification_report(y_true, y_pred))
    
    @staticmethod
    def train_and_evaluate(model, X_train, X_test, y_train, y_test):
        """
        일반적인 머신러닝 모델 학습 및 평가 함수
        :param model: 학습할 모델 (예: LinearRegression, RandomForestClassifier 등)
        :param X_train: 학습 데이터
        :param X_test: 테스트 데이터
        :param y_train: 학습 라벨
        :param y_test: 테스트 라벨
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            print("Evaluating Classification Model")
            ModelEvaluator.evaluate_classification(y_test, y_pred)
        else:
            print("Evaluating Regression Model")
            ModelEvaluator.evaluate_regression(y_test, y_pred)
    
    @staticmethod
    def evaluate_pytorch_model(model, dataloader, device):
        """
        PyTorch 딥러닝 모델 평가 함수
        :param model: PyTorch 모델
        :param dataloader: 평가 데이터 로더
        :param device: 실행 장치 (CPU/GPU)
        :return: 평균 손실 및 정확도 출력
        """
        model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples
        
        print(f'PyTorch Model Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels):
        """
        혼동 행렬 시각화 함수
        :param y_true: 실제 라벨
        :param y_pred: 예측 라벨
        :param labels: 클래스 라벨 리스트
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
