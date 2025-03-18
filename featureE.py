import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

class FeatureEngineering:
    """
    특성 엔지니어링 모듈
    - 특성 스케일링 (표준화, 정규화)
    - 특성 선택 (상관관계 기반, 중요도 기반)
    - 특성 추출 (PCA, 다항식 변환)
    - 범주형 인코딩 (One-Hot, Label Encoding)
    - 날짜형 데이터 변환
    """
    
    @staticmethod
    def scale_features(df, columns, method='standard'):
        """
        특성 스케일링
        :param df: 데이터프레임
        :param columns: 스케일링할 컬럼 리스트
        :param method: 'standard' 또는 'minmax'
        :return: 스케일링된 데이터프레임
        """
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df
    
    @staticmethod
    def select_features(X, y, method='anova', k=5):
        """
        특성 선택 (ANOVA, Mutual Information)
        :param X: 입력 데이터
        :param y: 타겟 변수
        :param method: 'anova' 또는 'mutual_info'
        :param k: 선택할 특성 개수
        :return: 선택된 특성 데이터프레임
        """
        if method == 'anova':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new
    
    @staticmethod
    def extract_pca(X, n_components=2):
        """
        PCA를 이용한 차원 축소
        :param X: 입력 데이터
        :param n_components: 축소할 차원 수
        :return: 변환된 데이터프레임
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca
    
    @staticmethod
    def create_polynomial_features(X, degree=2):
        """
        다항식 특성 생성
        :param X: 입력 데이터
        :param degree: 다항 차수
        :return: 변환된 데이터프레임
        """
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        return X_poly
    
    @staticmethod
    def encode_categorical(df, columns, method='onehot'):
        """
        범주형 변수 인코딩
        :param df: 데이터프레임
        :param columns: 인코딩할 컬럼 리스트
        :param method: 'onehot' 또는 'label'
        :return: 변환된 데이터프레임
        """
        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns)
        else:
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        return df
    
    @staticmethod
    def extract_datetime_features(df, column):
        """
        날짜형 변수 변환 (년, 월, 일, 요일)
        :param df: 데이터프레임
        :param column: 변환할 날짜 컬럼
        :return: 변환된 데이터프레임
        """
        df[column] = pd.to_datetime(df[column])
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        df[f'{column}_day'] = df[column].dt.day
        df[f'{column}_weekday'] = df[column].dt.weekday
        df.drop(columns=[column], inplace=True)
        return df
