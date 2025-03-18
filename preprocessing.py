import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """
    데이터 전처리 모듈
    이 모듈은 결측치 처리, 이상치 제거, 데이터 스케일링 기능을 포함합니다.
    """
    def __init__(self, df: pd.DataFrame):
        """
        데이터 전처리 클래스
        :param df: 입력 데이터프레임
        """
        self.df = df.copy()
    
    def fill_missing(self, strategy='mean', columns=None):
        """
        결측치 처리 함수
        :param strategy: 'mean', 'median', 'mode', 'constant' 중 선택
        :param columns: 특정 컬럼을 지정하지 않으면 모든 컬럼에 적용
        :return: 결측치 처리된 데이터프레임
        """
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if self.df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'constant':
                    self.df[col].fillna(0, inplace=True)
        return self.df

    def remove_outliers(self, columns, method='zscore', threshold=3):
        """
        이상치 제거 함수
        :param columns: 이상치를 제거할 컬럼 리스트
        :param method: 'zscore' 또는 'iqr' 중 선택
        :param threshold: z-score 기준치 (기본값: 3)
        :return: 이상치가 제거된 데이터프레임
        """
        if method == 'zscore':
            for col in columns:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
        elif method == 'iqr':
            for col in columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[(self.df[col] >= Q1 - 1.5 * IQR) & (self.df[col] <= Q3 + 1.5 * IQR)]
        return self.df

    def scale_features(self, columns, method='standard'):
        """
        데이터 스케일링 함수
        :param columns: 스케일링할 컬럼 리스트
        :param method: 'standard' 또는 'minmax' 중 선택
        :return: 스케일링된 데이터프레임
        """
        if method not in ['standard', 'minmax']:
            raise ValueError("method must be either 'standard' or 'minmax'")
        
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def get_dataframe(self):
        """
        전처리된 데이터프레임 반환
        :return: 전처리된 데이터프레임
        """
        return self.df
