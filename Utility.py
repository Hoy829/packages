import pandas as pd
import json
import time
import logging
from datetime import datetime

class Utility:
    """
    유틸리티 모듈
    - 파일 입출력 (CSV, JSON, Excel)
    - 로깅 기능
    - 실행 시간 측정
    - 데이터 요약 기능
    """

    @staticmethod
    def load_data(filepath, file_type='csv'):
        """
        파일 로드 함수
        :param filepath: 파일 경로
        :param file_type: 'csv', 'json', 'excel' 중 선택
        :return: 데이터프레임
        """
        if file_type == 'csv':
            return pd.read_csv(filepath)
        elif file_type == 'json':
            return pd.read_json(filepath)
        elif file_type == 'excel':
            return pd.read_excel(filepath)
        else:
            raise ValueError("지원되지 않는 파일 형식입니다. ('csv', 'json', 'excel' 만 지원)")

    @staticmethod
    def save_data(df, filepath, file_type='csv'):
        """
        파일 저장 함수
        :param df: 데이터프레임
        :param filepath: 저장할 파일 경로
        :param file_type: 'csv', 'json', 'excel' 중 선택
        """
        if file_type == 'csv':
            df.to_csv(filepath, index=False)
        elif file_type == 'json':
            df.to_json(filepath, orient='records', lines=True)
        elif file_type == 'excel':
            df.to_excel(filepath, index=False)
        else:
            raise ValueError("지원되지 않는 파일 형식입니다. ('csv', 'json', 'excel' 만 지원)")
    
    @staticmethod
    def setup_logging(logfile='app.log'):
        """
        로깅 설정 함수
        :param logfile: 로그 파일명
        """
        logging.basicConfig(
            filename=logfile,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("Logging initialized.")
    
    @staticmethod
    def log_message(message, level='info'):
        """
        메시지를 로그로 기록
        :param message: 기록할 메시지
        :param level: 로그 레벨 ('info', 'warning', 'error')
        """
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
    
    @staticmethod
    def time_execution(func):
        """
        실행 시간 측정 데코레이터
        :param func: 실행할 함수
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"실행 시간: {end_time - start_time:.4f}초")
            return result
        return wrapper
    
    @staticmethod
    def summarize_data(df):
        """
        데이터 요약 함수
        :param df: 데이터프레임
        :return: 기본 통계 요약 출력
        """
        print("📊 데이터 요약 📊")
        print("----------------------------")
        print(df.info())
        print("\n📌 기본 통계 정보")
        print(df.describe())
        print("\n🛑 결측치 확인")
        print(df.isnull().sum())
