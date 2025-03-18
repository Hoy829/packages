import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure, show, output_file
import altair as alt

class DataVisualizer:
    """
    데이터 시각화 모듈
    주요 기능:
    - 데이터 분포 시각화
    - 상관관계 히트맵
    - 카테고리별 박스플롯
    - 대화형 시각화 (Plotly, Bokeh, Altair 지원)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        데이터프레임을 입력으로 받는 시각화 클래스
        :param df: 입력 데이터프레임
        """
        self.df = df.copy()
    
    def plot_distribution(self, column):
        """
        연속형 변수의 분포 시각화
        :param column: 시각화할 컬럼
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    
    def plot_correlation_heatmap(self):
        """
        상관관계 히트맵 시각화
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.show()
    
    def plot_boxplot(self, column, category):
        """
        카테고리별 박스플롯 시각화
        :param column: 연속형 변수 컬럼
        :param category: 그룹화할 범주형 변수 컬럼
        """
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=self.df[category], y=self.df[column])
        plt.title(f'Boxplot of {column} by {category}')
        plt.xlabel(category)
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()
    
    def plot_interactive_scatter(self, x_column, y_column):
        """
        대화형 산점도 (Plotly 사용)
        :param x_column: X축 변수
        :param y_column: Y축 변수
        """
        fig = px.scatter(self.df, x=x_column, y=y_column, title=f'Scatter Plot of {x_column} vs {y_column}',
                         trendline='ols', opacity=0.7)
        fig.show()
    
    def plot_interactive_bar(self, category, value):
        """
        대화형 막대그래프 (Plotly 사용)
        :param category: 범주형 변수
        :param value: 수치형 변수
        """
        fig = px.bar(self.df, x=category, y=value, title=f'Bar Chart of {value} by {category}')
        fig.show()
    
    def plot_bokeh_line(self, x_column, y_column):
        """
        대화형 선 그래프 (Bokeh 사용)
        :param x_column: X축 변수
        :param y_column: Y축 변수
        """
        p = figure(title=f'Bokeh Line Chart: {x_column} vs {y_column}', x_axis_label=x_column, y_axis_label=y_column,
                   plot_width=800, plot_height=500)
        p.line(self.df[x_column], self.df[y_column], legend_label=f'{y_column} Trend', line_width=2)
        output_file("bokeh_line_chart.html")
        show(p)
    
    def plot_altair_histogram(self, column):
        """
        대화형 히스토그램 (Altair 사용)
        :param column: 시각화할 컬럼
        """
        chart = alt.Chart(self.df).mark_bar().encode(
            alt.X(column, bin=True),
            alt.Y('count()'),
            tooltip=[column]
        ).properties(title=f'Altair Histogram of {column}')
        chart.show()
