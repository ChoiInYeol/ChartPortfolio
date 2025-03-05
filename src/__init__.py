"""
ChartPortfolio 프로젝트

이 프로젝트는 주식 차트 이미지를 CNN(Convolutional Neural Network) 모델로 분석하여 
주가 상승 확률을 예측하고, 이를 기반으로 포트폴리오를 구성하는 시스템을 구현합니다.

주요 패키지:
- data: 데이터 처리 모듈
- CNN_Model: CNN 모델 관련 모듈
"""

from . import data
from . import CNN_Model

__all__ = ['data', 'CNN_Model'] 