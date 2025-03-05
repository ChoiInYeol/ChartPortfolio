"""
CNN_Model 패키지

이 패키지는 주식 차트 이미지를 CNN(Convolutional Neural Network) 모델로 분석하여 
주가 상승 확률을 예측하고, 이를 기반으로 포트폴리오를 구성하는 기능을 제공합니다.

주요 모듈:
- Data: 차트 데이터 처리 관련 모듈
- Experiments: 실험 관련 코드
- Misc: 유틸리티 및 설정
- Model: 모델 구현
- Portfolio: 포트폴리오 관리
"""

from . import Data
from . import Experiments
from . import Misc
from . import Model
from . import Portfolio

__all__ = ['Data', 'Experiments', 'Misc', 'Model', 'Portfolio', 'experiment', 'generate_data']
