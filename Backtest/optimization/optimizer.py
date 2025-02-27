"""포트폴리오 최적화를 위한 모듈입니다."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
import logging
from tqdm import tqdm
import os
import pickle
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB
import math
from dataclasses import dataclass
from scipy import stats

@dataclass
class SharpeRatioResult:
    """최적의 Sharpe ratio를 가지는 포트폴리오 결과를 저장하는 클래스입니다."""
    x: Union[np.ndarray, pd.Series]
    sharpe_ratio: float
    ret: float
    risk: float

def max_sharpe_ratio(cov_matrix: Union[np.ndarray, pd.DataFrame],
                    mu: Union[np.ndarray, pd.Series],
                    rf_rate: float = 0) -> SharpeRatioResult:
    """
    Sharpe ratio를 최대화하는 포트폴리오를 찾습니다.
    
    Args:
        cov_matrix: 공분산 행렬
        mu: 기대수익률
        rf_rate: 무위험 수익률 (기본값: 0)
        
    Returns:
        SharpeRatioResult: 최적화 결과
    """
    indices = None
    
    if isinstance(cov_matrix, pd.DataFrame):
        indices = cov_matrix.index
        cov_matrix = cov_matrix.to_numpy()
    
    if isinstance(mu, pd.Series):
        if indices is None:
            indices = mu.index
        mu = mu.to_numpy()
    
    # Gurobi 모델 생성
    model = gp.Model("sharpe_ratio")
    model.setParam('OutputFlag', 0)
    
    # 변수 추가
    y = model.addMVar(mu.size, name="y")
    
    # 제약조건
    model.addConstr((mu - rf_rate) @ y == 1)
    
    # 목적함수
    model.setObjective(y @ cov_matrix @ y, sense=GRB.MINIMIZE)
    
    # 최적화
    model.optimize()
    
    # 결과 변환
    x = y.X / y.X.sum()
    ret = mu @ x
    risk = x @ cov_matrix @ x
    sharpe_ratio = (ret - rf_rate) / math.sqrt(risk)
    
    if indices is not None:
        x = pd.Series(data=x, index=indices)
    
    return SharpeRatioResult(x, sharpe_ratio, ret, risk)

class OptimizationManager:
    """포트폴리오 최적화를 위한 클래스입니다."""
    
    def __init__(self, result_dir: str = 'results'):
        """OptimizationManager 초기화"""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = os.path.join(result_dir, 'optimization_cache')
        self.result_dir = result_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'weights'), exist_ok=True)

    def _get_cache_path(self, method: str, date: str) -> str:
        """캐시 파일 경로를 반환합니다."""
        return os.path.join(self.cache_dir, f"{method}_{date}.pkl")

    def _load_from_cache(self, cache_path: str) -> Dict:
        """캐시에서 최적화 결과를 로드합니다."""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def _save_to_cache(self, cache_path: str, data: Dict) -> None:
        """최적화 결과를 캐시에 저장합니다."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {str(e)}")

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        """
        시리즈의 값을 최소-최대 정규화하여 [0, 1] 범위로 변환합니다.
        
        Args:
            series (pd.Series): 정규화할 시리즈
        
        Returns:
            pd.Series: 정규화된 시리즈
        """
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    def _compute_overall_score(self, factor_values: Dict[str, pd.Series]) -> pd.Series:
        """
        각 팩터 점수를 정규화한 후 동일 가중치 평균을 계산하여 종목의 종합 팩터 점수를 반환합니다.
        
        Args:
            factor_values (Dict[str, pd.Series]): 팩터 이름과 해당 날짜의 팩터 점수를 담은 딕셔너리
        
        Returns:
            pd.Series: 종목별 종합 팩터 점수
        """
        normalized_scores = []
        for factor, series in factor_values.items():
            norm_series = self._normalize_series(series)
            normalized_scores.append(norm_series)
        overall_score = np.mean(normalized_scores, axis=0)
        return pd.Series(overall_score, index=list(factor_values.values())[0].index)

    def _fama_macbeth_regression(self,
                               returns: pd.DataFrame,
                               factors: pd.DataFrame,
                               window: int = 21) -> pd.Series:
        """
        Fama-MacBeth 회귀분석을 수행합니다.
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            factors (pd.DataFrame): 팩터 데이터
            window (int): 회귀분석 윈도우 크기 (기본값: 21일)
            
        Returns:
            pd.Series: 예측 수익률
        """
        try:
            import statsmodels.api as sm
            from tqdm import tqdm
            
            # 데이터 준비
            aligned_factors = factors.reindex(index=returns.index, columns=returns.columns)
            aligned_factors = aligned_factors.infer_objects(copy=False)
            aligned_factors = aligned_factors.fillna(aligned_factors.mean())
            
            # 각 기간별 회귀분석 수행
            betas = []
            n_periods = len(returns) // window
            
            # tqdm으로 진행률 표시 (한 번만 표시되도록 수정)
            for i in tqdm(range(0, len(returns), window),
                         desc="Fama-MacBeth Regression",
                         total=n_periods,
                         ncols=80,
                         position=0,
                         leave=True):
                period_returns = returns.iloc[i:i+window]
                period_factors = aligned_factors.iloc[i:i+window]
                
                if len(period_returns) < window:
                    continue
                
                # 각 종목별 회귀분석
                period_betas = []
                for stock in returns.columns:
                    try:
                        # 회귀분석 수행
                        X = sm.add_constant(period_factors[stock])
                        y = period_returns[stock]
                        model = sm.OLS(y, X).fit()
                        period_betas.append(model.params.iloc[1])
                    except:
                        period_betas.append(0)  # 회귀분석 실패 시 0으로 설정
                
                betas.append(period_betas)
            
            if not betas:
                return pd.Series(0, index=returns.columns)
            
            # 평균 베타 계산
            avg_betas = np.mean(betas, axis=0)
            
            # 최종 예측 수익률 계산
            latest_factors = aligned_factors.iloc[-1]
            predicted_returns = latest_factors * avg_betas
            
            return pd.Series(predicted_returns, index=returns.columns)
            
        except Exception as e:
            self.logger.error(f"Error in Fama-MacBeth regression: {str(e)}")
            return pd.Series(0, index=returns.columns)

    def _simple_regression(self,
                         returns: pd.DataFrame,
                         factors: pd.DataFrame) -> pd.Series:
        """
        단순 회귀분석을 수행합니다. Fama-MacBeth가 실패할 경우의 대체 방법입니다.
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            factors (pd.DataFrame): 팩터 데이터
            
        Returns:
            pd.Series: 예측 수익률
        """
        try:
            import statsmodels.api as sm
            
            # 데이터 준비
            aligned_factors = factors.reindex(index=returns.index, columns=returns.columns)
            aligned_factors = aligned_factors.fillna(aligned_factors.mean())
            
            # 각 종목별 회귀분석
            predicted_returns = pd.Series(0, index=returns.columns)
            
            for stock in returns.columns:
                try:
                    # 회귀분석 수행
                    X = sm.add_constant(aligned_factors[stock])
                    y = returns[stock]
                    model = sm.OLS(y, X).fit()
                    
                    # 예측 수익률 계산
                    latest_factor = aligned_factors[stock].iloc[-1]
                    predicted_returns[stock] = model.params[0] + model.params[1] * latest_factor
                except:
                    predicted_returns[stock] = returns[stock].mean()  # 회귀분석 실패 시 평균 수익률 사용
            
            return predicted_returns
            
        except Exception as e:
            self.logger.error(f"Error in simple regression: {str(e)}")
            return pd.Series(0, index=returns.columns)

    def create_factor_timing_portfolios(self,
                                      returns: pd.DataFrame,
                                      factor_scores: Dict[str, pd.DataFrame],
                                      rebalance_freq: str = 'ME',
                                      top_pct: float = 0.3,
                                      fm_window: int = 12,
                                      lookback_period: int = 60,
                                      min_history_months: int = 6,
                                      use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        팩터 타이밍 포트폴리오를 생성합니다.
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            factor_scores (Dict[str, pd.DataFrame]): 팩터 점수 데이터
            rebalance_freq (str): 리밸런싱 주기 (예: 'ME' - 월말)
            top_pct (float): 상위 종목 선택 비율
            fm_window (int): Fama-MacBeth 회귀분석 윈도우 크기
            lookback_period (int): 과거 데이터 사용 기간
            min_history_months (int): 최소 과거 데이터 필요 기간
            use_cache (bool): 캐시 사용 여부
            
        Returns:
            Dict[str, pd.DataFrame]: 포트폴리오 가중치
        """
        portfolio_weights = {}
        
        # 리밸런싱 날짜 계산
        rebalance_dates = pd.date_range(
            start=returns.index[0],
            end=returns.index[-1],
            freq=rebalance_freq
        )
        
        # 실제 거래일에 맞춰 리밸런싱 날짜 조정
        adjusted_dates = []
        for date in rebalance_dates:
            month_end = date
            available_dates = returns.index[returns.index <= month_end]
            if len(available_dates) > 0:
                adjusted_dates.append(available_dates[-1])
        
        rebalance_dates = pd.DatetimeIndex(sorted(set(adjusted_dates)))
        
        # 1. Factor Timing ND (Naive Diversification)
        print("Creating Factor Timing ND portfolio...")
        nd_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=returns.columns)
        
        for date in rebalance_dates:
            try:
                # 현재 시점의 팩터 점수 가져오기
                current_factors = {
                    name: scores.loc[:date].iloc[-1].astype(float)
                    for name, scores in factor_scores.items()
                }
                
                # 종합 점수 계산
                overall_score = self._compute_overall_score(current_factors)
                
                # 상위 종목 선택
                n_select = int(len(overall_score) * top_pct)
                top_stocks = overall_score.nlargest(n_select).index
                
                # 동일 가중치 할당
                nd_weights.loc[date, top_stocks] = 1.0 / n_select
                
            except Exception as e:
                self.logger.error(f"Error in Factor Timing ND at {date}: {str(e)}")
                # 이전 가중치 사용
                if date != rebalance_dates[0]:
                    nd_weights.loc[date] = nd_weights.loc[rebalance_dates[rebalance_dates < date][-1]]
        
        portfolio_weights['Factor Timing ND'] = nd_weights
        
        # 2. Factor Timing FM (Fama-MacBeth)
        print("Creating Factor Timing FM portfolio...")
        fm_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=returns.columns)
        
        for date in rebalance_dates:
            try:
                # 캐시 확인
                cache_path = self._get_cache_path('fm_weights', date.strftime('%Y%m%d'))
                cached_result = None
                if use_cache:
                    cached_result = self._load_from_cache(cache_path)
                
                if cached_result is not None:
                    fm_weights.loc[date] = cached_result
                    continue
                
                # 과거 데이터 선택
                hist_returns = returns.loc[:date].tail(lookback_period)
                hist_factors = {
                    name: scores.loc[:date].tail(lookback_period).astype(float)
                    for name, scores in factor_scores.items()
                }
                
                # 팩터 점수 결합
                combined_factors = pd.DataFrame(hist_factors)
                combined_factors = combined_factors.infer_objects(copy=False)
                combined_factors = combined_factors.fillna(method='ffill')
                
                # Fama-MacBeth 회귀분석으로 예측 수익률 계산
                predicted_returns = self._fama_macbeth_regression(
                    returns=hist_returns,
                    factors=combined_factors,
                    window=fm_window
                )
                
                if predicted_returns is not None and len(predicted_returns) > 0:
                    # 상위 종목 선택
                    n_select = int(len(predicted_returns) * top_pct)
                    top_stocks = predicted_returns.nlargest(n_select).index
                    
                    # 동일 가중치 할당
                    weights = pd.Series(0.0, index=returns.columns)
                    weights[top_stocks] = 1.0 / n_select
                    fm_weights.loc[date] = weights
                    
                    # 캐시 저장
                    if use_cache:
                        self._save_to_cache(cache_path, weights)
                else:
                    # 예측 실패 시 이전 가중치 사용
                    if date != rebalance_dates[0]:
                        fm_weights.loc[date] = fm_weights.loc[rebalance_dates[rebalance_dates < date][-1]]
                    else:
                        # 첫 번째 날짜인 경우 동일 가중치 사용
                        fm_weights.loc[date] = pd.Series(1.0 / len(returns.columns), index=returns.columns)
                
            except Exception as e:
                self.logger.error(f"Error in Factor Timing FM at {date}: {str(e)}")
                # 이전 가중치 사용
                if date != rebalance_dates[0]:
                    fm_weights.loc[date] = fm_weights.loc[rebalance_dates[rebalance_dates < date][-1]]
                else:
                    # 첫 번째 날짜인 경우 동일 가중치 사용
                    fm_weights.loc[date] = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        
        portfolio_weights['Factor Timing FM'] = fm_weights
        
        return portfolio_weights

    def optimize_portfolio_with_gurobi(self,
                                     returns: pd.DataFrame,
                                     method: str = 'max_sharpe',
                                     constraints: Optional[Dict] = None,
                                     target_return: float = None,
                                     show_progress: bool = True) -> np.ndarray:
        """
        Gurobi를 사용하여 포트폴리오를 최적화합니다.
        
        Args:
            returns (pd.DataFrame): 자산별 수익률 데이터
            method (str): 최적화 방법 ('max_sharpe', 'min_variance', 'min_cvar', 'target_return' 중 하나)
            constraints (Dict, optional): 제약조건 딕셔너리
            target_return (float, optional): 목표 연간 수익률 (예: 0.10 = 10%). method가 'target_return'일 때 사용
            show_progress (bool): 진행상황 표시 여부
            
        Returns:
            np.ndarray: 최적화된 포트폴리오 가중치
        """
        try:
            n_assets = returns.shape[1]
            
            # 데이터 검증
            if n_assets < 2:
                self.logger.warning("Not enough assets for optimization")
                return np.ones(n_assets) / n_assets
            
            if len(returns) < 20:
                self.logger.warning("Not enough historical data for optimization")
                return np.ones(n_assets) / n_assets
            
            # 기본 제약조건 설정
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,
                    'max_weight': 0.2
                }
            
            # 수익률 및 공분산 행렬 계산
            mu = returns.mean() * 252  # 연간화된 기대수익률
            Sigma = returns.cov() * 252  # 연간화된 공분산 행렬
            
            if method == 'max_sharpe':
                try:
                    # Sharpe ratio 최적화
                    result = max_sharpe_ratio(Sigma, mu, rf_rate=0)
                    weights = result.x
                    
                    # 제약조건 적용
                    weights = np.clip(weights, constraints['min_weight'], constraints['max_weight'])
                    weights = weights / weights.sum()  # 정규화
                    
                    # 결과 로깅
                    self.logger.info(f"Optimization successful:")
                    self.logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
                    self.logger.info(f"Expected Return: {result.ret:.1%}")
                    self.logger.info(f"Expected Risk: {math.sqrt(result.risk):.1%}")
                    
                    return np.array(weights)
                    
                except Exception as e:
                    self.logger.error(f"Sharpe ratio optimization failed: {str(e)}")
                    return np.ones(n_assets) / n_assets
            
            # Gurobi 모델 생성
            model = gp.Model("Portfolio_Optimization")
            model.setParam('OutputFlag', 1 if show_progress else 0)
            
            # 결정변수 추가: 각 자산의 투자 비중
            x = model.addMVar(n_assets,
                            lb=constraints['min_weight'],
                            ub=constraints['max_weight'],
                            name="weights")
            
            # 예산 제약조건: 투자 비중의 합이 1
            model.addConstr(x.sum() == 1, name="budget")
            
            if method == 'min_variance':
                # 최소 분산 포트폴리오
                model.setObjective(x @ Sigma.to_numpy() @ x, GRB.MINIMIZE)
                
            elif method == 'target_return':
                if target_return is None:
                    target_return = 0.10  # 기본값 10%
                
                # 목표수익률 제약조건
                model.addConstr(mu.to_numpy() @ x >= target_return, name="return_target")
                
                # 목적함수: 포트폴리오 분산 최소화
                model.setObjective(x @ Sigma.to_numpy() @ x, GRB.MINIMIZE)
                
            elif method == 'min_cvar':
                # CVaR 최소화
                alpha = 0.05  # 신뢰수준
                scenarios = len(returns)
                
                # VaR 및 초과손실 변수 추가
                var = model.addVar(name="VaR")
                z = model.addMVar(scenarios, name="excess_loss")
                
                # CVaR 제약조건
                for t in range(scenarios):
                    model.addConstr(
                        z[t] >= -(returns.iloc[t].to_numpy() @ x) - var,
                        name=f"CVaR_constr_{t}"
                    )
                    model.addConstr(z[t] >= 0, name=f"CVaR_nonneg_{t}")
                
                # CVaR 최소화 목적함수
                model.setObjective(
                    var + (1.0 / (alpha * scenarios)) * z.sum(),
                    GRB.MINIMIZE
                )
            
            # 모델 최적화
            try:
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    weights = x.X
                    
                    # 최적화 결과 로깅
                    final_return = mu.to_numpy() @ weights
                    final_risk = np.sqrt(weights @ Sigma.to_numpy() @ weights)
                    
                    self.logger.info(f"Optimization successful:")
                    self.logger.info(f"Expected Return: {final_return:.1%}")
                    self.logger.info(f"Expected Risk: {final_risk:.1%}")
                    self.logger.info(f"Sharpe Ratio: {final_return/final_risk:.2f}")
                    
                    return np.array(weights)
                else:
                    self.logger.warning(f"Optimization failed with status {model.status}")
                    return np.ones(n_assets) / n_assets
                    
            except gp.GurobiError as e:
                self.logger.error(f"Gurobi optimization error: {str(e)}")
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            return np.ones(n_assets) / n_assets

    def create_benchmark_portfolios(self, 
                                  up_prob: pd.DataFrame,
                                  returns: pd.DataFrame,
                                  rebalance_dates: pd.DatetimeIndex = None,
                                  use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        벤치마크 포트폴리오들을 생성합니다.
        
        Args:
            up_prob (pd.DataFrame): 상승확률 데이터 (월초 기준)
            returns (pd.DataFrame): 수익률 데이터 (거래일 기준)
            rebalance_dates (pd.DatetimeIndex): 리밸런싱 날짜
            use_cache (bool): 캐시 사용 여부
        
        Returns:
            Dict[str, pd.DataFrame]: 포트폴리오별 가중치
        """
        benchmark_weights = {}
        lookback_period = 60
        
        # 첫 번째 가능한 날짜 계산
        first_valid_date = returns.index[lookback_period]
        
        # 리밸런싱 날짜 설정
        if rebalance_dates is None:
            # up_prob의 인덱스(월초)를 기준으로 가장 가까운 거래일 찾기
            rebalance_dates = []
            for prob_date in up_prob.index:
                # prob_date 이후의 첫 거래일 찾기
                future_dates = returns.index[returns.index >= prob_date]
                if len(future_dates) > 0:
                    rebalance_dates.append(future_dates[0])
            rebalance_dates = pd.DatetimeIndex(sorted(set(rebalance_dates)))
        else:
            # 주어진 리밸런싱 날짜에 대해 가장 가까운 거래일 찾기
            adjusted_dates = []
            for date in rebalance_dates:
                # date 이후의 첫 거래일 찾기
                future_dates = returns.index[returns.index >= date]
                if len(future_dates) > 0:
                    adjusted_dates.append(future_dates[0])
            rebalance_dates = pd.DatetimeIndex(sorted(set(adjusted_dates)))
        
        if len(rebalance_dates) == 0:
            raise ValueError("No valid rebalancing dates found")
        
        # 각 리밸런싱 날짜에 대해 처리
        for date in tqdm(rebalance_dates[rebalance_dates >= first_valid_date],
                        desc="Creating benchmark portfolios"):
            date_str = date.strftime('%Y%m%d')
            
            # 해당 날짜에 대응되는 up_prob 데이터 찾기
            prob_dates = up_prob.index[up_prob.index <= date]
            if len(prob_dates) == 0:
                self.logger.warning(f"No probability data available for date {date}")
                continue
            
            prob_date = prob_dates[-1]
            month_probs = up_prob.loc[prob_date]
            
            # 상위 50% 종목 선택
            threshold = month_probs.median()
            up_stocks = month_probs[month_probs >= threshold].index
            
            # CNN Top 포트폴리오 초기화
            if 'CNN Top' not in benchmark_weights:
                benchmark_weights['CNN Top'] = pd.DataFrame(
                    0.0,
                    index=rebalance_dates,
                    columns=returns.columns,
                    dtype=np.float64
                )
            
            # 상승예측 종목이 없는 경우 처리
            if len(up_stocks) == 0:
                # 이전 달의 가중치를 유지
                prev_dates = benchmark_weights['CNN Top'].index[
                    benchmark_weights['CNN Top'].index < date
                ]
                if len(prev_dates) > 0:
                    prev_date = prev_dates[-1]
                    for portfolio_name in benchmark_weights:
                        benchmark_weights[portfolio_name].loc[date] = \
                            benchmark_weights[portfolio_name].loc[prev_date]
                else:
                    # 이전 가중치가 없는 경우 동일가중
                    equal_weight = np.float64(1.0/len(returns.columns))
                    for portfolio_name in benchmark_weights:
                        benchmark_weights[portfolio_name].loc[date] = equal_weight
                continue
            
            # CNN Top (상승예측 종목 동일가중)
            equal_weight = np.float64(1.0/len(up_stocks))
            benchmark_weights['CNN Top'].loc[date, up_stocks] = equal_weight
            
            # 전통적 벤치마크 전략들
            try:
                # 2-12 Momentum (MOM)
                mom_returns = self._calculate_momentum_returns(returns, date, 2, 12)
                mom_stocks = mom_returns.nlargest(len(up_stocks)).index
                if 'MOM' not in benchmark_weights:
                    benchmark_weights['MOM'] = pd.DataFrame(0.0, index=rebalance_dates,
                                                         columns=returns.columns, dtype=np.float64)
                benchmark_weights['MOM'].loc[date, mom_stocks] = equal_weight
                
                # 1-month Short-term Reversal (STR)
                str_returns = self._calculate_str_returns(returns, date, 21)  # 약 1달
                str_stocks = str_returns.nsmallest(len(up_stocks)).index  # 수익률이 낮은 종목 선택
                if 'STR' not in benchmark_weights:
                    benchmark_weights['STR'] = pd.DataFrame(0.0, index=rebalance_dates,
                                                         columns=returns.columns, dtype=np.float64)
                benchmark_weights['STR'].loc[date, str_stocks] = equal_weight
                
                # 1-week Short-term Reversal (WSTR)
                wstr_returns = self._calculate_str_returns(returns, date, 5)  # 1주일
                wstr_stocks = wstr_returns.nsmallest(len(up_stocks)).index
                if 'WSTR' not in benchmark_weights:
                    benchmark_weights['WSTR'] = pd.DataFrame(0.0, index=rebalance_dates,
                                                          columns=returns.columns, dtype=np.float64)
                benchmark_weights['WSTR'].loc[date, wstr_stocks] = equal_weight
                
                # TREND (Han, Zhou, and Zhu, 2016)
                trend_scores = self._calculate_trend_scores(returns, date)
                trend_stocks = trend_scores.nlargest(len(up_stocks)).index
                if 'TREND' not in benchmark_weights:
                    benchmark_weights['TREND'] = pd.DataFrame(0.0, index=rebalance_dates,
                                                           columns=returns.columns, dtype=np.float64)
                benchmark_weights['TREND'].loc[date, trend_stocks] = equal_weight
                
            except Exception as e:
                self.logger.error(f"Error calculating benchmark strategies for date {date}: {str(e)}")
                continue
            
            # 최적화 포트폴리오 생성
            for method in ['max_sharpe', 'min_variance', 'min_cvar']:
                method_name = method.replace('_', ' ').title()
                if method == 'min_cvar':
                    method_name = 'Min CVaR'
                
                # 캐시 확인
                cache_path = self._get_cache_path(method, date_str)
                cached_weights = None
                if use_cache:
                    cached_weights = self._load_from_cache(cache_path)
                
                # 기본 포트폴리오 초기화
                if method_name not in benchmark_weights:
                    benchmark_weights[method_name] = pd.DataFrame(
                        0.0,
                        index=rebalance_dates,
                        columns=returns.columns,
                        dtype=np.float64
                    )
                
                try:
                    if cached_weights is not None:
                        weights = cached_weights
                    else:
                        # 전체 종목에 대한 최적화
                        weights = self.optimize_portfolio_with_gurobi(
                            returns.loc[:date].tail(lookback_period), 
                            method=method,
                            show_progress=False
                        )
                        weights = np.array(weights, dtype=np.float64)
                        
                        # 캐시 저장
                        if use_cache:
                            self._save_to_cache(cache_path, weights)
                    
                    benchmark_weights[method_name].loc[date] = weights
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {method_name} for date {date}: {str(e)}")
                    # 최적화 실패 시 이전 가중치 유지
                    prev_dates = benchmark_weights[method_name].index[
                        benchmark_weights[method_name].index < date
                    ]
                    if len(prev_dates) > 0:
                        prev_date = prev_dates[-1]
                        benchmark_weights[method_name].loc[date] = \
                            benchmark_weights[method_name].loc[prev_date]
                    else:
                        # 이전 가중치가 없는 경우 동일가중
                        equal_weight = np.float64(1.0/len(returns.columns))
                        benchmark_weights[method_name].loc[date] = equal_weight
                
                # CNN Top + 최적화 포트폴리오
                portfolio_name = f'CNN Top + {method_name}'
                cache_path = self._get_cache_path(f"cnn_top_{method}", date_str)
                cached_weights = None
                if use_cache:
                    cached_weights = self._load_from_cache(cache_path)
                
                if portfolio_name not in benchmark_weights:
                    benchmark_weights[portfolio_name] = pd.DataFrame(
                        0.0,
                        index=rebalance_dates,
                        columns=returns.columns,
                        dtype=np.float64
                    )
                
                try:
                    if cached_weights is not None:
                        weights = cached_weights
                    else:
                        # CNN이 선택한 종목들에 대해서만 최적화
                        up_returns = returns.loc[:date].tail(lookback_period)[up_stocks]
                        weights = self.optimize_portfolio_with_gurobi(
                            up_returns,
                            method=method,
                            show_progress=False
                        )
                        weights = np.array(weights, dtype=np.float64)
                        
                        # 캐시 저장
                        if use_cache:
                            self._save_to_cache(cache_path, weights)
                    
                    benchmark_weights[portfolio_name].loc[date, up_stocks] = weights
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {portfolio_name} for date {date}: {str(e)}")
                    # 최적화 실패 시 동일가중
                    benchmark_weights[portfolio_name].loc[date, up_stocks] = equal_weight
        
        # 모든 포트폴리오의 가중치를 월말 기준으로 리샘플링
        for name in benchmark_weights:
            benchmark_weights[name] = benchmark_weights[name].resample('ME').last()
        
        return benchmark_weights

    def _calculate_momentum_returns(self, returns: pd.DataFrame, 
                                  current_date: pd.Timestamp,
                                  skip_months: int,
                                  lookback_months: int) -> pd.Series:
        """2-12 Momentum 수익률을 계산합니다."""
        end_date = current_date - pd.DateOffset(months=skip_months)
        start_date = end_date - pd.DateOffset(months=lookback_months)
        
        # 해당 기간의 수익률 데이터 추출
        period_returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
        
        # 누적 수익률 계산
        cum_returns = (1 + period_returns).prod() - 1
        return cum_returns

    def _calculate_str_returns(self, returns: pd.DataFrame,
                             current_date: pd.Timestamp,
                             lookback_days: int) -> pd.Series:
        """Short-term Reversal 수익률을 계산합니다."""
        start_date = current_date - pd.Timedelta(days=lookback_days)
        
        # 해당 기간의 수익률 데이터 추출
        period_returns = returns[(returns.index >= start_date) & (returns.index <= current_date)]
        
        # 누적 수익률 계산
        cum_returns = (1 + period_returns).prod() - 1
        return cum_returns

    def _calculate_trend_scores(self, returns: pd.DataFrame,
                              current_date: pd.Timestamp) -> pd.Series:
        """TREND 전략의 점수를 계산합니다 (Han, Zhou, and Zhu, 2016)."""
        # 단기(1주일), 중기(1개월), 장기(12개월) 트렌드 계산
        short_term = self._calculate_str_returns(returns, current_date, 5)
        mid_term = self._calculate_str_returns(returns, current_date, 21)
        long_term = self._calculate_momentum_returns(returns, current_date, 0, 12)
        
        # 각 기간별 가중치 (논문 기준)
        weights = {'short': 0.5, 'mid': 0.3, 'long': 0.2}
        
        # 종합 점수 계산
        trend_scores = (short_term * weights['short'] + 
                       mid_term * weights['mid'] + 
                       long_term * weights['long'])
        
        return trend_scores