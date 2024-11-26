"""
확률 기반의 High/Low 포트폴리오 성과 시각화

이 스크립트는 다음을 수행합니다:
1. 각 시점에서 상승 확률이 가장 높은/낮은 종목 선택
2. High/Low/H-L 포트폴리오의 수익률 계산
3. 누적 수익률 시각화
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


class PortfolioBacktest:
    """포트폴리오 백테스팅 클래스"""

    def __init__(self, prob_df: pd.DataFrame, ret_df: pd.DataFrame, n_stocks: int = 500):
        """
        Args:
            prob_df: 확률 데이터프레임 (index=investment_date, columns=StockID)
            ret_df: 일별 수익률 데이터프레임 (index=date, columns=StockID)
            n_stocks: 포트폴리오 내 종목 수
        """
        self.prob_df = prob_df.copy()
        self.ret_df = ret_df.copy()
        self.n_stocks = n_stocks
        self.freq = 'month'

        # 인덱스를 datetime으로 변환
        self.prob_df.index = pd.to_datetime(self.prob_df.index)
        self.ret_df.index = pd.to_datetime(self.ret_df.index)

        # 투자 기간 설정
        self.start_date = pd.Timestamp("2001-01-01")
        self.end_date = pd.Timestamp("2024-10-01")
        self.cutoff_date = pd.Timestamp("2018-01-01")

        # 거래일 목록 생성
        self.trading_days = self.ret_df.index.sort_values()

        # 기간 필터링
        self.prob_df = self.prob_df[
            (self.prob_df.index >= self.start_date)
            & (self.prob_df.index <= self.end_date)
        ]
        self.ret_df = self.ret_df[
            (self.ret_df.index >= self.start_date)
            & (self.ret_df.index <= self.end_date)
        ]

    def get_trading_day_offset(self, date: pd.Timestamp, offset: int) -> Optional[pd.Timestamp]:
        """
        특정 날짜로부터 offset만큼 떨어진 거래일 반환

        Args:
            date: 기준 날짜
            offset: 거래일 기준 오프셋 (양수: 미래, 음수: 과거)

        Returns:
            offset만큼 떨어진 거래일의 날짜 또는 None
        """
        date = pd.Timestamp(date)
        trading_days = self.trading_days

        # date가 trading_days에 없는 경우 가장 가까운 미래의 거래일로 매핑
        try:
            idx = trading_days.get_loc(date)
        except KeyError:
            idx = trading_days.get_indexer([date], method="bfill")[0]
            if idx == -1:
                return None  # date 이후의 거래일이 없음

        target_idx = idx + offset
        if 0 <= target_idx < len(trading_days):
            return trading_days[target_idx]
        else:
            return None

    def generate_portfolio_with_delay(self, delay: int, cut: int = 10) -> pd.DataFrame:
        """지정된 delay로 포트폴리오 수익률 생성"""
        portfolio_returns = []

        prob_dates = self.prob_df.index.sort_values()
        for i, date in enumerate(prob_dates):
            probs = self.prob_df.loc[date].dropna()
            if probs.empty:
                continue

            # delay에 따른 투자일 계산
            investment_date = self.get_trading_day_offset(date, delay)
            if investment_date is None:
                continue

            # 다음 리밸런싱 날짜 계산
            if i + 1 < len(prob_dates):
                next_date = prob_dates[i + 1]
            else:
                next_date = self.end_date
            next_investment_date = self.get_trading_day_offset(next_date, delay)
            if next_investment_date is None:
                next_investment_date = self.end_date

            # 투자 기간 수익률 데이터 가져오기
            mask = (self.ret_df.index >= investment_date) & (
                self.ret_df.index < next_investment_date
            )
            holding_period_returns = self.ret_df.loc[mask]

            if holding_period_returns.empty:
                continue

            # 상위/하위 종목 선택
            cut_size = len(probs) // cut
            high_stocks = probs.nlargest(cut_size).index
            low_stocks = probs.nsmallest(cut_size).index

            # 수익률 데이터에 존재하는 종목만 선택
            available_stocks = set(holding_period_returns.columns)
            high_stocks = list(set(high_stocks) & available_stocks)
            low_stocks = list(set(low_stocks) & available_stocks)

            if not high_stocks or not low_stocks:
                continue

            # High 포트폴리오 수익률 계산
            high_returns = holding_period_returns[high_stocks]
            high_cum_returns = (1 + high_returns).prod() - 1
            high_portfolio_return = high_cum_returns.mean()

            # Low 포트폴리오 수익률 계산
            low_returns = holding_period_returns[low_stocks]
            low_cum_returns = (1 + low_returns).prod() - 1
            low_portfolio_return = low_cum_returns.mean()

            portfolio_returns.append(
                {
                    "date": investment_date,
                    "High": high_portfolio_return,
                    "Low": low_portfolio_return,
                    "H-L": high_portfolio_return - low_portfolio_return,
                }
            )

        return pd.DataFrame(portfolio_returns).set_index("date")

    def calculate_metrics(self, returns_df: pd.DataFrame, delay: int) -> pd.DataFrame:
        """
        포트폴리오 성과 지표와 상관계수를 계산합니다.

        Args:
            returns_df: 포트폴리오 수익률 DataFrame (columns=['High', 'Low', 'H-L'])
            delay: 수익률 계산에 사용된 딜레이

        Returns:
            계산된 메트릭 DataFrame (index=메트릭, columns=포트폴리오)
        """
        if returns_df.empty:
            return pd.DataFrame()

        annual_factor = 252  # 일간 수익률 기준
        metrics_dict = {}

        # 포트폴리오별 기본 성과 지표 계산
        for portfolio in ["High", "Low", "H-L"]:
            rets = returns_df[portfolio].dropna()

            if len(rets) < 2:
                metrics_dict[portfolio] = {
                    "Annual Return": np.nan,
                    "Annual Volatility": np.nan,
                    "Sharpe Ratio": np.nan,
                    "Win Rate": np.nan,
                    "Max Drawdown": np.nan,
                }
                continue

            # 일간 평균 수익률 및 표준편차 계산
            daily_return = rets.mean()
            daily_vol = rets.std()

            # 단순 연율화 방식으로 변경
            annual_return = daily_return * annual_factor
            annual_vol = daily_vol * np.sqrt(annual_factor)

            sharpe_ratio = (annual_return / annual_vol) if annual_vol != 0 else np.nan
            win_rate = (rets > 0).mean()

            cum_returns = (1 + rets).cumprod()
            drawdown = cum_returns / cum_returns.cummax() - 1
            max_drawdown = drawdown.min()

            metrics_dict[portfolio] = {
                "Annual Return": annual_return,
                "Annual Volatility": annual_vol,
                "Sharpe Ratio": sharpe_ratio,
                "Win Rate": win_rate,
                "Max Drawdown": max_drawdown,
            }

        # 상관계수 계산을 위한 데이터 수집
        prob_df = self.prob_df.copy()
        ret_df = self.ret_df.copy()

        ic_data = []
        rank_ic_data = []
        dates = []

        for date in prob_df.index:
            probs = prob_df.loc[date].dropna()
            if probs.empty:
                continue

            # 딜레이 적용
            ret_date = self.get_trading_day_offset(date, delay)
            if ret_date is None or ret_date not in ret_df.index:
                continue

            rets = ret_df.loc[ret_date]

            common_stocks = list(set(probs.index) & set(rets.index))
            if len(common_stocks) < 10:  # 최소 10개 종목 필요
                continue

            probs = probs[common_stocks]
            rets = rets[common_stocks]

            # 전체 종목에 대한 IC 계산
            ic = np.corrcoef(probs, rets)[0, 1]
            rank_ic = spearmanr(probs, rets)[0]

            # 데이터 저장
            if not np.isnan(ic):
                ic_data.append(ic)
            if not np.isnan(rank_ic):
                rank_ic_data.append(rank_ic)
            dates.append(date)

        # t-stat 계산 함수
        def safe_t_stat(data):
            if not data or len(data) < 2:
                return np.nan
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.nan
            return mean / (std / np.sqrt(len(data)))

        # IC 메트릭은 High 포트폴리오에만 계산
        metrics_dict['High'].update({
            'IC (Mean)': np.mean(ic_data) if ic_data else np.nan,
            'IC (Std)': np.std(ic_data) if ic_data else np.nan,
            'IC (t-stat)': safe_t_stat(ic_data),
            'Rank IC (Mean)': np.mean(rank_ic_data) if rank_ic_data else np.nan,
            'Rank IC (Std)': np.std(rank_ic_data) if rank_ic_data else np.nan,
            'Rank IC (t-stat)': safe_t_stat(rank_ic_data),
        })

        # Low와 H-L 포트폴리오에는 NaN 값 할당
        for portfolio in ['Low', 'H-L']:
            metrics_dict[portfolio].update({
                'IC (Mean)': np.nan,
                'IC (Std)': np.nan,
                'IC (t-stat)': np.nan,
                'Rank IC (Mean)': np.nan,
                'Rank IC (Std)': np.nan,
                'Rank IC (t-stat)': np.nan,
            })

        return pd.DataFrame(metrics_dict)

    def compare_delays_with_periods(
        self, delays: Optional[List[int]] = None, cut: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """다양한 delay에 대한 포트폴리오 성과 비교"""
        if delays is None:
            delays = [0, 1, 2, 3, 5, 10, 20]

        is_results = []
        oos_results = []

        # 전체 데이터 보존
        full_prob_df = self.prob_df.copy()
        full_ret_df = self.ret_df.copy()

        for delay in delays:
            print(f"Analyzing delay: {delay} trading days")

            # In-Sample 기간
            self.prob_df = full_prob_df[full_prob_df.index < self.cutoff_date]
            self.ret_df = full_ret_df[
                (full_ret_df.index >= self.start_date)
                & (full_ret_df.index < self.cutoff_date)
            ]
            is_portfolio_rets = self.generate_portfolio_with_delay(delay, cut)

            # IS 기간의 상관계수 계산
            if not is_portfolio_rets.empty:
                is_metrics = self.calculate_metrics(is_portfolio_rets, delay).round(4)
                is_metrics.columns = pd.MultiIndex.from_product(
                    [[delay], is_metrics.columns]
                )
                is_results.append(is_metrics)

            # Out-of-Sample 기간
            self.prob_df = full_prob_df[full_prob_df.index >= self.cutoff_date]
            self.ret_df = full_ret_df[full_ret_df.index >= self.cutoff_date]
            oos_portfolio_rets = self.generate_portfolio_with_delay(delay, cut)

            # OOS 기간의 상관계수 계산
            if not oos_portfolio_rets.empty:
                oos_metrics = self.calculate_metrics(oos_portfolio_rets, delay).round(4)
                oos_metrics.columns = pd.MultiIndex.from_product(
                    [[delay], oos_metrics.columns]
                )
                oos_results.append(oos_metrics)

        # 결과 병합
        is_df = pd.concat(is_results, axis=1)
        oos_df = pd.concat(oos_results, axis=1)

        # 결과 저장
        results = {"IS": is_df.round(4), "OOS": oos_df.round(4)}
        for period, df in results.items():
            output_path = f"./processed/delay_comparison_{period}.csv"
            df.to_csv(output_path, float_format='%.4f')
            print(f"{period} comparison saved to {output_path}")

        # 최적의 delay 찾기
        best_delays = self._find_best_delays(is_df, oos_df)
        print("Best Delays:")
        print(best_delays)

        return results
    
    def plot_ic_distribution(self, results: Dict[str, pd.DataFrame], save_dir: str = "./processed") -> None:
        """IC 분포를 시각화합니다."""
        prob_df = self.prob_df.copy()
        ret_df = self.ret_df.copy()
        
        ic_data = []
        rank_ic_data = []
        dates = []
        
        for date in prob_df.index:
            probs = prob_df.loc[date].dropna()
            if probs.empty:
                continue
                
            next_date = self.get_trading_day_offset(date, 1)
            if next_date is None or next_date not in ret_df.index:
                continue
                
            rets = ret_df.loc[next_date]
            
            common_stocks = list(set(probs.index) & set(rets.index))
            if not common_stocks:
                continue
                
            probs = probs[common_stocks]
            rets = rets[common_stocks]
            
            ic = np.corrcoef(probs, rets)[0, 1]
            rank_ic = spearmanr(probs, rets)[0]
            
            ic_data.append(ic)
            rank_ic_data.append(rank_ic)
            dates.append(date)
        
        # IC 시계열 플롯
        plt.figure(figsize=(12, 6))
        plt.plot(dates, ic_data, label='IC', alpha=0.7)
        plt.plot(dates, rank_ic_data, label='Rank IC', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Information Coefficient Time Series')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, "ic_timeseries.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # IC 히스토그램
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # IC 히스토그램
        sns.histplot(ic_data, kde=True, ax=ax1)
        ax1.axvline(np.mean(ic_data), color='r', linestyle='--', label=f'Mean: {np.mean(ic_data):.3f}')
        ax1.set_title('IC Distribution')
        ax1.set_xlabel('IC')
        ax1.legend()
        
        # Rank IC 히스토그램
        sns.histplot(rank_ic_data, kde=True, ax=ax2)
        ax2.axvline(np.mean(rank_ic_data), color='r', linestyle='--', label=f'Mean: {np.mean(rank_ic_data):.3f}')
        ax2.set_title('Rank IC Distribution')
        ax2.set_xlabel('Rank IC')
        ax2.legend()
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "ic_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _find_best_delays(self, is_df: pd.DataFrame, oos_df: pd.DataFrame) -> pd.DataFrame:
        """각 지표와 포트폴리오에 대해 최적의 delay를 찾습니다."""
        metrics = is_df.index
        portfolios = ["High", "Low", "H-L"]
        best_delays = {"IS": {}, "OOS": {}}

        for metric in metrics:
            for portfolio in portfolios:
                # Low 포트폴리오는 최소값을 찾고, 나머지는 최대값을 찾습니다
                find_max = portfolio != "Low"
                
                # IS 최적 delay
                is_values = is_df.loc[metric, pd.IndexSlice[:, portfolio]]
                if is_values.notna().any():
                    best_is_idx = is_values.values.argmax() if find_max else is_values.values.argmin()
                    best_is_delay = is_values.index[best_is_idx][0]
                    best_is_value = is_values.iloc[best_is_idx]
                else:
                    best_is_delay = np.nan
                    best_is_value = np.nan

                # OOS 최적 delay
                oos_values = oos_df.loc[metric, pd.IndexSlice[:, portfolio]]
                if oos_values.notna().any():
                    best_oos_idx = oos_values.values.argmax() if find_max else oos_values.values.argmin()
                    best_oos_delay = oos_values.index[best_oos_idx][0]
                    best_oos_value = oos_values.iloc[best_oos_idx]
                else:
                    best_oos_delay = np.nan
                    best_oos_value = np.nan

                best_delays["IS"][f"{portfolio}_{metric}"] = {
                    "delay": best_is_delay,
                    "value": round(best_is_value, 4),
                }
                best_delays["OOS"][f"{portfolio}_{metric}"] = {
                    "delay": best_oos_delay,
                    "value": round(best_oos_value, 4),
                }

        return pd.DataFrame(best_delays)

    def plot_delay_comparison(self, results: Dict[str, pd.DataFrame], metrics: List[str] = None) -> None:
        """각 delay별 IS/OOS 성과 비교 플롯"""
        # 기본 성과 지표
        performance_metrics = ["Annual Return", "Sharpe Ratio", "Win Rate", "Max Drawdown"]
        # IC 관련 지표 (High 포트폴리오만)
        ic_metrics = ["IC (Mean)", "Rank IC (Mean)"]
        
        # 성과 지표 플롯
        for metric in performance_metrics:
            fig, axes = plt.subplots(3, 1, figsize=(12, 18))
            portfolios = ["High", "Low", "H-L"]

            for idx, portfolio in enumerate(portfolios):
                ax = axes[idx]
                find_max = portfolio != "Low"  # Low 포트폴리오는 최소값을 찾습니다

                # 데이터 추출
                is_df = results["IS"]
                oos_df = results["OOS"]
                
                is_values = []
                oos_values = []
                delays = []
                
                for col in is_df.columns:
                    if col[1] == portfolio:
                        delays.append(col[0])
                        is_values.append(is_df.loc[metric, col])
                        oos_values.append(oos_df.loc[metric, col])
                
                is_values = np.array(is_values)
                oos_values = np.array(oos_values)

                # 데이터 플롯
                ax.plot(delays, is_values, "b-o", label="In-Sample")
                ax.plot(delays, oos_values, "r-o", label="Out-of-Sample")

                # 최적 포인트 표시
                if len(is_values) > 0:
                    best_is_idx = is_values.argmax() if find_max else is_values.argmin()
                    ax.plot(
                        delays[best_is_idx],
                        is_values[best_is_idx],
                        "b*",
                        markersize=15,
                        label=f"Best IS: {delays[best_is_idx]}",
                    )

                if len(oos_values) > 0:
                    best_oos_idx = oos_values.argmax() if find_max else oos_values.argmin()
                    ax.plot(
                        delays[best_oos_idx],
                        oos_values[best_oos_idx],
                        "r*",
                        markersize=15,
                        label=f"Best OOS: {delays[best_oos_idx]}",
                    )

                ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
                
                ax.set_xticks(delays)
                ax.set_xticklabels(delays)
                
                ax.set_xlabel("Delay (Trading Days)")
                ax.set_title(f"{metric} by Delay - {portfolio} Portfolio")
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = f"./processed/delay_comparison_{metric.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

        # IC 관련 지표 플롯 (High 포트폴리오만)
        for metric in ic_metrics:
            plt.figure(figsize=(12, 6))
            
            # High 포트폴리오 데이터만 추출
            is_values = []
            oos_values = []
            delays = []
            
            for col in results["IS"].columns:
                if col[1] == "High":
                    delays.append(col[0])
                    is_values.append(results["IS"].loc[metric, col])
                    oos_values.append(results["OOS"].loc[metric, col])
            
            is_values = np.array(is_values)
            oos_values = np.array(oos_values)

            # 데이터 플롯
            plt.plot(delays, is_values, "b-o", label="In-Sample")
            plt.plot(delays, oos_values, "r-o", label="Out-of-Sample")

            # 최적 포인트 표시
            if len(is_values) > 0:
                best_is_idx = is_values.argmax()
                plt.plot(
                    delays[best_is_idx],
                    is_values[best_is_idx],
                    "b*",
                    markersize=15,
                    label=f"Best IS: {delays[best_is_idx]}",
                )

            if len(oos_values) > 0:
                best_oos_idx = oos_values.argmax()
                plt.plot(
                    delays[best_oos_idx],
                    oos_values[best_oos_idx],
                    "r*",
                    markersize=15,
                    label=f"Best OOS: {delays[best_oos_idx]}",
                )

            plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
            
            plt.xticks(delays, delays)
            
            plt.xlabel("Delay (Trading Days)")
            plt.title(f"{metric} by Delay - High Portfolio")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = f"./processed/delay_comparison_{metric.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

    def plot_cumulative_returns(self, portfolio_rets: pd.DataFrame, title: str) -> None:
        """누적 수익률 플롯"""
        cum_rets = (1 + portfolio_rets).cumprod() - 1
        plt.figure(figsize=(12, 6))
        for col in ["High", "Low", "H-L"]:
            plt.plot(cum_rets.index, cum_rets[col], label=col)

        plt.title(f"Cumulative Returns: {title}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = f"./processed/portfolio_performance_{title.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# 데이터 로드
prob_df = pd.read_csv("./processed/ensem_res.csv")
ret_df = pd.read_csv("./processed/return_df.csv")

# 날짜를 datetime으로 변환하고 인덱스로 설정
prob_df["investment_date"] = pd.to_datetime(prob_df["investment_date"])
prob_df.set_index("investment_date", inplace=True)

ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df.set_index("date", inplace=True)

# 백테스트 초기화
backtest = PortfolioBacktest(prob_df, ret_df, n_stocks=500)

# 비교 수행
results = backtest.compare_delays_with_periods(
    delays=[0, 1, 2, 3, 5, 10, 20], cut=10
)

# 비교 플롯 생성
for metric in ["Annual Return", "Sharpe Ratio", "Win Rate"]:
    backtest.plot_delay_comparison(results, metric)

# 특정 delay에 대한 누적 수익률 플롯 (예: delay = -20)
portfolio_rets = backtest.generate_portfolio_with_delay(delay=0, cut=10)
backtest.plot_cumulative_returns(portfolio_rets, title="Delay 1 Trading Days")

# 포트폴리오 성과 지표 계산
metrics = backtest.calculate_metrics(portfolio_rets, delay=0)
print("\nPortfolio Performance Metrics:")
print(metrics.round(4))

# IC 분포 시각화
backtest.plot_ic_distribution(results)