import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

import scienceplots
plt.style.use(['science', 'no-latex'])


class PortfolioBackend:
    def __init__(self, prob_df: pd.DataFrame, ret_df: pd.DataFrame, n_stocks: int = 500):
        """
        Args:
            prob_df: 확률 데이터 (index=investment_date, columns=StockID)
            ret_df: 일별 수익률 데이터 (index=date, columns=StockID)
            n_stocks: 포트폴리오 내 종목 수
        """
        self.prob_df = prob_df.copy()
        
        # return에 이상치 필터링 t stat 95 범위 내
        self.ret_df = ret_df.copy()
        
        # 각 종목별로 t-statistic 계산 및 이상치 필터링
        for col in self.ret_df.columns:
            returns = self.ret_df[col].dropna()
            if len(returns) > 0:
                mean = returns.mean()
                std = returns.std()
                t_stat = (returns - mean) / std
                # 95% 신뢰구간 (t-statistic ±1.96)
                mask = (t_stat >= -1.96) & (t_stat <= 1.96)
                # 이상치를 NaN으로 대체
                self.ret_df.loc[returns.index[~mask], col] = np.nan
        
        self.n_stocks = n_stocks
        self.freq = 'month'
        self.prob_df.index = pd.to_datetime(self.prob_df.index)
        self.ret_df.index = pd.to_datetime(self.ret_df.index)

        print("prob_df date range:", self.prob_df.index.min(), "to", self.prob_df.index.max())
        print("ret_df date range:", self.ret_df.index.min(), "to", self.ret_df.index.max())
        print("\n=== Return Statistics ===")
        print("Before filtering - Number of NaN:", self.ret_df.isna().sum().sum())
        print("After filtering - Number of NaN:", self.ret_df.isna().sum().sum())
        print("Filtered return range:", self.ret_df.min().min(), "to", self.ret_df.max().max())

        self.start_date = self.prob_df.index.min()
        self.end_date = self.prob_df.index.max()
        self.cutoff_date = pd.Timestamp("2018-01-01")
        print("Cutoff date set to:", self.cutoff_date)

        self.trading_days = self.ret_df.index.sort_values()

    def get_trading_day_offset(self, date: pd.Timestamp, offset: int) -> Optional[pd.Timestamp]:
        date = pd.Timestamp(date)
        try:
            idx = self.trading_days.get_loc(date)
        except KeyError:
            idx = self.trading_days.get_indexer([date], method="bfill")[0]
            if idx == -1:
                return None
        target_idx = idx + offset
        if 0 <= target_idx < len(self.trading_days):
            return self.trading_days[target_idx]
        return None

    def analyze_delay_performance(self, delays: List[int], cut: int = 10) -> Dict[str, pd.DataFrame]:
        """
        지연(delay)별 포트폴리오 성과 분석 (IS 및 OOS)
        """
        is_results, oos_results = [], []
        full_prob_df = self.prob_df.copy()
        full_ret_df = self.ret_df.copy()

        for delay in delays:
            print(f"Analyzing delay: {delay} trading days")
            # In-Sample
            is_prob = full_prob_df[full_prob_df.index < self.cutoff_date]
            is_ret = full_ret_df[(full_ret_df.index >= self.start_date) & (full_ret_df.index < self.cutoff_date)]
            is_portfolio_rets = self._generate_portfolio_with_delay(is_prob, is_ret, delay, cut)
            if not is_portfolio_rets.empty:
                is_metrics = self._calculate_metrics(is_portfolio_rets, delay).round(4)
                is_metrics.columns = pd.MultiIndex.from_product([[delay], is_metrics.columns])
                is_results.append(is_metrics)

            # Out-of-Sample
            oos_prob = full_prob_df[full_prob_df.index >= self.cutoff_date]
            oos_ret = full_ret_df[full_ret_df.index >= self.cutoff_date]
            oos_portfolio_rets = self._generate_portfolio_with_delay(oos_prob, oos_ret, delay, cut)
            if not oos_portfolio_rets.empty:
                oos_metrics = self._calculate_metrics(oos_portfolio_rets, delay).round(4)
                oos_metrics.columns = pd.MultiIndex.from_product([[delay], oos_metrics.columns])
                oos_results.append(oos_metrics)

        return {"IS": pd.concat(is_results, axis=1), "OOS": pd.concat(oos_results, axis=1)}

    def find_optimal_delay(self, delay_results: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        최적 지연(optimal delay) 탐색 (IS 및 OOS)
        """
        is_df = delay_results["IS"]
        oos_df = delay_results["OOS"]
        best_delays = {"IS": {}, "OOS": {}}
        metrics = is_df.index
        portfolios = ["High", "Low", "H-L"]

        for metric in metrics:
            for portfolio in portfolios:
                find_max = portfolio != "Low"
                is_values = is_df.loc[metric, pd.IndexSlice[:, portfolio]]
                if is_values.notna().any():
                    best_is_idx = is_values.values.argmax() if find_max else is_values.values.argmin()
                    best_is_delay = is_values.index[best_is_idx][0]
                    best_is_value = is_values.iloc[best_is_idx]
                else:
                    best_is_delay = np.nan
                    best_is_value = np.nan

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
        return best_delays

    def analyze_quantile_portfolios(self, delay: int, n_quantiles: int = 10) -> Dict[str, pd.DataFrame]:
        """
        특정 delay를 사용하여 10분위(quantile)별 포트폴리오 성과 분석 (OOS 기간)
        """
        full_prob_df = self.prob_df.copy()
        full_ret_df = self.ret_df.copy()

        # OOS 기간만 사용
        oos_prob = full_prob_df[full_prob_df.index >= self.cutoff_date]
        oos_ret = full_ret_df[full_ret_df.index >= self.cutoff_date]
        oos_portfolio_rets = self._generate_quantile_portfolios(oos_prob, oos_ret, delay, n_quantiles)
        oos_metrics = self._calculate_metrics(oos_portfolio_rets, delay).round(4)
        return {"OOS": oos_metrics, "OOS_returns": oos_portfolio_rets}

    def _generate_portfolio_with_delay(
        self, prob_df: pd.DataFrame, ret_df: pd.DataFrame, delay: int, cut: int = 10
    ) -> pd.DataFrame:
        portfolio_returns = []
        prob_dates = prob_df.index.sort_values()

        for i, date in enumerate(prob_dates):
            probs = prob_df.loc[date].dropna()
            if probs.empty:
                continue

            investment_date = self.get_trading_day_offset(date, delay)
            if investment_date is None:
                continue

            next_date = prob_dates[i + 1] if i + 1 < len(prob_dates) else self.end_date
            next_investment_date = self.get_trading_day_offset(next_date, delay) or self.end_date

            mask = (ret_df.index >= investment_date) & (ret_df.index < next_investment_date)
            holding_period_returns = ret_df.loc[mask]
            if holding_period_returns.empty:
                continue

            cut_size = len(probs) // cut
            high_stocks = list(set(probs.nlargest(cut_size).index) & set(holding_period_returns.columns))
            low_stocks = list(set(probs.nsmallest(cut_size).index) & set(holding_period_returns.columns))
            if not high_stocks or not low_stocks:
                continue

            high_ret = (1 + holding_period_returns[high_stocks]).prod() - 1
            low_ret = (1 + holding_period_returns[low_stocks]).prod() - 1

            portfolio_returns.append({
                "investment_date": investment_date,
                "High": high_ret.mean(),
                "Low": low_ret.mean(),
                "H-L": high_ret.mean() - low_ret.mean(),
            })
        return pd.DataFrame(portfolio_returns).set_index("investment_date")

    def _generate_quantile_portfolios(
        self, prob_df: pd.DataFrame, ret_df: pd.DataFrame, delay: int, n_quantiles: int = 10
    ) -> pd.DataFrame:
        portfolio_returns = []
        prob_dates = prob_df.index.sort_values()

        for i, date in enumerate(prob_dates):
            probs = prob_df.loc[date].dropna()
            if probs.empty:
                continue

            investment_date = self.get_trading_day_offset(date, delay)
            if investment_date is None:
                continue

            next_date = prob_dates[i + 1] if i + 1 < len(prob_dates) else self.end_date
            next_investment_date = self.get_trading_day_offset(next_date, delay) or self.end_date

            mask = (ret_df.index >= investment_date) & (ret_df.index < next_investment_date)
            holding_period_returns = ret_df.loc[mask]
            if holding_period_returns.empty:
                continue

            available = set(holding_period_returns.columns)
            portfolio_dict = {"investment_date": investment_date}

            # 가장 높은 확률 포트폴리오 (High)
            high_stocks = list(set(probs.nlargest(len(probs) // n_quantiles).index) & available)
            if high_stocks:
                high_ret = (1 + holding_period_returns[high_stocks]).prod() - 1
                portfolio_dict["High"] = high_ret.mean()

            # 가장 낮은 확률 포트폴리오 (Low) 
            low_stocks = list(set(probs.nsmallest(len(probs) // n_quantiles).index) & available)
            if low_stocks:
                low_ret = (1 + holding_period_returns[low_stocks]).prod() - 1
                portfolio_dict["Low"] = low_ret.mean()

            # High-Low 포트폴리오
            if high_stocks and low_stocks:
                portfolio_dict["H-L"] = portfolio_dict["High"] - portfolio_dict["Low"]

            # 나머지 포트폴리오들 (2 ~ n_quantiles-1)
            sorted_probs = probs.sort_values(ascending=False)
            for q in range(1, n_quantiles-1):
                start = int(len(probs) * q / n_quantiles)
                end = int(len(probs) * (q + 1) / n_quantiles)
                q_stocks = list(set(sorted_probs.iloc[start:end].index) & available)
                if q_stocks:
                    q_ret = (1 + holding_period_returns[q_stocks]).prod() - 1
                    portfolio_dict[str(q+1)] = q_ret.mean()

            portfolio_returns.append(portfolio_dict)
        return pd.DataFrame(portfolio_returns).set_index("investment_date")

    def _calculate_metrics(self, returns_df: pd.DataFrame, delay: int) -> pd.DataFrame:
        if returns_df.empty:
            return pd.DataFrame()
        annual_factor = 252
        metrics_dict = {}

        for portfolio in returns_df.columns:
            rets = returns_df[portfolio].dropna()
            if len(rets) < 2:
                metrics_dict[portfolio] = {
                    "Annual Return": np.nan,
                    "Annual Volatility": np.nan,
                    "Sharpe Ratio": np.nan,
                    "Win Rate": np.nan,
                    "Max Drawdown": np.nan,
                    "IC (Mean)": np.nan,
                    "IC (Std)": np.nan,
                    "IC (t-stat)": np.nan,
                    "Rank IC (Mean)": np.nan,
                    "Rank IC (Std)": np.nan,
                    "Rank IC (t-stat)": np.nan,
                }
                continue

            daily_return = rets.mean()
            daily_vol = rets.std()
            annual_return = daily_return * annual_factor
            annual_vol = daily_vol * np.sqrt(annual_factor)
            sharpe_ratio = (annual_return / annual_vol) if annual_vol != 0 else np.nan
            win_rate = (rets > 0).mean()
            cum_returns = (1 + rets).cumprod()
            drawdown = cum_returns / cum_returns.cummax() - 1
            max_drawdown = drawdown.min()

            # IC 계산
            ic_values = []
            rank_ic_values = []
            for i in range(len(rets)-1):
                current_date = rets.index[i]
                next_date = rets.index[i+1]
                
                # 현재 시점의 확률과 다음 시점의 수익률 간의 상관관계
                try:
                    current_probs = self.prob_df.loc[:current_date].iloc[-1].dropna()
                    next_rets = self.ret_df.loc[next_date, current_probs.index].dropna()
                    
                    # 공통 종목만 선택
                    common_stocks = list(set(current_probs.index) & set(next_rets.index))
                    if len(common_stocks) > 0:
                        ic = spearmanr(current_probs[common_stocks], next_rets[common_stocks])[0]
                        ic_values.append(ic)
                        
                        # Rank IC 계산
                        rank_ic = spearmanr(current_probs[common_stocks].rank(), next_rets[common_stocks].rank())[0]
                        rank_ic_values.append(rank_ic)
                except (KeyError, IndexError):
                    continue

            ic_mean = np.mean(ic_values) if ic_values else np.nan
            ic_std = np.std(ic_values) if ic_values else np.nan
            ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_values))) if ic_values and ic_std != 0 else np.nan
            
            rank_ic_mean = np.mean(rank_ic_values) if rank_ic_values else np.nan
            rank_ic_std = np.std(rank_ic_values) if rank_ic_values else np.nan
            rank_ic_tstat = rank_ic_mean / (rank_ic_std / np.sqrt(len(rank_ic_values))) if rank_ic_values and rank_ic_std != 0 else np.nan

            metrics_dict[portfolio] = {
                "Annual Return": annual_return,
                "Annual Volatility": annual_vol,
                "Sharpe Ratio": sharpe_ratio,
                "Win Rate": win_rate,
                "Max Drawdown": max_drawdown,
                "IC (Mean)": ic_mean,
                "IC (Std)": ic_std,
                "IC (t-stat)": ic_tstat,
                "Rank IC (Mean)": rank_ic_mean,
                "Rank IC (Std)": rank_ic_std,
                "Rank IC (t-stat)": rank_ic_tstat,
            }
        return pd.DataFrame(metrics_dict)


class PortfolioVisualizer:
    def __init__(self, save_dir: str = "./processed"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_delay_analysis(self, delay_results: Dict[str, pd.DataFrame]) -> None:
        metrics = ["Annual Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"]
        portfolios = ["High", "Low", "H-L"]

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        for port_idx, portfolio in enumerate(portfolios):
            for met_idx, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[port_idx, met_idx])
                is_values, oos_values, delays = [], [], []
                for col in delay_results["IS"].columns:
                    if col[1] == portfolio:
                        delays.append(col[0])
                        is_values.append(delay_results["IS"].loc[metric, col])
                        oos_values.append(delay_results["OOS"].loc[metric, col])
                ax.plot(delays, is_values, "b-o", label="In-Sample", markersize=4)
                ax.plot(delays, oos_values, "r-o", label="Out-of-Sample", markersize=4)
                ax.set_xticks(delays)
                ax.set_xticklabels([int(d) for d in delays], rotation=45)
                ax.set_xlabel("Delay (Trading Days)", fontsize=9)
                ax.set_ylabel(metric, fontsize=9)
                if port_idx == 0:
                    ax.set_title(f"({chr(97 + met_idx)}) {metric}", fontsize=10)
                if met_idx == 0:
                    ax.text(-0.2, 0.5, portfolio, rotation=90,
                            transform=ax.transAxes, fontsize=10, verticalalignment='center')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.tick_params(labelsize=8)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        save_path = os.path.join(self.save_dir, "delay_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_oos_cumulative_returns(self, quantile_results: Dict[str, pd.DataFrame]) -> None:
        """
        OOS 기간의 누적 수익률을 시각화 (일별 데이터, 필요시 월별로 resample 가능)
        """
        oos_returns = quantile_results["OOS_returns"]
        # 월별로 집계하려면 아래 주석 해제
        # oos_returns = oos_returns.resample('M').last().pct_change().dropna()
        cum_rets = (1 + oos_returns).cumprod() - 1
        
        # 색상 맵 생성 (칼럼 수에 맞게 동적으로)
        n_columns = len(cum_rets.columns)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_columns))
        
        plt.figure(figsize=(12, 6))
        
        # 칼럼 순서 정의 (High, 중간 분위수들, Low, H-L 순서로)
        middle_quantiles = [col for col in cum_rets.columns if col not in ['High', 'Low', 'H-L']]
        plot_order = ['High'] + middle_quantiles + ['Low', 'H-L']
        
        for i, portfolio in enumerate(plot_order):
            if portfolio in cum_rets.columns:
                plt.plot(cum_rets.index, cum_rets[portfolio],
                        label=portfolio, color=colors[i])
            
        # legend 표시
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("OOS Cumulative Returns (Quantile Portfolios)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "oos_cumulative_returns.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    # 데이터 로드 및 전처리
    prob_df = pd.read_csv("./processed/ensem_res_20D20P.csv")
    ret_df = pd.read_csv("./processed/return_df.csv")
    prob_df["investment_date"] = pd.to_datetime(prob_df["investment_date"])
    prob_df.set_index("investment_date", inplace=True)
    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df.set_index("date", inplace=True)

    backend = PortfolioBackend(prob_df, ret_df, n_stocks=500)

    # 1. Delay에 따른 성과 분석 (IS & OOS)
    delays = [0, 1, 2, 3, 5, 10, 20]
    delay_results = backend.analyze_delay_performance(delays, cut=10)
    optimal_delays = backend.find_optimal_delay(delay_results)
    
    # 결과 저장
    save_dir = "./processed"
    os.makedirs(save_dir, exist_ok=True)
    
    # Delay 분석 결과 저장
    delay_results["IS"].to_csv(os.path.join(save_dir, "delay_analysis_is.csv"))
    delay_results["OOS"].to_csv(os.path.join(save_dir, "delay_analysis_oos.csv"))
    pd.DataFrame(optimal_delays).to_csv(os.path.join(save_dir, "optimal_delays.csv"))
    
    print("\nOptimal Delays:")
    print(pd.DataFrame(optimal_delays))

    # 2. 특정 delay를 사용하여 10분위 포트폴리오 성과 분석 (OOS)
    specific_delay = 0
    quantile_results = backend.analyze_quantile_portfolios(specific_delay, n_quantiles=10)
    
    # Quantile 포트폴리오 결과 저장
    quantile_results["OOS"].to_csv(os.path.join(save_dir, "quantile_portfolio_metrics.csv"))
    quantile_results["OOS_returns"].to_csv(os.path.join(save_dir, "quantile_portfolio_returns.csv"))

    # 3. 시각화: 기존 delay 분석 및 OOS 누적 수익률
    visualizer = PortfolioVisualizer()
    visualizer.plot_delay_analysis(delay_results)
    visualizer.plot_oos_cumulative_returns(quantile_results)
    
    print("\nResults have been saved to the 'processed' directory:")
    print("1. delay_analysis_is.csv - In-Sample delay analysis results")
    print("2. delay_analysis_oos.csv - Out-of-Sample delay analysis results")
    print("3. optimal_delays.csv - Optimal delay values")
    print("4. quantile_portfolio_metrics.csv - Quantile portfolio performance metrics")
    print("5. quantile_portfolio_returns.csv - Quantile portfolio returns")
    print("6. delay_analysis.png - Delay analysis plots")
    print("7. oos_cumulative_returns.png - OOS cumulative returns plot")


if __name__ == "__main__":
    main()
