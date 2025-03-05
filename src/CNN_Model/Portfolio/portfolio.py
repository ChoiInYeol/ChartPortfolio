import os
import os.path as op
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Misc import utilities as ut
from Data import equity_data as eqd
from Data import dgp_config as dcf


class PortfolioManager(object):
    def __init__(
        self,
        signal_df: pd.DataFrame,
        freq: str,
        portfolio_dir: str,
        start_year=2018,
        end_year=2024,
        country="USA",
        delay_list=None,
        load_signal=True,
        custom_ret=None,
        transaction_cost=False,
        model_name=None
    ):
        assert freq in ["week", "month", "quarter"]
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.start_year = start_year
        self.end_year = end_year
        self.country = country
        self.no_delay_ret_name = f"next_{freq}_ret"
        self.custom_ret = custom_ret
        self.delay_list = [0] if delay_list is None else delay_list
        self.transaction_cost = transaction_cost
        self.result_dir = os.path.join(dcf.WORK_DIR, f"portfolio_results_{self.country}")
        self.model_name = model_name
        os.makedirs(self.result_dir, exist_ok=True)
        if load_signal:
            assert "up_prob" in signal_df.columns
            self.signal_df = self.get_up_prob_with_period_ret(signal_df)

    def __add_period_ret_to_us_res_df_w_delays(self, signal_df):
        period_ret = eqd.get_period_ret(self.freq, country=self.country)
        columns = [
            f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list
        ]
        if self.custom_ret is not None:
            columns += [self.custom_ret]
        signal_df = signal_df.copy()
        signal_df[columns] = period_ret[columns]
        signal_df[self.no_delay_ret_name] = signal_df[f"next_{self.freq}_ret_0delay"]
        signal_df.fillna(0, inplace=True) # 의심
        for dl in self.delay_list:
            dl_ret_name = f"next_{self.freq}_ret_{dl}delay"
            print(
                f"{len(signal_df)} samples, {dl} delay \
                    nan values {np.sum(signal_df[dl_ret_name].isna())},\
                    zero values {np.sum(signal_df[dl_ret_name] == 0.)}"
            )
        return signal_df.copy()

    def get_up_prob_with_period_ret(self, signal_df):
        signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(
                range(self.start_year, self.end_year + 1)
            )
        ]
        signal_df = self.__add_period_ret_to_us_res_df_w_delays(signal_df)
        if self.country not in ["future", "new_future"]:
            signal_df.MarketCap = signal_df.MarketCap.abs()
            signal_df = signal_df[~signal_df.MarketCap.isnull()].copy()
        return signal_df

    def calculate_portfolio_rets(self, weight_type, cut=10, delay=0):
        assert weight_type in ["ew"]
        assert delay in self.delay_list
        if self.custom_ret:
            print(f"Calculating portfolio using {self.custom_ret}")
            ret_name = self.custom_ret
        else:
            ret_name = (
                self.no_delay_ret_name
                if delay == 0
                else f"next_{self.freq}_ret_{delay}delay"
            )
        df = self.signal_df.copy()

        def __get_decile_df_with_inv_ret(reb_df, decile_idx):
            rebalance_up_prob = reb_df.up_prob
            low = np.percentile(rebalance_up_prob, decile_idx * 100.0 / cut)
            high = np.percentile(rebalance_up_prob, (decile_idx + 1) * 100.0 / cut)
            if decile_idx == 0:
                pf_filter = (rebalance_up_prob >= low) & (rebalance_up_prob <= high)
            else:
                pf_filter = (rebalance_up_prob > low) & (rebalance_up_prob <= high)
            _decile_df = reb_df[pf_filter].copy()
            if weight_type == "ew":
                stock_num = len(_decile_df)
                _decile_df["weight"] = (
                    pd.Series(
                        np.ones(stock_num), dtype=np.float64, index=_decile_df.index
                    )
                    / stock_num
                )
            else:
                value = _decile_df.MarketCap
                _decile_df["weight"] = pd.Series(value, dtype=np.float64) / np.sum(value)
            _decile_df["inv_ret"] = _decile_df["weight"] * _decile_df[ret_name]
            return _decile_df

        dates = np.sort(np.unique(df.index.get_level_values("Date")))
        print(f"Calculating portfolio from {dates[0]}, {dates[1]} to {dates[-1]}")
        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        prev_to_df = None
        prob_ret_corr = []
        prob_ret_pearson_corr = []
        prob_inv_ret_corr = []
        prob_inv_ret_pearson_corr = []

        for i, date in enumerate(dates):
            rebalance_df = df.loc[date].copy()
            rank_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="spearman"
            )
            pearson_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="pearson"
            )
            prob_ret_corr.append(rank_corr)
            prob_ret_pearson_corr.append(pearson_corr)

            low = np.percentile(rebalance_df.up_prob, 10)
            high = np.percentile(rebalance_df.up_prob, 90)
            if low == high:
                print(f"Skipping {date}")
                continue
            for j in range(cut):
                decile_df = __get_decile_df_with_inv_ret(rebalance_df, j)
                if self.transaction_cost:
                    if j == cut - 1:
                        decile_df["inv_ret"] -= (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                    elif j == 0:
                        decile_df["inv_ret"] += (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                portfolio_ret.loc[date, j] = np.sum(decile_df["inv_ret"])

            sell_decile = __get_decile_df_with_inv_ret(rebalance_df, 0)
            buy_decile = __get_decile_df_with_inv_ret(rebalance_df, cut - 1)
            buy_sell_decile = pd.concat([sell_decile, buy_decile])
            prob_inv_ret_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="spearman")
            )
            prob_inv_ret_pearson_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="pearson")
            )

            sell_decile[["weight", "inv_ret"]] = sell_decile[["weight", "inv_ret"]] * (
                -1
            )
            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                tto_df = pd.DataFrame(
                    index=np.unique(list(to_df.index) + list(prev_to_df.index))
                )
                try:
                    tto_df["cur_weight"] = to_df["weight"]
                except ValueError:
                    pdb.set_trace()
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[
                    ["weight", ret_name, "inv_ret"]
                ]
                tto_df.fillna(0, inplace=True)
                denom = 1 + np.sum(tto_df["inv_ret"])
                turnover[i - 1] = np.sum(
                    (
                        tto_df["cur_weight"]
                        - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom
                    ).abs()
                )
                turnover[i - 1] = turnover[i - 1] * 0.5
            prev_to_df = to_df
            
        pd.set_option('future.no_silent_downcasting', True)
        portfolio_ret = portfolio_ret.fillna(0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]
        print(
            f"Spearman Corr between Prob and Stock Return is {np.nanmean(prob_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Stock Return is {np.nanmean(prob_ret_pearson_corr):.4f}"
        )
        print(
            f"Spearman Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_pearson_corr):.4f}"
        )
        
        return portfolio_ret, np.mean(turnover)

    @staticmethod
    def _ret_to_cum_log_ret(rets):
        log_rets = np.log(rets.astype(float) + 1)
        return log_rets.cumsum()

    def make_portfolio_plot(
            self, portfolio_ret, cut=None, weight_type=None, save_path=None, plot_title=None
        ):
        for weight_type in ["ew"]:
            pf_name = self.get_portfolio_name(weight_type, delay=0, cut=10)
            print(f"Calculating {pf_name}")
            portfolio_ret, turnover = self.calculate_portfolio_rets(weight_type, cut=10, delay=0)
        
            model_prefix = self.model_name if self.model_name else ""
            file_name = f"{model_prefix}_{self.country}_{self.freq}_{weight_type}_cut{cut}_{self.start_year}-{self.end_year}.png"
            save_path = os.path.join(self.result_dir, file_name)
            plot_title = f'{model_prefix} {self.freq} "Re(-)Imag(in)ing Price Trend" {self.country} {weight_type} cut{cut} {self.start_year}-{self.end_year}'
            
            ret_name = "nxt_freq_ewret"
            df = portfolio_ret.copy()
            df.columns = ["Low(L)"] + [str(i) for i in range(2, cut)] + ["High(H)", "H-L"]
            
            # SPY와 Benchmark 데이터 가져오기
            bench = eqd.get_bench_freq_rets(self.freq)
            spy = eqd.get_spy_freq_rets(self.freq)
            
            # 포트폴리오 데이터의 시작과 끝 날짜로 SPY와 Benchmark 데이터 자르기
            start_date = df.index[0]
            end_date = df.index[-1]
            bench = bench.loc[start_date:end_date]
            spy = spy.loc[start_date:end_date]
            
            df["SPY"] = spy[ret_name]
            df["Benchmark"] = bench[ret_name]
            df.dropna(inplace=True)
            
            # 누적 로그 수익률 계산
            log_ret_df = pd.DataFrame(index=df.index)
            for column in df.columns:
                log_ret_df[column] = self._ret_to_cum_log_ret(df[column])
            
            # 모든 시리즈가 0부터 시작하도록 조정
            prev_year = pd.to_datetime(log_ret_df.index[0]).year - 1
            prev_day = pd.to_datetime(f"{prev_year}-12-31")
            log_ret_df.loc[prev_day] = 0
            log_ret_df = log_ret_df.sort_index()
            
            columns_to_plot = ["High(H)", "Low(L)", "H-L", "SPY", "Benchmark"]
            plot = log_ret_df[columns_to_plot].plot(
                style={"SPY": "y", "Benchmark": "g", "High(H)": "b", "Low(L)": "r", "H-L": "k"},
                lw=1,
                title=plot_title,
            )
            plot.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Portfolio plot saved to {save_path}")

    def portfolio_res_summary(self, portfolio_ret, turnover, cut=10):
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        res = np.zeros((cut + 1, 3))
        period = 52 if self.freq == "week" else 12 if self.freq == "month" else 4
        res[:, 0] = avg * period
        res[:, 1] = std * np.sqrt(period)
        res[:, 2] = res[:, 0] / res[:, 1]

        summary_df = pd.DataFrame(res, columns=["ret", "std", "SR"])
        summary_df = summary_df.set_index(
            pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L"])
        )
        summary_df.loc["Turnover", "SR"] = turnover / (
            1 / 4
            if self.freq == "week"
            else 1
            if self.freq == "month"
            else 3
            if self.freq == "quarter"
            else 12
        )
        print(summary_df)
        return summary_df

    def generate_portfolio(self, cut=10, delay=0):
        assert delay in self.delay_list
        for weight_type in ["ew"]:
            pf_name = self.get_portfolio_name(weight_type, delay, cut)
            print(f"Calculating {pf_name}")
            portfolio_ret, turnover = self.calculate_portfolio_rets(
                weight_type, cut=cut, delay=delay
            )
            data_dir = ut.get_dir(op.join(self.portfolio_dir, "pf_data"))
            portfolio_ret.to_csv(op.join(data_dir, f"pf_data_{pf_name}.csv"))

            summary_df = self.portfolio_res_summary(portfolio_ret, turnover, cut)
            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            print(f"Summary saved to {smry_path}")
            summary_df.to_csv(smry_path)
            with open(os.path.join(self.portfolio_dir, f"{pf_name}.txt"), "w+") as file:
                summary_df = summary_df.astype(float).round(2)
                file.write(ut.to_latex_w_turnover(summary_df, cut=cut))

    def get_portfolio_name(self, weight_type, delay, cut):
        assert weight_type.lower() in ["ew"]
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_surfix = "" if cut == 10 else f"_{cut}cut"
        custom_ret_surfix = "" if self.custom_ret is None else self.custom_ret
        tc_surfix = "" if not self.transaction_cost else "_w_transaction_cost"
        pf_name = f"{delay_prefix}{weight_type.lower()}{cut_surfix}{custom_ret_surfix}{tc_surfix}"
        return pf_name

    def load_portfolio_ret(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        try:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}.csv"),
                index_col=0,
            )
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}_100.csv"),
                index_col=0,
            )
        df.index = pd.to_datetime(df.index)
        return df

    def load_portfolio_summary(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        try:
            df = pd.read_csv(op.join(self.portfolio_dir, f"{pf_name}.csv"), index_col=0)
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, f"{pf_name}_100.csv"), index_col=0
            )
        return df


def main():
    pass


if __name__ == "__main__":
    main()