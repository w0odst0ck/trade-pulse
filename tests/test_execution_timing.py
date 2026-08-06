#!/usr/bin/env python3
"""execution_timing.py 单元测试（unittest + pandas，无网络）

覆盖：
  1. parse_min_time —— min15 time 字段解析
  2. aggregate_intraday_bars —— 盘中 bar 聚合逻辑（截至 14:30 剔除尾部 bar）
  3. run_backtest_timing 成交价选取 —— t_close 用特征日收盘 / t1_close 用次日收盘
  4. 参数化回测与生产 run_backtest 同构（t1_close 模式输出全等）
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools' / 'daily_pipeline'))

from tools.daily_pipeline import execution_timing as et  # noqa: E402
from tools.daily_pipeline import backtest  # noqa: E402


# 迷你回测配置：固定阈值（无自适应），风控关闭
BASE_CONFIG = {
    'thresholds': {
        'buy': 0.1,
        'sell': -0.07,
        'confirm_days': 2,
        'weekly_filter_percentile': 0.2,
    },
    'adaptive_thresholds': {'enabled': False},
    'weekly_modifier': {'enabled': True, 'min_modifier': -0.3, 'max_modifier': 0.3},
    'risk_control': {'enabled': False},
}


def make_mini_features(n_days: int = 15) -> pd.DataFrame:
    """迷你特征序列：0-4 天空仓观望，5-9 天买入信号(0.5)，10-14 天卖出信号(-0.5)。

    状态机：空仓 → 0.5>0.1 即买入；持仓 → -0.5<-0.07 连续 2 天确认后卖出。
    """
    dates = pd.bdate_range('2024-01-02', periods=n_days)
    close = np.round(np.linspace(1.0, 1.3, n_days), 4)
    total_score = ([0.0] * 5 + [0.5] * 5 + [-0.5] * 5)[:n_days]
    return pd.DataFrame({
        'date': dates,
        'close': close,
        'total_score': total_score,
        'weekly_modifier': [0.0] * n_days,
        'ma60_slope': [0.0] * n_days,
    })


class TestParseMinTime(unittest.TestCase):
    def test_hhmm_extraction(self):
        self.assertEqual(et.parse_min_time(20260104094500000), 945)
        self.assertEqual(et.parse_min_time(20260104143000000), 1430)
        self.assertEqual(et.parse_min_time(20260104150000000), 1500)
        self.assertEqual(et.parse_min_time('20260104093000000'), 930)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            et.parse_min_time(12345)


class TestAggregateIntradayBars(unittest.TestCase):
    def _make_min15(self):
        """两个交易日；第二日含 14:30 及之后的两根 bar（14:25 决策时均不可见，应被剔除）"""
        rows = [
            # date, time, open, high, low, close, volume, amount
            ('2026-01-05', 20260105094500000, 1.40, 1.44, 1.38, 1.42, 100, 1000),
            ('2026-01-05', 20260105100000000, 1.42, 1.46, 1.41, 1.45, 150, 1500),
            ('2026-01-05', 20260105143000000, 1.45, 1.50, 1.44, 1.48, 200, 2000),
            ('2026-01-05', 20260105150000000, 1.48, 1.52, 1.47, 1.51, 300, 3000),  # 剔除
            ('2026-01-06', 20260106094500000, 1.50, 1.53, 1.49, 1.52, 120, 1200),
            ('2026-01-06', 20260106150000000, 1.55, 1.58, 1.54, 1.57, 400, 4000),  # 剔除
        ]
        return pd.DataFrame(rows, columns=['date', 'time', 'open', 'high', 'low', 'close',
                                           'volume', 'amount'])

    def test_aggregation_with_cutoff(self):
        # 14:25 决策语义：14:30 的 bar 在决策时不可见（<1430 排除），保留 09:45/10:00
        df = pd.DataFrame({
            'date': ['2026-01-05'] * 4,
            'time': [20260105094500000, 20260105100000000,
                     20260105143000000, 20260105150000000],
            'open': [1.40, 1.42, 1.45, 1.48],
            'high': [1.44, 1.46, 1.50, 1.52],
            'low': [1.38, 1.41, 1.44, 1.47],
            'close': [1.42, 1.45, 1.48, 1.51],
            'volume': [100, 150, 200, 300],
            'amount': [1000, 1500, 2000, 3000],
        })
        agg = et.aggregate_intraday_bars(df)
        self.assertEqual(len(agg), 1)
        row = agg.iloc[0]
        self.assertEqual(row['open'], 1.40)          # 首根 open
        self.assertEqual(row['close'], 1.45)         # 最后可见 bar（10:00）close；14:30 bar 剔除
        self.assertEqual(row['high'], 1.46)          # 截至 10:00 极值
        self.assertEqual(row['low'], 1.38)
        self.assertEqual(row['volume'], 100 + 150)   # 截至 10:00 累加
        self.assertEqual(row['amount'], 1000 + 1500)

    def test_multiple_dates(self):
        agg = et.aggregate_intraday_bars(self._make_min15())
        self.assertEqual(len(agg), 2)
        by_date = agg.set_index('date')
        # 第二日只剩 9:45 一根参与 → close = 1.52
        self.assertEqual(by_date.loc['2026-01-06', 'close'], 1.52)
        self.assertEqual(by_date.loc['2026-01-06', 'volume'], 120)

    def test_empty_returns_empty_frame(self):
        df = pd.DataFrame(columns=['date', 'time', 'open', 'high', 'low', 'close',
                                   'volume', 'amount'])
        agg = et.aggregate_intraday_bars(df)
        self.assertTrue(len(agg) == 0)

    def test_cutoff_before_first_bar(self):
        """cutoff 早于所有 bar → 空结果"""
        df = self._make_min15()
        agg = et.aggregate_intraday_bars(df, cutoff_hhmm=900)
        self.assertTrue(len(agg) == 0)


class TestBacktestExecPrice(unittest.TestCase):
    def _run(self, exec_price: str):
        feats = make_mini_features()
        return et.run_backtest_timing(
            feats, BASE_CONFIG,
            str(feats['date'].min().date()), str(feats['date'].max().date()),
            0.00055, exec_price,
        )

    def test_t1_close_uses_next_day_close(self):
        feats = make_mini_features()
        res = self._run('t1_close')
        trades = res['trades']
        buys = trades[trades['action'] == '买入']
        self.assertEqual(len(buys), 1)
        # 信号日 idx5(2024-01-09 附近) → 成交日 idx6，价格 = idx6 close
        self.assertEqual(buys.iloc[0]['entry_date'], feats['date'].iloc[6])
        self.assertAlmostEqual(buys.iloc[0]['entry_price'], feats['close'].iloc[6])
        # 卖出确认日 idx11 → 成交日 idx12
        self.assertEqual(buys.iloc[0]['exit_date'], feats['date'].iloc[12])

    def test_t_close_uses_same_day_close(self):
        feats = make_mini_features()
        res = self._run('t_close')
        trades = res['trades']
        buys = trades[trades['action'] == '买入']
        self.assertEqual(len(buys), 1)
        # 信号日 idx5 → 当日成交，价格 = idx5 close
        self.assertEqual(buys.iloc[0]['entry_date'], feats['date'].iloc[5])
        self.assertAlmostEqual(buys.iloc[0]['entry_price'], feats['close'].iloc[5])
        # 卖出确认日 idx11 → 当日成交
        self.assertEqual(buys.iloc[0]['exit_date'], feats['date'].iloc[11])

    def test_last_day_no_trade_t1_close(self):
        """t1_close：最后一天的信号不产生成交（无 T+1）"""
        res = self._run('t1_close')
        equity = res['equity_curve']
        # equity 应覆盖到最后一天（补记录），但最后一天无新交易
        self.assertEqual(equity['date'].iloc[-1], make_mini_features()['date'].iloc[-1])

    def test_invalid_exec_price_raises(self):
        with self.assertRaises(ValueError):
            et.run_backtest_timing(make_mini_features(), BASE_CONFIG, '2024-01-02',
                                   '2024-01-20', 0.00055, 'bad_price')

    def test_insufficient_data_raises(self):
        feats = make_mini_features(n_days=5)
        with self.assertRaises(ValueError):
            et.run_backtest_timing(feats, BASE_CONFIG, '2024-01-02', '2024-01-20',
                                   0.00055, 't_close')


class TestIsomorphismWithProduction(unittest.TestCase):
    def test_t1_close_matches_production_backtest(self):
        """迷你数据上：参数化回测(t1_close) 输出 == 生产 run_backtest 输出"""
        feats = make_mini_features()
        start = str(feats['date'].min().date())
        end = str(feats['date'].max().date())
        prod = backtest.run_backtest(feats, BASE_CONFIG, start, end, 0.00055)
        timing = et.run_backtest_timing(feats, BASE_CONFIG, start, end, 0.00055, 't1_close')

        pe, te = prod['equity_curve'], timing['equity_curve']
        self.assertEqual(len(pe), len(te))
        pd.testing.assert_frame_equal(
            pe[['date', 'equity']].reset_index(drop=True),
            te[['date', 'equity']].reset_index(drop=True),
        )
        pt, tt = prod['trades'], timing['trades']
        self.assertEqual(len(pt), len(tt))
        if len(pt) > 0:
            cols = ['action', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
                    'signal_date', 'signal_score', 'return']
            pd.testing.assert_frame_equal(
                pt[cols].reset_index(drop=True),
                tt[cols].reset_index(drop=True),
                check_dtype=False,
            )
        self.assertAlmostEqual(prod['final_value'], timing['final_value'], places=9)


class TestFactorCorr(unittest.TestCase):
    def test_factor_corr_identical_series(self):
        """同一序列对拍 corr=1.0"""
        df1 = pd.DataFrame({
            'date': pd.bdate_range('2024-01-02', periods=30),
            'momentum': np.linspace(-1, 1, 30),
            'total_score': np.linspace(-0.5, 0.5, 30),
        })
        df2 = df1.copy()
        corr = et.factor_corr(df1, df2)
        self.assertAlmostEqual(corr['momentum'], 1.0, places=9)
        self.assertAlmostEqual(corr['total_score'], 1.0, places=9)

    def test_factor_corr_subset_start(self):
        df1 = pd.DataFrame({
            'date': pd.bdate_range('2024-01-02', periods=30),
            'trend': np.linspace(-1, 1, 30),
        })
        df2 = df1.copy()
        df2.loc[df2.index >= 15, 'trend'] = -df2.loc[df2.index >= 15, 'trend']
        corr = et.factor_corr(df1, df2, subset_start='2024-01-16')
        # 子集内含正相关过渡段，仍应呈明显负相关
        self.assertLess(corr['trend'], -0.8)


if __name__ == '__main__':
    unittest.main()
