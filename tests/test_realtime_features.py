#!/usr/bin/env python3
"""compute_features 实时路径 + daily_panel 实时模式单元测试（方案 B）"""

import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "daily_pipeline"))

import pandas as pd

import compute_features as cf


class TestAppendRealtimeBar(unittest.TestCase):
    """append_realtime_bar：历史 + 实时 bar 拼接"""

    def _make_df(self, dates=("2026-08-06", "2026-08-07")):
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                'date': pd.Timestamp(d), 'open': 1.7 + i * 0.1, 'close': 1.75 + i * 0.1,
                'high': 1.8 + i * 0.1, 'low': 1.68 + i * 0.1,
                'volume': 4e7 + i, 'amount': 7e7 + i,
                'amplitude': 0, 'change_pct': 0, 'change': 0, 'turnover': 0, 'symbol': '588000',
            })
        return pd.DataFrame(rows)

    def test_append_new_date(self):
        df = self._make_df()
        bar = {'date': date(2026, 8, 10), 'open': 1.84, 'high': 1.851, 'low': 1.799,
               'close': 1.835, 'volume': 34364399.0, 'amount': 6255060479.0}
        out = cf.append_realtime_bar(df, bar)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.iloc[-1]['close'], 1.835)
        self.assertEqual(out.iloc[-1]['date'], pd.Timestamp('2026-08-10'))

    def test_replace_same_date(self):
        """同日重复调用（14:25 preview 后 14:50 confirm）→ 替换而非追加"""
        df = self._make_df(dates=("2026-08-06", "2026-08-10"))
        bar = {'date': date(2026, 8, 10), 'open': 1.84, 'high': 1.86, 'low': 1.80,
               'close': 1.85, 'volume': 4e7, 'amount': 7e7}
        out = cf.append_realtime_bar(df, bar)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[-1]['close'], 1.85)  # 实时刷新语义

    def test_missing_optional_cols_filled(self):
        """历史 df 缺 amount 等列时实时 bar 也能拼（fill 0）"""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2026-08-07')], 'open': [1.7], 'close': [1.75],
            'high': [1.8], 'low': [1.68], 'volume': [4e7], 'symbol': ['588000'],
        })
        bar = {'date': date(2026, 8, 10), 'open': 1.84, 'high': 1.851, 'low': 1.799,
               'close': 1.835, 'volume': 34364399.0, 'amount': 6255060479.0}
        out = cf.append_realtime_bar(df, bar)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[-1]['amount'], 6255060479.0)

    def test_missing_price_fields_become_nan(self):
        """缺价格/量字段 → NaN 而非 0（0 价会被因子函数当真实数据，静默产生错误信号）"""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2026-08-07')], 'open': [1.7], 'close': [1.75],
            'high': [1.8], 'low': [1.68], 'volume': [4e7], 'symbol': ['588000'],
        })
        bar = {'date': date(2026, 8, 10)}  # 缺 OHLC/volume（部分数据场景）
        out = cf.append_realtime_bar(df, bar)
        last = out.iloc[-1]
        self.assertTrue(pd.isna(last['open']))
        self.assertTrue(pd.isna(last['close']))
        self.assertTrue(pd.isna(last['volume']))
        # 0 价 bar 被 NaN 取代：避免驱动错误买卖信号
        self.assertNotEqual(last['close'], 0.0)

    def test_missing_symbol_col_handled(self):
        """历史 df 缺 symbol 列 → 补空字符串，不产生 NaN symbol（不污染替换路径）"""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2026-08-07')], 'open': [1.7], 'close': [1.75],
            'high': [1.8], 'low': [1.68], 'volume': [4e7],
        })  # 无 symbol 列
        bar = {'date': date(2026, 8, 10), 'open': 1.84, 'high': 1.851, 'low': 1.799,
               'close': 1.835, 'volume': 34364399.0, 'amount': 6255060479.0}
        out = cf.append_realtime_bar(df, bar)
        self.assertIn('symbol', out.columns)
        self.assertEqual(out.iloc[-1]['symbol'], '')  # 空字符串而非 NaN
        self.assertEqual(out.iloc[-1]['close'], 1.835)


class TestComputeRealtimeFeatures(unittest.TestCase):
    """compute_realtime_features：历史 + 实时 → 因子（不落盘）"""

    def test_returns_full_frame_and_does_not_persist(self):
        config = {
            'data_dir': 'data', 'symbol': '588000', 'benchmark': '000688',
            'weights': {'momentum': 0.25, 'trend': 0.25, 'volume_price': 0.25, 'rsrs': 0.25},
            'momentum_window': 5, 'trend_window': 20, 'atr_window': 14,
            'rsrs_window': 18, 'rel_strength_window': 20,
            'weekly_modifier': {'min_modifier': -0.3, 'max_modifier': 0.3},
            'weights_by_regime': {'enabled': False},
        }
        # 自包含 fixture：合成 ~130 天日线（覆盖 momentum/trend/rsrs/rel_strength 窗口），
        # mock load_data 返回合成 df，不依赖真实数据文件（CI/干净环境可复现）
        rng = pd.date_range('2026-01-01', periods=130, freq='B')
        price = pd.Series(range(len(rng)), index=rng, dtype=float) / 100 + 1.0
        hist_sym = pd.DataFrame({
            'date': rng, 'open': price, 'close': price + 0.01, 'high': price + 0.03,
            'low': price - 0.02, 'volume': 4e7, 'amount': 7e7,
            'amplitude': 0, 'change_pct': 0, 'change': 0, 'turnover': 0, 'symbol': '588000',
        })
        hist_bench = pd.DataFrame({
            'date': rng, 'open': price, 'close': price, 'high': price + 0.02,
            'low': price - 0.02, 'volume': 4e7, 'amount': 7e7,
            'amplitude': 0, 'change_pct': 0, 'change': 0, 'turnover': 0, 'symbol': '000688',
        })
        bar = {'date': date(2026, 7, 15), 'open': 1.84, 'high': 1.851, 'low': 1.799,
               'close': 1.835, 'volume': 34364399.0, 'amount': 6255060479.0}
        with mock.patch.object(cf, 'load_data', side_effect=[hist_sym, hist_bench]), \
             mock.patch.object(cf, 'save_features_cache') as mock_save:
            features = cf.compute_realtime_features('588000', '000688', bar, None, config)
        self.assertEqual(len(features), 131)  # 130 历史 + 1 实时行
        self.assertIn('total_score', features.columns)
        # 不落盘：save_features_cache 不被调用
        mock_save.assert_not_called()
        # 最后一行为实时 bar 日期
        self.assertEqual(str(features.iloc[-1]['date'])[:10], '2026-07-15')


if __name__ == '__main__':
    unittest.main()
