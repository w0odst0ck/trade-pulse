#!/usr/bin/env python3
"""daily_panel 实时模式测试（方案 B）：preview/confirm/fallback 状态机安全"""

import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "daily_pipeline"))

import daily_panel as dp
import realtime_quote as rq
from signal_rules import STATE_PATH


def _make_bar(source='tencent'):
    return {
        'date': date(2026, 8, 12), 'open': 1.84, 'high': 1.851, 'low': 1.799,
        'close': 1.835, 'volume': 34364399.0, 'amount': 6255060479.0,
        'prev_close': 1.84, 'source': source,
    }


class _Args:
    """模拟 argparse.Namespace"""
    def __init__(self, realtime=False, realtime_confirm=False, push=False, json=False):
        self.realtime = realtime
        self.realtime_confirm = realtime_confirm
        self.push = push
        self.json = json


class TestRealtimePanel(unittest.TestCase):
    """daily_panel --realtime / --realtime-confirm 行为（状态安全属性隔离测试）

    隔离策略：mock compute_realtime_features 和 decide，避免依赖真实数据文件
    （干净环境无 daily.csv 时 load_data 会抛异常，导致测试无法验证状态安全属性）。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / 'state.json'
        self.state_path.write_text(json.dumps({
            'state': '持仓', 'waiting_days': 1, 'last_decision_date': '2026-08-11',
        }), encoding='utf-8')
        # daily_panel 内部 import signal_rules.decide → 实际写盘路径在 signal_rules.STATE_PATH
        self.state_patcher = mock.patch('signal_rules.STATE_PATH', self.state_path)
        self.state_patcher.start()
        # _persist_realtime_signal 写 data/{symbol}/realtime_signal.json → 隔离到临时目录
        self.proj_patcher = mock.patch('daily_panel.PROJECT_ROOT', Path(self.tmp.name))
        self.proj_patcher.start()

        # 自包含 feature df（模拟 compute_realtime_features 返回）
        import pandas as pd
        self.feat_df = pd.DataFrame([{
            'date': pd.Timestamp('2026-08-12'), 'close': 1.835, 'volume': 3.4e7,
            'momentum': 0.35, 'trend': -0.9, 'volume_price': 0.2, 'rsrs': -0.5,
            'relative_strength': 0.03, 'weekly_modifier': 0.12, 'ma60_slope': -0.01,
            'total_score': -0.26,
        }])
        self.patchers = [
            mock.patch('compute_features.compute_realtime_features', return_value=self.feat_df),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self.proj_patcher.stop()
        self.state_patcher.stop()
        self.tmp.cleanup()

    @mock.patch('realtime_quote.fetch_realtime_bar', return_value=_make_bar())
    @mock.patch('trading_calendar.is_trading_day', return_value=True)
    def test_preview_does_not_write_state(self, mock_day, mock_fetch):
        """preview：不写 state（实盘安全关键）"""
        args = _Args(realtime=True, json=True)
        with mock.patch.object(rq, 'is_market_open', return_value=True):
            result = dp.run_realtime(args, dp.load_config(), '588000')
        self.assertEqual(result['signal_mode'], 'realtime')
        self.assertTrue(result['is_preview'])
        # state 文件未被修改（preview 不写盘）
        saved = json.loads(self.state_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['last_decision_date'], '2026-08-11')  # 未被改写
        self.assertNotIn('signal_mode', saved)  # 原 state 无此字段（未写入）

    @mock.patch('realtime_quote.fetch_realtime_bar', return_value=_make_bar())
    @mock.patch('trading_calendar.is_trading_day', return_value=True)
    def test_confirm_writes_state(self, mock_day, mock_fetch):
        """confirm：写 state + signal_mode=realtime"""
        args = _Args(realtime_confirm=True, json=True)
        with mock.patch.object(rq, 'is_market_open', return_value=True):
            result = dp.run_realtime(args, dp.load_config(), '588000')
        self.assertEqual(result['signal_mode'], 'realtime')
        saved = json.loads(self.state_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['signal_mode'], 'realtime')
        self.assertIsNotNone(saved.get('signal_data_date'))

    @mock.patch('realtime_quote.fetch_realtime_bar', return_value=None)  # 双源失败
    @mock.patch('trading_calendar.is_trading_day', return_value=True)
    def test_fallback_does_not_write_state(self, mock_day, mock_fetch):
        """实时源失败 → fallback 收盘口径且不写 state（最危险的路径）"""
        args = _Args(realtime_confirm=True, json=True)  # 即使 confirm 也不写
        with mock.patch.object(rq, 'is_market_open', return_value=True):
            result = dp.run_realtime(args, dp.load_config(), '588000')
        self.assertEqual(result['signal_mode'], 'close')
        self.assertIn('fallback', result)
        saved = json.loads(self.state_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['last_decision_date'], '2026-08-11')  # 未被改写
        self.assertNotIn('signal_mode', saved)

    @mock.patch('realtime_quote.fetch_realtime_bar', return_value=_make_bar())
    def test_non_trading_day_skips(self, mock_fetch):
        """非交易日 → 跳过（不拉实时、不写 state）"""
        args = _Args(realtime=True, json=True)
        with mock.patch('trading_calendar.is_trading_day', return_value=False):
            result = dp.run_realtime(args, dp.load_config(), '588000')
        self.assertEqual(result['decision'], '非交易日')
        mock_fetch.assert_not_called()

    @mock.patch('realtime_quote.fetch_realtime_bar', return_value=_make_bar())
    @mock.patch('trading_calendar.is_trading_day', return_value=True)
    def test_realtime_signal_data_date(self, mock_day, mock_fetch):
        """实时信号 signal_data_date = 当日盘中 bar 日期"""
        args = _Args(realtime_confirm=True, json=True)
        with mock.patch.object(rq, 'is_market_open', return_value=True):
            result = dp.run_realtime(args, dp.load_config(), '588000')
        self.assertEqual(result['signal_data_date'], '2026-08-12')

    @mock.patch('realtime_quote.fetch_realtime_bar', return_value=_make_bar())
    @mock.patch('trading_calendar.is_trading_day', return_value=True)
    def test_fallback_exception_routes_to_fallback(self, mock_day, mock_fetch):
        """实时特征计算抛异常 → 不崩溃，走兜底且不写 state（ocr high 修复验证）"""
        with mock.patch('compute_features.compute_realtime_features',
                        side_effect=RuntimeError('boom')), \
             mock.patch.object(rq, 'is_market_open', return_value=True):
            args = _Args(realtime_confirm=True, json=True)
            result = dp.run_realtime(args, dp.load_config(), '588000')
        self.assertEqual(result['signal_mode'], 'close')
        self.assertIn('fallback', result)
        self.assertIn('实时特征计算失败', result['fallback'])
        saved = json.loads(self.state_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['last_decision_date'], '2026-08-11')  # 未被改写


if __name__ == '__main__':
    unittest.main()
