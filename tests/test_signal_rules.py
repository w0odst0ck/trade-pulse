#!/usr/bin/env python3
"""signal_rules.py 单元测试（仅标准库 unittest，无第三方依赖）"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# 从项目根导入 tools 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.daily_pipeline import signal_rules as sr

# 测试用配置：关闭自适应阈值，thresholds 固定 buy=0.1 / sell=-0.07 / confirm_days=2
BASE_CONFIG = {
    'thresholds': {
        'buy': 0.1,
        'sell': -0.07,
        'confirm_days': 2,
        'weekly_filter_percentile': 0.2,
    },
    'adaptive_thresholds': {'enabled': False},
}

# 开启自适应阈值的配置，供 get_adaptive_thresholds 测试
ADAPTIVE_CONFIG = {
    'thresholds': BASE_CONFIG['thresholds'],
    'adaptive_thresholds': {
        'enabled': True,
        'uptrend_buy': 0.1,
        'uptrend_sell': -0.05,
        'downtrend_buy': 0.15,
        'downtrend_sell': -0.1,
        'sideways_buy': 0.12,
        'sideways_sell': -0.08,
        'ma60_slope_uptrend': 0.005,
        'ma60_slope_downtrend': -0.005,
    },
}

EMPTY_STATE = {'state': '空仓', 'waiting_days': 0, 'last_decision_date': None}


class TestFactorVisualization(unittest.TestCase):
    """factor_emoji / factor_arrow 边界值"""

    def test_factor_emoji_boundaries(self):
        self.assertEqual(sr.factor_emoji(0.6), '🟢 强')
        self.assertEqual(sr.factor_emoji(0.5), '🟢')          # > 0.2 档上限边界
        self.assertEqual(sr.factor_emoji(0.21), '🟢')
        self.assertEqual(sr.factor_emoji(0.2), '⚪ 中性')     # <= 0.2
        self.assertEqual(sr.factor_emoji(0.0), '⚪ 中性')
        self.assertEqual(sr.factor_emoji(-0.21), '🔴')       # <= -0.2
        self.assertEqual(sr.factor_emoji(-0.2), '🔴')        # -0.2 > -0.2 为 False
        self.assertEqual(sr.factor_emoji(-0.49), '🔴')       # -0.49 > -0.5
        self.assertEqual(sr.factor_emoji(-0.5), '🔴 弱')     # -0.5 > -0.5 为 False
        self.assertEqual(sr.factor_emoji(-0.51), '🔴 弱')
        self.assertEqual(sr.factor_emoji(-1.0), '🔴 弱')

    def test_factor_arrow_boundaries(self):
        self.assertEqual(sr.factor_arrow(0.5), '↑')
        self.assertEqual(sr.factor_arrow(0.3), '↗')           # > 0.3 不含边界
        self.assertEqual(sr.factor_arrow(0.31), '↑')
        self.assertEqual(sr.factor_arrow(0.1), '↗')
        self.assertEqual(sr.factor_arrow(0.0), '→')           # > 0 不含边界
        self.assertEqual(sr.factor_arrow(-0.1), '→')
        self.assertEqual(sr.factor_arrow(-0.31), '↘')
        self.assertEqual(sr.factor_arrow(-0.3), '↘')          # -0.3 > -0.3 为 False
        self.assertEqual(sr.factor_arrow(-0.5), '↘')          # -0.5 > -0.6
        self.assertEqual(sr.factor_arrow(-0.6), '↓')          # -0.6 > -0.6 为 False
        self.assertEqual(sr.factor_arrow(-0.61), '↓')
        self.assertEqual(sr.factor_arrow(-1.0), '↓')


class TestAdaptiveThresholds(unittest.TestCase):
    """get_adaptive_thresholds：禁用时用 config 阈值，启用时按 MA60 斜率自适应"""

    def test_disabled_returns_config_thresholds(self):
        cfg = dict(BASE_CONFIG)
        cfg['adaptive_thresholds'] = {'enabled': False}
        result = sr.get_adaptive_thresholds(cfg, {'ma60_slope': 0.01})
        self.assertIs(result, cfg['thresholds'])  # 原样返回，未复制
        self.assertEqual(result['buy'], 0.1)
        self.assertEqual(result['sell'], -0.07)

    def test_adaptive_by_ma60_slope(self):
        # 上升趋势
        up = sr.get_adaptive_thresholds(ADAPTIVE_CONFIG, {'ma60_slope': 0.01})
        self.assertEqual(up['buy'], 0.1)
        self.assertEqual(up['sell'], -0.05)
        # 下降趋势
        down = sr.get_adaptive_thresholds(ADAPTIVE_CONFIG, {'ma60_slope': -0.01})
        self.assertEqual(down['buy'], 0.15)
        self.assertEqual(down['sell'], -0.1)
        # 横盘（0 是合法值）
        flat = sr.get_adaptive_thresholds(ADAPTIVE_CONFIG, {'ma60_slope': 0.0})
        self.assertEqual(flat['buy'], 0.12)
        self.assertEqual(flat['sell'], -0.08)
        # 缺失斜率 → 默认 0 → 横盘
        missing = sr.get_adaptive_thresholds(ADAPTIVE_CONFIG, {})
        self.assertEqual(missing['buy'], 0.12)
        # NaN 回退到基础阈值
        nan = sr.get_adaptive_thresholds(ADAPTIVE_CONFIG, {'ma60_slope': float('nan')})
        self.assertEqual(nan['buy'], 0.1)
        self.assertEqual(nan['sell'], -0.07)
        # 各分支都保留 confirm_days
        self.assertEqual(up['confirm_days'], 2)
        self.assertEqual(down['confirm_days'], 2)


class TestDecideStateMachine(unittest.TestCase):
    """decide() 三态状态机：全部通过 config_override/state_override，persist=False 不落盘"""

    def _decide(self, total_score, state=EMPTY_STATE, date='2026-01-05', **features):
        feat = {'total_score': total_score, 'date': date, 'weekly_modifier': 0.0}
        feat.update(features)
        return sr.decide(feat, state_override=dict(state),
                         config_override=BASE_CONFIG, persist=False)

    def test_kongchang_to_chicang(self):
        r = self._decide(0.5)  # adjusted 0.5 > buy 0.1
        self.assertEqual(r['decision'], '持仓')
        self.assertEqual(r['action'], '买入（试仓）')
        self.assertEqual(r['_new_state']['state'], '持仓')
        self.assertEqual(r['position'], '50%')

    def test_kongchang_wait(self):
        # 模糊区：sell < score < buy → 等待
        r = self._decide(0.0)
        self.assertEqual(r['decision'], '空仓')
        self.assertEqual(r['action'], '等待')
        self.assertIn('未达买入阈值', r['explanation'])
        # 偏空：score < sell → 等待
        r2 = self._decide(-0.5)
        self.assertEqual(r2['decision'], '空仓')
        self.assertEqual(r2['action'], '等待')
        self.assertIn('市场偏空', r2['explanation'])

    def test_chicang_to_watch(self):
        # 持仓 + 分数回落到模糊区 → 观望 / 减仓
        state = {'state': '持仓', 'waiting_days': 0, 'last_decision_date': None}
        r = self._decide(0.0, state)
        self.assertEqual(r['decision'], '观望')
        self.assertEqual(r['action'], '减仓观望')
        self.assertEqual(r['_new_state']['state'], '观望')
        self.assertEqual(r['_new_state']['waiting_days'], 0)

    def test_chicang_sell_after_confirm_days(self):
        # 持仓 + 连续偏空：第 1 天只计数（confirm_days=2），第 2 天才卖出
        state = {'state': '持仓', 'waiting_days': 0, 'last_decision_date': None}
        r1 = self._decide(-0.2, state)  # -0.2 < sell -0.07
        self.assertEqual(r1['decision'], '持仓')
        self.assertEqual(r1['action'], '持有观察')
        self.assertEqual(r1['_new_state']['waiting_days'], 1)
        self.assertIn('1/2', r1['explanation'])

        r2 = self._decide(-0.2, r1['_new_state'], date='2026-01-06')  # 第 2 天（新交易日）
        self.assertEqual(r2['decision'], '空仓')
        self.assertEqual(r2['action'], '卖出')
        self.assertEqual(r2['_new_state']['waiting_days'], 0)

    def test_watch_transitions(self):
        state = {'state': '观望', 'waiting_days': 0, 'last_decision_date': None}
        # 观望 → 持仓
        r_up = self._decide(0.5, state)
        self.assertEqual(r_up['decision'], '持仓')
        self.assertEqual(r_up['action'], '买入加仓')
        # 观望 → 空仓
        r_down = self._decide(-0.5, state)
        self.assertEqual(r_down['decision'], '空仓')
        self.assertEqual(r_down['action'], '卖出')
        # 观望 + 模糊区 → 继续观望
        r_mid = self._decide(0.0, state)
        self.assertEqual(r_mid['decision'], '观望')
        self.assertEqual(r_mid['action'], '继续观望')

    def test_persist_false_does_not_write_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_state = Path(tmp) / 'state.json'
            with mock.patch.object(sr, 'STATE_PATH', fake_state):
                r = sr.decide(
                    {'total_score': 0.5, 'date': '2026-01-05'},
                    state_override=dict(EMPTY_STATE),
                    config_override=BASE_CONFIG,
                    persist=False,
                )
                self.assertFalse(fake_state.exists())
                self.assertEqual(r['_new_state']['state'], '持仓')

                # persist=True 时才落盘
                sr.decide(
                    {'total_score': 0.5, 'date': '2026-01-06'},
                    state_override=dict(EMPTY_STATE),
                    config_override=BASE_CONFIG,
                    persist=True,
                )
                self.assertTrue(fake_state.exists())
                saved = json.loads(fake_state.read_text(encoding='utf-8'))
                self.assertEqual(saved['state'], '持仓')
                self.assertEqual(saved['last_decision_date'], '2026-01-06')


class TestConfigStateIO(unittest.TestCase):
    """load_config / save_state / load_state 往返（路径重定向到临时文件）"""

    def test_load_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / 'config.json'
            cfg_path.write_text(json.dumps(BASE_CONFIG, ensure_ascii=False), encoding='utf-8')
            with mock.patch.object(sr, 'CONFIG_PATH', cfg_path):
                loaded = sr.load_config()
            self.assertEqual(loaded['thresholds']['buy'], 0.1)
            self.assertEqual(loaded['adaptive_thresholds']['enabled'], False)

    def test_state_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / 'nested' / 'state.json'  # 验证 save_state 自动建目录
            with mock.patch.object(sr, 'STATE_PATH', state_path):
                saved = {'state': '持仓', 'waiting_days': 1, 'last_decision_date': '2026-01-05'}
                sr.save_state(saved)
                self.assertTrue(state_path.exists())
                loaded = sr.load_state()
            # load_state 会用默认值补齐新字段（signal_mode/signal_data_date），
            # 保证旧 state.json 无缝升级；已保存字段原样保留
            self.assertEqual(loaded['state'], '持仓')
            self.assertEqual(loaded['waiting_days'], 1)
            self.assertEqual(loaded['last_decision_date'], '2026-01-05')
            self.assertEqual(loaded['signal_mode'], 'close')
            self.assertIsNone(loaded['signal_data_date'])

    def test_load_state_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sr, 'STATE_PATH', Path(tmp) / 'nope.json'):
                loaded = sr.load_state()
            self.assertEqual(loaded['state'], '空仓')
            self.assertEqual(loaded['waiting_days'], 0)
            self.assertIsNone(loaded['last_decision_date'])
            self.assertEqual(loaded['signal_mode'], 'close')
            self.assertIsNone(loaded['signal_data_date'])

    def test_decide_signal_mode_fields(self):
        """decide 输出应带 signal_mode/signal_data_date（方案 B 字段）"""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / 'state.json'
            with mock.patch.object(sr, 'STATE_PATH', state_path):
                feat = {
                    'date': '2026-01-05',
                    'total_score': 0.3, 'weekly_modifier': 0.0,
                    'momentum': 0.5, 'trend': 0.4, 'volume_price': 0.2,
                    'rsrs': 0.1, 'relative_strength': 0.0,
                }
                result = sr.decide(feat, None, persist=False, signal_mode='realtime',
                                   signal_data_date='2026-01-05')
                self.assertEqual(result['signal_mode'], 'realtime')
                self.assertEqual(result['signal_data_date'], '2026-01-05')
                self.assertEqual(result['_new_state']['signal_mode'], 'realtime')
                self.assertEqual(result['_new_state']['signal_data_date'], '2026-01-05')
                self.assertEqual(result['_new_state']['state'], '持仓')

    def test_decide_persist_writes_signal_mode(self):
        """confirm 模式 persist=True 时应把 signal_mode 写入 state.json"""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / 'state.json'
            with mock.patch.object(sr, 'STATE_PATH', state_path):
                feat = {
                    'date': '2026-01-05',
                    'total_score': -0.2, 'weekly_modifier': 0.0,
                    'momentum': -0.1, 'trend': -0.2, 'volume_price': 0.0,
                    'rsrs': -0.3, 'relative_strength': 0.0,
                }
                sr.decide(feat, None, persist=True, signal_mode='realtime',
                          signal_data_date='2026-01-05')
                loaded = sr.load_state()
                self.assertEqual(loaded['signal_mode'], 'realtime')
                self.assertEqual(loaded['signal_data_date'], '2026-01-05')


if __name__ == '__main__':
    unittest.main()
