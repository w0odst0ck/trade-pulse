#!/usr/bin/env python3
"""risk_control.py 单元测试（仅标准库 unittest，无第三方依赖）

覆盖验收要求：
  - 单笔止损：触发 / 不触发（含边界：等于止损线、pct=None 禁用）
  - 权益回撤熔断：达到限值触发 / 未达不触发（含 None 禁用）
  - 冷却期：set/tick/check 边界（0、正值递减、负值截断）
  - 持仓均价加权 / 峰值权益滚动 / 配置解析
"""

import sys
import unittest
from pathlib import Path

# 从项目根导入 tools 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.daily_pipeline import risk_control as rc
from tools.daily_pipeline import backtest as bt  # 引擎级集成测试


class TestStopLoss(unittest.TestCase):
    """单笔止损：收盘价严格跌破 entry*(1-pct) 触发"""

    def test_trigger_when_price_below_threshold(self):
        # entry=1.0, pct=0.08 → 止损线 0.92，收盘 0.91 跌破 → 触发
        self.assertTrue(rc.check_stop_loss(1.0, 0.91, 0.08))

    def test_not_trigger_at_threshold(self):
        # 收盘价 == 止损线 0.92 → 不触发（严格跌破）
        self.assertFalse(rc.check_stop_loss(1.0, 0.92, 0.08))

    def test_not_trigger_above_threshold(self):
        self.assertFalse(rc.check_stop_loss(1.0, 0.95, 0.08))
        self.assertFalse(rc.check_stop_loss(1.0, 1.10, 0.08))

    def test_disabled_when_pct_none(self):
        # stop_loss_pct=None → 永不触发（即使价格跌到 0）
        self.assertFalse(rc.check_stop_loss(1.0, 0.01, None))

    def test_not_trigger_without_position(self):
        # entry_price <= 0（未持仓）→ 永不触发
        self.assertFalse(rc.check_stop_loss(0.0, 0.5, 0.08))
        self.assertFalse(rc.check_stop_loss(-1.0, 0.5, 0.08))

    def test_stop_loss_threshold_value(self):
        self.assertAlmostEqual(rc.stop_loss_threshold(2.0, 0.10), 1.8)
        self.assertIsNone(rc.stop_loss_threshold(2.0, None))
        self.assertIsNone(rc.stop_loss_threshold(0.0, 0.1))


class TestDrawdownLimit(unittest.TestCase):
    """权益回撤熔断：从峰值回撤达到或超过 dd_limit_pct 触发（需求签名）"""

    def test_trigger_at_limit(self):
        # 峰值 1.0，权益 0.85 → 回撤 -15%，达到限值 0.15 → 触发
        self.assertTrue(rc.check_drawdown_limit(1.0, 0.85, 0.15))

    def test_trigger_beyond_limit(self):
        self.assertTrue(rc.check_drawdown_limit(1.0, 0.80, 0.15))
        self.assertTrue(rc.check_drawdown_limit(1.0, 0.50, 0.15))

    def test_not_trigger_below_limit(self):
        self.assertFalse(rc.check_drawdown_limit(1.0, 0.86, 0.15))
        self.assertFalse(rc.check_drawdown_limit(1.0, 1.05, 0.15))  # 创新高

    def test_disabled_when_pct_none(self):
        self.assertFalse(rc.check_drawdown_limit(1.0, 0.10, None))
        self.assertFalse(rc.check_drawdown_limit(1.0, 0.10, 0))

    def test_not_trigger_without_peak(self):
        self.assertFalse(rc.check_drawdown_limit(0.0, 0.8, 0.15))
        self.assertFalse(rc.check_drawdown_limit(-1.0, 0.8, 0.15))
        self.assertFalse(rc.check_drawdown_limit(None, 0.8, 0.15))


class TestInCooldown(unittest.TestCase):
    """冷却期日期版（需求签名）：自 last_stop_date 起 cooldown_days 内禁止开新仓"""

    def test_trigger_day_is_cooldown(self):
        # 触发当日（delta=0）即视为冷却中
        self.assertTrue(rc.in_cooldown('2026-07-30', '2026-07-30', 5))

    def test_within_cooldown(self):
        # delta=4 < 5 → 冷却中
        self.assertTrue(rc.in_cooldown('2026-07-30', '2026-08-03', 5))

    def test_cooldown_expires_at_exact_days(self):
        # delta == cooldown_days → 解除冷却
        self.assertFalse(rc.in_cooldown('2026-07-30', '2026-08-04', 5))
        self.assertFalse(rc.in_cooldown('2026-07-30', '2026-08-10', 5))

    def test_datetime_inputs(self):
        import datetime
        self.assertTrue(rc.in_cooldown(datetime.date(2026, 7, 30),
                                       datetime.date(2026, 8, 1), 5))

    def test_datetime_with_time_inputs(self):
        # 带时刻的 datetime：归一化后按日期差计算
        import datetime
        self.assertTrue(rc.in_cooldown(datetime.datetime(2026, 7, 30, 15, 0),
                                       datetime.datetime(2026, 8, 1, 10, 30), 5))
        self.assertFalse(rc.in_cooldown(datetime.datetime(2026, 7, 30, 15, 0),
                                        datetime.datetime(2026, 8, 4, 10, 30), 5))

    def test_disabled(self):
        self.assertFalse(rc.in_cooldown('2026-07-30', '2026-08-01', None))
        self.assertFalse(rc.in_cooldown('2026-07-30', '2026-08-01', 0))
        self.assertFalse(rc.in_cooldown(None, '2026-08-01', 5))
        self.assertFalse(rc.in_cooldown('2026-07-30', None, 5))


class TestPeakEquity(unittest.TestCase):
    """update_peak_equity：峰值滚动更新"""

    def test_update_to_new_high(self):
        self.assertEqual(rc.update_peak_equity(1.2, 1.0), 1.2)

    def test_keep_old_peak(self):
        self.assertEqual(rc.update_peak_equity(0.9, 1.0), 1.0)

    def test_equal_keeps_peak(self):
        self.assertEqual(rc.update_peak_equity(1.0, 1.0), 1.0)


class TestCooldownTradingDays(unittest.TestCase):
    """冷却期（回测引擎交易日计数版）：set_cooldown 设值 → 每交易日 tick 递减 → check 判禁开仓"""

    def test_set_and_tick_boundary(self):
        # cooldown_days=3：触发后 3 个交易日内禁止开仓，第 4 天恢复
        rem = rc.set_cooldown(3)
        self.assertEqual(rem, 3)
        self.assertTrue(rc.check_cooldown(rem))          # 触发日
        rem = rc.tick_cooldown(rem)                       # 次日
        self.assertEqual(rem, 2)
        self.assertTrue(rc.check_cooldown(rem))
        rem = rc.tick_cooldown(rem)
        self.assertEqual(rem, 1)
        self.assertTrue(rc.check_cooldown(rem))
        rem = rc.tick_cooldown(rem)
        self.assertEqual(rem, 0)
        self.assertFalse(rc.check_cooldown(rem))          # 冷却结束，可开仓

    def test_tick_floor_at_zero(self):
        self.assertEqual(rc.tick_cooldown(0), 0)
        self.assertEqual(rc.tick_cooldown(1), 0)
        self.assertFalse(rc.check_cooldown(rc.tick_cooldown(1)))

    def test_negative_clamped(self):
        self.assertEqual(rc.set_cooldown(-2), 0)
        self.assertEqual(rc.tick_cooldown(-1), 0)
        self.assertFalse(rc.check_cooldown(-1))

    def test_set_none_disables(self):
        self.assertEqual(rc.set_cooldown(None), 0)


class TestEntryPrice(unittest.TestCase):
    """持仓均价：加仓按份额加权，减仓均价不变"""

    def test_first_buy(self):
        # 首仓 100 份 @1.0 → 均价 1.0
        self.assertAlmostEqual(rc.update_entry_price(0.0, 0.0, 100.0, 1.0), 1.0)

    def test_add_weighted(self):
        # 100 份 @1.0 加 100 份 @1.2 → 均价 1.1
        self.assertAlmostEqual(rc.update_entry_price(100.0, 1.0, 100.0, 1.2), 1.1)

    def test_add_weighted_uneven(self):
        # 100 份 @1.0 加 50 份 @1.5 → 均价 1.1666...
        self.assertAlmostEqual(rc.update_entry_price(100.0, 1.0, 50.0, 1.5), 1.1666666666666667)

    def test_reduce_keeps_average(self):
        # 减仓（new_shares<=0）→ 均价不变
        self.assertAlmostEqual(rc.update_entry_price(100.0, 1.1, -30.0, 0.9), 1.1)
        self.assertAlmostEqual(rc.update_entry_price(100.0, 1.1, 0.0, 0.9), 1.1)

    def test_all_sold_reset(self):
        # 清仓后份额 0，下笔首仓直接取新价
        self.assertAlmostEqual(rc.update_entry_price(0.0, 0.0, 200.0, 1.05), 1.05)


class TestParseRiskConfig(unittest.TestCase):
    """配置解析：默认 enabled=false，缺省即关闭"""

    def test_missing_config_disabled(self):
        p = rc.parse_risk_config({})
        self.assertFalse(p['enabled'])
        self.assertEqual(p['cooldown_days'], 5)   # 需求默认冷却 5 天
        self.assertEqual(p['stop_loss_pct'], 0.08)  # 需求默认止损 8%
        self.assertEqual(p['dd_limit_pct'], 0.10)   # 需求默认熔断 10%

    def test_missing_config_is_disabled(self):
        p = rc.parse_risk_config(None)
        self.assertFalse(p['enabled'])

    def test_explicit_enabled(self):
        cfg = {'risk_control': {
            'enabled': True, 'stop_loss_pct': 0.10,
            'dd_limit_pct': 0.15, 'cooldown_days': 5,
        }}
        p = rc.parse_risk_config(cfg)
        self.assertTrue(p['enabled'])
        self.assertEqual(p['stop_loss_pct'], 0.10)
        self.assertEqual(p['dd_limit_pct'], 0.15)
        self.assertEqual(p['cooldown_days'], 5)

    def test_none_limits_disabled(self):
        cfg = {'risk_control': {
            'enabled': True, 'stop_loss_pct': None,
            'dd_limit_pct': None, 'cooldown_days': 3,
        }}
        p = rc.parse_risk_config(cfg)
        self.assertIsNone(p['stop_loss_pct'])
        self.assertIsNone(p['dd_limit_pct'])

    def test_non_dict_value_falls_back(self):
        p = rc.parse_risk_config({'risk_control': None})
        self.assertFalse(p['enabled'])

    def test_enabled_string_variants(self):
        # 字符串形式的 enabled 也按语义解析（'false' 不应被 bool() 误判为 True）
        for s, exp in [('true', True), ('True', True), ('yes', True), ('1', True),
                       ('on', True), ('false', False), ('no', False), ('0', False)]:
            p = rc.parse_risk_config({'risk_control': {'enabled': s}})
            self.assertEqual(p['enabled'], exp, f'enabled={s!r}')


class TestRiskEngineIntegration(unittest.TestCase):
    """run_backtest 引擎级集成：止损触发强制清仓、冷却期拦截、reason 标注

    特征序列：价格 1.00 涨至 1.08 后一路跌至 0.80，total_score 恒 0.5（状态机
    始终『持仓』、不会主动卖出）→ 唯一离场路径是风控止损。
    止损线 = entry(1.02) × (1-0.05) = 0.969，收盘 0.94 触发。
    """

    @staticmethod
    def _make_features(closes):
        import pandas as pd
        dates = pd.bdate_range('2024-01-02', periods=len(closes))
        return pd.DataFrame({'date': dates, 'close': closes, 'total_score': 0.5})

    @staticmethod
    def _config(enabled: bool) -> dict:
        return {
            'thresholds': {'buy': 0.1, 'sell': -0.07, 'confirm_days': 2,
                           'weekly_filter_percentile': 0.2},
            'adaptive_thresholds': {'enabled': False},
            'risk_control': {'enabled': enabled, 'stop_loss_pct': 0.05,
                             'dd_limit_pct': None, 'cooldown_days': 3},
        }

    def _run(self, enabled: bool):
        import contextlib
        import io
        closes = [1.00, 1.02, 1.04, 1.06, 1.08, 1.06, 1.02, 0.98, 0.94, 0.90,
                  0.86, 0.82, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94,
                  0.96, 0.98, 1.00, 1.02, 1.04]
        df = self._make_features(closes)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = bt.run_backtest(df, self._config(enabled), '2024-01-02',
                                  '2024-12-31', 0.00055)
        return res

    def test_stop_loss_forced_liquidation(self):
        res = self._run(enabled=True)
        # 触发一次止损，reason 正确，清仓价 0.94
        self.assertEqual(len(res['risk_events']), 1)
        ev = res['risk_events'][0]
        self.assertEqual(ev['reason'], '止损')
        self.assertEqual(float(ev['close']), 0.94)
        # trades 中强制清仓记录带 reason，且 LIFO 闭合了持仓批次（return 为负）
        trades = res['trades']
        liq = trades[trades['reason'] == '止损']
        self.assertEqual(len(liq), 1)
        closed = trades[(trades['action'].isin(['买入', '加仓'])) & trades['exit_date'].notna()]
        self.assertEqual(len(closed), 1)
        self.assertLess(closed.iloc[0]['return'], 0)   # 0.94 < 1.02 亏损离场
        # 清仓当日（T+1 收盘后）仓位归零、持有市值归零
        eq = res['equity_curve']
        liq_day = ev['date']
        liq_row = eq[eq['date'] == liq_day].iloc[0]
        self.assertAlmostEqual(liq_row['position'], 0.0, places=3)
        self.assertAlmostEqual(liq_row['hold_value'], 0.0, places=3)

    def test_cooldown_blocks_reentry(self):
        res = self._run(enabled=True)
        ev = res['risk_events'][0]
        eq = res['equity_curve']
        # 清仓日（含）之后 cooldown_days=3 个交易日内仓位保持 0（禁止开新仓）
        idx = eq.index[eq['date'] == ev['date']][0]
        for k in range(1, 4):
            self.assertAlmostEqual(eq['position'].iloc[idx + k], 0.0, places=3,
                                   msg=f"冷却期第 {k} 个交易日不应开仓")
        # 冷却结束后（第 4 个交易日）价格已回升，应能重新建仓
        self.assertGreater(eq['position'].iloc[idx + 4], 0.0)

    def test_disabled_no_risk_events(self):
        res = self._run(enabled=False)
        self.assertEqual(res['risk_events'], [])
        self.assertNotIn('reason', res['trades'].columns)


if __name__ == '__main__':
    unittest.main()
