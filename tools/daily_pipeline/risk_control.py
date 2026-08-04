#!/usr/bin/env python3
"""
risk_control.py — 独立风控层（vnpy risk_manager 理念）

与策略状态机（signal_rules.decide）完全分离的可插拔风控层：
  - 单笔止损：以持仓批次买入均价 entry_price 为基准，收盘价跌破 entry*(1-pct) 触发
  - 权益回撤熔断：从峰值权益回撤达到/超过 dd_limit_pct 触发
  - 熔断后冷却期：触发清仓后 cooldown_days 内禁止开新仓

设计约束：
  - 纯函数、零依赖（仅 Python 标准库，不 import 任何项目内模块、不读文件）
  - 不持有状态：回测/实盘引擎以标量变量驱动这些纯函数（peak_equity /
    entry_price / cooldown_remaining 由调用方维护）
  - 所有边界均为「严格」语义并有单测覆盖（tests/test_risk_control.py）

风控参数统一从 config 读（config['risk_control']）；config 未含该段时
使用 DEFAULT_RISK_CONTROL（默认 enabled=false，保证关闭时回测行为与
无风控层完全一致）。在 config.json 中添加如下段落即可启用：

    "risk_control": {
      "enabled": true,
      "stop_loss_pct": 0.08,     # 单笔止损幅度，None=禁用
      "dd_limit_pct": 0.10,      # 权益回撤熔断限，None=禁用
      "cooldown_days": 5         # 触发后禁止开新仓的交易日数
    }

冷却期两种实现（在交易日序列下等价）：
  - in_cooldown(last_stop_date, today, cooldown_days)：按自然日差（通用纯函数，
    供单测与外部调用；传入交易日历日期序列时与交易日计数版等价）。
  - set_cooldown / tick_cooldown / check_cooldown：回测引擎内部用的交易日
    计数版（真实日历含周末时避免自然日漂移，语义精确）。
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional


def _as_date(v):
    """把 str('YYYY-MM-DD') / date / datetime / Timestamp 归一化为 date。"""
    if isinstance(v, str):
        return _dt.date.fromisoformat(v[:10])
    if hasattr(v, 'date'):
        return v.date()
    return v


# 需求 1 规定的默认风控参数（config.json 未含 risk_control 段时生效）
DEFAULT_RISK_CONTROL = {
    'enabled': False,
    'stop_loss_pct': 0.08,
    'dd_limit_pct': 0.10,
    'cooldown_days': 5,
}


# ── 单笔止损 ──────────────────────────────────────────

def check_stop_loss(entry_price: Optional[float], current_close: Optional[float],
                    stop_loss_pct: Optional[float]) -> bool:
    """单笔止损检查：以持仓批次买入均价 entry_price 为基准，
    收盘价严格跌破 entry*(1-pct) 触发。

    - stop_loss_pct 为 None（禁用）或 <= 0 → 永不触发
    - entry_price 为 None 或 <= 0（未持仓）→ 永不触发
    - 边界：收盘价 == 止损线时不触发（严格跌破）
    """
    if stop_loss_pct is None or stop_loss_pct <= 0:
        return False
    if entry_price is None or entry_price <= 0 or current_close is None:
        return False
    threshold = float(entry_price) * (1.0 - float(stop_loss_pct))
    return float(current_close) < threshold


def stop_loss_threshold(entry_price: Optional[float],
                        stop_loss_pct: Optional[float]) -> Optional[float]:
    """止损线价格 entry*(1-pct)；参数无效时返回 None（便于报告展示）。"""
    if stop_loss_pct is None or entry_price is None or entry_price <= 0:
        return None
    return float(entry_price) * (1.0 - float(stop_loss_pct))


# ── 权益回撤熔断 ──────────────────────────────────────

def check_drawdown_limit(peak_equity: Optional[float], current_equity: Optional[float],
                         dd_limit_pct: Optional[float]) -> bool:
    """权益回撤熔断：从峰值权益回撤达到或超过 dd_limit_pct 触发。

    - dd_limit_pct 为 None（禁用）或 <= 0 → 永不触发
    - peak_equity 为 None 或 <= 0 → 永不触发
    - 边界：回撤 == dd_limit_pct 时触发（达到限值即熔断）
    """
    if dd_limit_pct is None or dd_limit_pct <= 0:
        return False
    if peak_equity is None or peak_equity <= 0 or current_equity is None:
        return False
    drawdown = float(current_equity) / float(peak_equity) - 1.0
    return drawdown <= -float(dd_limit_pct)


def update_peak_equity(equity: Optional[float], peak_equity: Optional[float]) -> float:
    """滚动更新峰值权益（新高返回新值，否则保留旧值）。"""
    if equity is None:
        equity = 0.0
    if peak_equity is None:
        return float(equity)
    return float(max(equity, peak_equity))


# ── 熔断后冷却期（通用日期差版） ─────────────────────

def in_cooldown(last_stop_date, today, cooldown_days: Optional[int]) -> bool:
    """是否处于熔断后冷却期：today 距 last_stop_date 不足 cooldown_days 天。

    - last_stop_date 为 None → 从未触发，不在冷却期
    - cooldown_days 为 None 或 <= 0 → 无冷却期
    - 边界：today == last_stop_date（触发当日）即视为冷却中；
      相距恰好 cooldown_days 天 → 解除冷却。
    日期支持 date/datetime/Timestamp（内部按天数差计算）。
    """
    if last_stop_date is None or today is None:
        return False
    if cooldown_days is None or cooldown_days <= 0:
        return False
    delta = (_as_date(today) - _as_date(last_stop_date)).days
    return delta < int(cooldown_days)


# ── 熔断后冷却期（回测引擎交易日计数版） ──────────────

def set_cooldown(cooldown_days: Optional[int]) -> int:
    """触发清仓时进入冷却：返回剩余禁止开仓的交易日数（>=0 截断）。"""
    if cooldown_days is None:
        return 0
    return max(0, int(cooldown_days))


def tick_cooldown(cooldown_remaining: int) -> int:
    """每个交易日开始时递减冷却计数（下限 0）。"""
    return max(0, int(cooldown_remaining) - 1)


def check_cooldown(cooldown_remaining: int) -> bool:
    """是否处于冷却期（剩余交易日 > 0 → 禁止开新仓）。"""
    return int(cooldown_remaining) > 0


# ── 持仓均价跟踪 ──────────────────────────────────────

def update_entry_price(old_shares: float, old_entry_price: float,
                       new_shares: float, new_price: float) -> float:
    """加仓后按份额加权更新持仓均价。

    - 首仓（old_shares <= 0）→ 返回 new_price
    - 减仓（new_shares <= 0）→ 均价不变，返回 old_entry_price
    - 加权公式：(old_shares*old_price + new_shares*new_price) / (old_shares + new_shares)
    """
    old_shares = float(old_shares)
    new_shares = float(new_shares)
    if new_shares <= 0:
        return float(old_entry_price)
    total = old_shares + new_shares
    if total <= 1e-12:
        return float(new_price)
    if old_shares <= 1e-12:
        return float(new_price)
    return (old_shares * float(old_entry_price) + new_shares * float(new_price)) / total


# ── 配置读取 ──────────────────────────────────────────

def parse_risk_config(config: Optional[dict]) -> dict:
    """从 config 读取并规范化风控参数（默认 enabled=false）。

    返回 dict：{enabled, stop_loss_pct, dd_limit_pct, cooldown_days}。
    任何字段缺失/非法时取保守默认（DEFAULT_RISK_CONTROL 对应值）。
    """
    rc = config.get('risk_control') if isinstance(config, dict) else None
    if not isinstance(rc, dict):
        rc = {}
    defaults = DEFAULT_RISK_CONTROL

    def _num(key, default):
        v = rc.get(key, default)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return default

    return {
        'enabled': bool(rc.get('enabled', defaults['enabled'])),
        'stop_loss_pct': _num('stop_loss_pct', defaults['stop_loss_pct']),
        'dd_limit_pct': _num('dd_limit_pct', defaults['dd_limit_pct']),
        'cooldown_days': int(_num('cooldown_days', defaults['cooldown_days']) or 0),
    }
