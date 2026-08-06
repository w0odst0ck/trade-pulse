#!/usr/bin/env python3
"""
execution_timing.py — 决策点 F：成交时点 + 特征口径双对比（T+0 vs T+1 完整实验）

背景：回测口径 = T 日收盘特征 → T+1 收盘成交；实盘口径 = T 日 14:25 盘中特征 → 当天尾盘成交。
本脚本量化两者差距 = 特征口径差 + 成交时点差，指导实盘执行（信号出来当天尾盘买，还是等 T+1）。

4 组实验：
  ① 收盘特征 + T+1 收盘成交   ← 现状回测口径（基线）
  ② 收盘特征 + T 收盘成交     ← 纯时点差（决策点 F 原文）
  ③ 盘中特征(截至14:30) + T 收盘成交  ← 实盘最接近路径
  ④ 盘中特征(截至14:30) + T+1 收盘成交 ← 隔离特征影响

分析范围：
  - 全段（2023-01 ~ 数据末日）：仅 ① vs ②（时点差）
  - 子区间（2026-01-05 ~ min15 末日）：①②③④ 全比

实现原则：
  - 不修改任何生产代码（backtest.py / compute_features.py / signal_rules.py 等零改动）
  - 因子计算直接复用 compute_features.py 的因子函数（import 方式）
  - 回测状态机为 backtest.py run_backtest 的逐字同构复制，仅成交价来源参数化
    （t_close：特征日 T 收盘成交；t1_close：T+1 收盘成交，现状）
  - 运行期同构自检：exec_price='t1_close' 的输出必须与生产 run_backtest 完全一致

用法：
  python3 execution_timing.py              # 全流程（打印对比表 + 存 CSV）
  python3 execution_timing.py --cost 0.00055
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

# 把 SCRIPT_DIR 加入路径，复用同目录生产模块（只读 import，不改动）
sys.path.insert(0, str(SCRIPT_DIR))

from backtest import (  # noqa: E402  （生产回测：只读复用 run_backtest / compute_metrics / load_config）
    compute_metrics,
    load_config,
    run_backtest,
)
from risk_control import (  # noqa: E402 （生产风控层：复制状态机需要，enabled=false 时旁路）
    check_cooldown,
    check_drawdown_limit,
    check_stop_loss,
    parse_risk_config,
    set_cooldown,
    tick_cooldown,
    update_entry_price,
    update_peak_equity,
)
import compute_features as cf  # noqa: E402 （生产因子函数：只读复用）
from signal_rules import decide  # noqa: E402 （生产状态机决策）

# 盘中特征截止时点：14:25 信号决策时，最后已知的完整 15 分钟 bar 是 14:15 结束那根（time HHMM=1415）。
# 14:30 的 bar 在 14:30 才完成，14:25 时不可见——用 hhmm < 1430 排除，避免 5 分钟未来信息。
INTRADAY_CUTOFF_HHMM = 1430

# 对比组定义（特征口径, 成交时点, 含义）
GROUPS = {
    '①': {'feature': 'close', 'exec': 't1_close', 'label': '① 收盘特征 + T+1收盘(基线)'},
    '②': {'feature': 'close', 'exec': 't_close', 'label': '② 收盘特征 + T收盘(时点差)'},
    '③': {'feature': 'intraday', 'exec': 't_close', 'label': '③ 盘中特征(14:30) + T收盘(实盘路径)'},
    '④': {'feature': 'intraday', 'exec': 't1_close', 'label': '④ 盘中特征(14:30) + T+1收盘(特征影响)'},
}

FACTOR_COLS = [
    'momentum', 'trend', 'volume_price', 'rsrs',
    'relative_strength', 'weekly_modifier', 'ma60_slope', 'total_score',
]

# 子区间起点（min15 数据覆盖起点）
SUB_START = '2026-01-05'


# ── 数据加载 ─────────────────────────────────────────────


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)


def load_daily(config: dict) -> pd.DataFrame:
    """symbol 日线（前复权，腾讯源）"""
    return load_csv(PROJECT_ROOT / config['data_dir'] / config['symbol'] / 'daily.csv')


def load_min15(config: dict) -> pd.DataFrame:
    """symbol 15 分钟线：date,time,open,high,low,close,volume,amount"""
    return load_csv(PROJECT_ROOT / config['data_dir'] / config['symbol'] / 'min15.csv')


def load_benchmark(config: dict) -> pd.DataFrame:
    """benchmark 日线（相对强弱因子用）"""
    return load_csv(PROJECT_ROOT / config['data_dir'] / config['benchmark'] / 'daily.csv')


def load_prod_features(config: dict) -> pd.DataFrame:
    """生产特征缓存（收盘口径，含 total_score）——组①②的特征来源 + 对拍基准"""
    path = PROJECT_ROOT / config['data_dir'] / config['symbol'] / 'features_cache.csv'
    return load_csv(path)


# ── 盘中特征构造 ──────────────────────────────────────────


def parse_min_time(time_val) -> int:
    """min15 time 列 '20260104143000000' → HHMM 整数 1430"""
    s = str(int(time_val))
    if len(s) < 12:
        raise ValueError(f"无法解析 min15 time: {time_val}")
    return int(s[8:12])


def aggregate_intraday_bars(min15_df: pd.DataFrame,
                            cutoff_hhmm: int = INTRADAY_CUTOFF_HHMM) -> pd.DataFrame:
    """min15 按日聚合「截至 cutoff 的 bar」→ 日级 open/high/low/close/volume/amount

    - open  = 当日首根 bar 的 open
    - close = 截至 cutoff 的最后一根 bar 的 close（14:30 快照价）
    - high/low = 截至 cutoff 的极值
    - volume/amount = 截至 cutoff 的累加
    cutoff 之后的 bar（14:45 / 15:00）被剔除（决策时点不可见）。
    """
    df = min15_df.copy()
    df['hhmm'] = df['time'].map(parse_min_time)
    # 14:25 决策：排除 14:30 的 bar（该 bar 在 14:30 才完成，决策时不可见）
    df = df[df['hhmm'] < cutoff_hhmm].sort_values(['date', 'hhmm'])
    if len(df) == 0:
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    agg = df.groupby('date').agg(
        open=('open', 'first'),
        close=('close', 'last'),
        high=('high', 'max'),
        low=('low', 'min'),
        volume=('volume', 'sum'),
        amount=('amount', 'sum'),
    ).reset_index()
    return agg


def compute_feature_frame(df_sym: pd.DataFrame, df_bench: pd.DataFrame,
                          config: dict) -> pd.DataFrame:
    """与生产 compute_features.compute_all_features 的因子部分同构（逐列复用生产函数）"""
    windows = {
        'momentum': config.get('momentum_window', 5),
        'trend': config.get('trend_window', 20),
        'atr': config.get('atr_window', 14),
        'rsrs': config.get('rsrs_window', 18),
        'rel_strength': config.get('rel_strength_window', 20),
    }
    features = df_sym[['date', 'close', 'volume']].copy()
    features['momentum'] = cf.momentum_score(df_sym, windows['momentum'])
    features['trend'] = cf.trend_score(df_sym, windows['trend'])
    features['volume_price'] = cf.volume_price_score(df_sym)
    features['rsrs'] = cf.rsrs_score(df_sym, windows['rsrs'])
    features['relative_strength'] = cf.relative_strength_score(df_sym, df_bench, windows['rel_strength'])
    features['weekly_modifier'] = cf.weekly_modifier(df_sym, config)
    features['ma60_slope'] = cf.ma60_slope(df_sym)
    features['total_score'] = cf.compute_total_score(features, config)
    return features


def build_intraday_feature_series(daily_df: pd.DataFrame, min15_df: pd.DataFrame,
                                  bench_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """盘中特征序列：daily 全历史，min15 覆盖日的当日 bar 替换为「截至 14:30 聚合 bar」，
    再走生产因子函数。

    语义：2026-01-05 之前的行 = 收盘口径（仅作历史窗口）；之后的行 = 盘中口径（实盘 T 日
    14:25 决策时可见的信息）。benchmark 无 min15 数据，保持收盘口径——relative_strength
    列因此含当日 benchmark 收盘信息，但该因子权重为 0（config.weights 无此项），
    不影响 total_score。
    """
    intraday = aggregate_intraday_bars(min15_df)
    if len(intraday) == 0:
        raise ValueError("min15 数据为空，无法构造盘中特征序列")
    covered = set(pd.to_datetime(intraday['date']))
    df = daily_df.copy()
    mask = pd.to_datetime(df['date']).isin(covered)
    if mask.sum() == 0:
        raise ValueError("min15 覆盖日期与 daily 无交集")
    replace_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
    idx = intraday.set_index('date')
    for col in replace_cols:
        df.loc[mask, col] = idx.loc[pd.to_datetime(df.loc[mask, 'date']), col].values
    return compute_feature_frame(df, bench_df, config)


# ── 参数化回测（生产 run_backtest 同构复制，仅成交价来源参数化）──────────


def run_backtest_timing(
    features_df: pd.DataFrame,
    config: dict,
    start: str,
    end: str,
    cost_rate: float,
    exec_price: str = 't1_close',
) -> dict:
    """
    逐天模拟状态机回测（与 backtest.run_backtest 逐字同构，仅成交价来源参数化）。

    exec_price:
      't1_close' — T 日特征 → T 日信号 → T+1 日收盘成交（现状回测口径，与生产一致）
      't_close'  — T 日特征 → T 日信号 → T 日收盘成交（14:25 决策 → 当天尾盘成交）
    仓位：连续仓位模拟 0%~70%（复用 signal_rules 的 calc_position 逻辑）
    """
    if exec_price not in ('t1_close', 't_close'):
        raise ValueError(f"exec_price 必须是 't1_close' 或 't_close'，得到 {exec_price}")

    # 过滤区间
    mask = (
        (features_df['date'] >= pd.Timestamp(start))
        & (features_df['date'] <= pd.Timestamp(end))
    )
    df = features_df[mask].sort_values('date').reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"回测区间 {start} ~ {end} 数据不足 ({len(df)} 条)")

    n_days = len(df)

    # 初始状态（空仓起步）
    state = {'state': '空仓', 'waiting_days': 0, 'last_decision_date': None}

    # 帐户跟踪
    # t1_close：T 日特征出信号，T+1 日收盘执行交易（最后一天不产生交易）
    # t_close：T 日特征出信号，T 日收盘执行交易（最后一天可交易）
    cash = 1.0
    holding_value = 0.0
    holding_shares = 0.0
    last_close = 0.0

    prev_signal = '空仓'
    trades = []
    daily = []

    # ── 独立风控层状态（与生产完全一致；enabled=false 时全部旁路）──
    rc = parse_risk_config(config)
    rc_enabled = rc['enabled']
    stop_loss_pct = rc['stop_loss_pct'] if rc_enabled else None
    dd_limit_pct = rc['dd_limit_pct'] if rc_enabled else None
    cooldown_days = rc['cooldown_days'] if rc_enabled else 0
    peak_equity = 0.0
    entry_price = 0.0
    cooldown_remaining = 0
    risk_events = []

    # 循环范围：t1_close 只到倒数第二天（最后一天无成交）；t_close 全量
    loop_end = n_days - 1 if exec_price == 't1_close' else n_days

    for i in range(loop_end):
        today = df.iloc[i]
        if exec_price == 't1_close':
            exec_row = df.iloc[i + 1]   # T+1 行
        else:
            exec_row = today            # T 行
        exec_date = exec_row['date']
        exec_close = exec_row['close']
        total_score = today.get('total_score', 0)

        # ── 决策（T 日特征）──
        row_dict = today.to_dict()
        result = decide(
            row_dict, features_df,
            state_override=state, persist=False,
            config_override=config,
        )
        signal = result['decision']
        state = result['_new_state']

        # ── 风控检查（enabled=false 完全旁路；估值用成交日收盘价）──
        risk_reason = None
        cooldown_active = False
        if rc_enabled:
            cooldown_active = check_cooldown(cooldown_remaining)
            if cooldown_remaining > 0:
                cooldown_remaining = tick_cooldown(cooldown_remaining)
            if holding_shares > 1e-9:
                current_holdings_value = holding_shares * exec_close
                total_equity = current_holdings_value + cash
                if check_stop_loss(entry_price, exec_close, stop_loss_pct):
                    risk_reason = '止损'
                elif check_drawdown_limit(peak_equity, total_equity, dd_limit_pct):
                    risk_reason = '回撤熔断'

        if risk_reason is not None:
            sell_shares = holding_shares
            sell_value = sell_shares * exec_close
            cost = sell_value * cost_rate
            cash += sell_value - cost
            holding_shares = 0.0
            holding_value = 0.0
            action = '清仓'
            risk_events.append({
                'date': exec_date,
                'reason': risk_reason,
                'equity': round(total_equity, 6),
                'close': float(exec_close),
                'entry_price': round(entry_price, 6) if entry_price else None,
            })
            remaining = sell_shares
            for j in range(len(trades) - 1, -1, -1):
                t = trades[j]
                if t['action'] not in ('买入', '加仓'):
                    continue
                if t['exit_date'] is not None:
                    continue
                lot = t.get('entry_shares', t['entry_value'] / t['entry_price'])
                if lot <= 1e-12:
                    continue
                if remaining >= lot - 1e-12:
                    remaining -= lot
                    t['exit_date'] = exec_date
                    t['exit_price'] = exec_close
                    t['exit_value'] = lot * exec_close
                    t['return'] = (exec_close - t['entry_price']) / t['entry_price']
                else:
                    t['entry_shares'] = lot - remaining
                    t['entry_value'] = t['entry_shares'] * t['entry_price']
                    remaining = 0.0
                if remaining <= 1e-12:
                    break

            trades.append({
                'action': '清仓',
                'entry_date': exec_date,
                'entry_price': exec_close,
                'entry_value': sell_value,
                'exit_date': None,
                'exit_price': None,
                'exit_value': None,
                'return': None,
                'signal_date': today['date'],
                'signal_score': total_score,
                'reason': risk_reason,
            })
            cooldown_remaining = max(1, set_cooldown(cooldown_days))
            entry_price = 0.0
            peak_equity = cash

        # ── 交易执行：连续仓位调整 ──
        if risk_reason is None:
            action = '不动'
        current_holdings_value = holding_shares * exec_close
        total_equity = current_holdings_value + cash

        if total_equity > 1e-6:
            target_pct = float(result['position'].rstrip('%')) / 100.0
            current_pct = current_holdings_value / total_equity
            gap_pct = target_pct - current_pct

            if abs(gap_pct) > 0.05:
                gap_value = gap_pct * total_equity

                if gap_value > 0 and not (rc_enabled and (cooldown_active or risk_reason is not None)):  # 买入
                    cost = gap_value * cost_rate
                    total_needed = gap_value + cost
                    if total_needed <= cash:
                        new_shares = gap_value / exec_close
                        holding_shares += new_shares
                        cash -= total_needed
                        holding_value = holding_shares * exec_close
                        if rc_enabled:
                            entry_price = update_entry_price(
                                holding_shares - new_shares, entry_price,
                                new_shares, float(exec_close))

                        is_first_buy = abs(current_pct) < 1e-6
                        act_label = '买入' if is_first_buy else '加仓'
                        action = act_label
                        trades.append({
                            'action': act_label,
                            'entry_date': exec_date,
                            'entry_price': exec_close,
                            'entry_value': gap_value,
                            'entry_shares': gap_value / exec_close,
                            'exit_date': None,
                            'exit_price': None,
                            'exit_value': None,
                            'return': None,
                            'signal_date': today['date'],
                            'signal_score': total_score,
                        })
                    else:
                        action = '不动(现金不足)'
                elif gap_value > 0:
                    action = '不动(冷却期)' if risk_reason is None else '清仓'

                else:  # 卖出
                    sell_value = -gap_value
                    cost = sell_value * cost_rate
                    if sell_value <= current_holdings_value:
                        sell_shares = sell_value / exec_close
                        holding_shares -= sell_shares
                        cash += sell_value - cost
                        holding_value = holding_shares * exec_close

                        is_full_close = target_pct < 1e-6
                        action = '清仓' if is_full_close else '减仓'
                        if rc_enabled and is_full_close:
                            entry_price = 0.0

                        remaining = sell_shares
                        for j in range(len(trades) - 1, -1, -1):
                            t = trades[j]
                            if t['action'] not in ('买入', '加仓'):
                                continue
                            if t['exit_date'] is not None:
                                continue
                            lot = t.get('entry_shares', t['entry_value'] / t['entry_price'])
                            if lot <= 1e-12:
                                continue
                            if remaining >= lot - 1e-12:
                                remaining -= lot
                                t['exit_date'] = exec_date
                                t['exit_price'] = exec_close
                                t['exit_value'] = lot * exec_close
                                t['return'] = (exec_close - t['entry_price']) / t['entry_price']
                            else:
                                t['entry_shares'] = lot - remaining
                                t['entry_value'] = t['entry_shares'] * t['entry_price']
                                remaining = 0.0
                            if remaining <= 1e-12:
                                break

                        trades.append({
                            'action': action,
                            'entry_date': exec_date,
                            'entry_price': exec_close,
                            'entry_value': sell_value,
                            'exit_date': None,
                            'exit_price': None,
                            'exit_value': None,
                            'return': None,
                            'signal_date': today['date'],
                            'signal_score': total_score,
                        })

        # ── 日末权益（成交日收盘后）──
        total_equity = cash + holding_shares * exec_close
        last_close = exec_close
        if rc_enabled:
            peak_equity = update_peak_equity(total_equity, peak_equity)

        daily.append({
            'date': exec_date,
            'equity': round(total_equity, 6),
            'signal': signal,
            'total_score': round(total_score, 4),
            'position': round(holding_shares * exec_close / total_equity if total_equity > 0 else 0, 4),
            'cash': round(cash, 6),
            'hold_value': round(holding_shares * exec_close, 6),
        })

        prev_signal = signal

    # t1_close：补上最后一笔权益（最后一天只有特征，没有交易）——与生产一致
    if exec_price == 't1_close' and n_days > 0:
        last_row = df.iloc[-1]
        last_equity = cash + holding_shares * last_row['close']
        daily.append({
            'date': last_row['date'],
            'equity': round(last_equity, 6),
            'signal': prev_signal,
            'total_score': round(last_row.get('total_score', 0), 4),
            'position': round(holding_shares * last_row['close'] / last_equity if last_equity > 0 else 0, 4),
            'cash': round(cash, 6),
            'hold_value': round(holding_shares * last_row['close'], 6),
        })
    # t_close：循环已覆盖最后一天（当日收盘可成交），无需补充

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=['action', 'entry_date', 'entry_price', 'entry_value', 'entry_shares',
                 'exit_date', 'exit_price', 'exit_value', 'return',
                 'signal_date', 'signal_score']
    )
    daily_df = pd.DataFrame(daily) if daily else pd.DataFrame()

    return {
        'trades': trades_df,
        'equity_curve': daily_df,
        'final_value': float(round(cash + holding_shares * last_close, 6)),
        'risk_events': risk_events,
    }


# ── 同构校验 ──────────────────────────────────────────────


def assert_isomorphic(prod_result: dict, timing_result: dict, tag: str = ''):
    """断言参数化回测 t1_close 模式与生产 run_backtest 输出全等"""
    pe, te = prod_result['equity_curve'], timing_result['equity_curve']
    pt, tt = prod_result['trades'], timing_result['trades']

    if len(pe) != len(te):
        raise AssertionError(f"同构校验失败({tag}): equity 长度 {len(pe)} != {len(te)}")
    eq_ok = (
        pe[['date', 'equity']].reset_index(drop=True).equals(te[['date', 'equity']].reset_index(drop=True))
        and np.allclose(
            pe['equity'].astype(float).values, te['equity'].astype(float).values,
            rtol=1e-9, atol=1e-12,
        )
    )
    if not eq_ok:
        raise AssertionError(f"同构校验失败({tag}): equity_curve 不一致")

    if len(pt) != len(tt):
        raise AssertionError(f"同构校验失败({tag}): trades 长度 {len(pt)} != {len(tt)}")
    if len(pt) > 0:
        cols = [c for c in ['action', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
                            'signal_date', 'signal_score', 'return'] if c in pt.columns]
        # DataFrame.equals：NaT/NaN 视为相等（NaT == NaT 为 False，不能逐位比较）
        if not pt[cols].reset_index(drop=True).equals(tt[cols].reset_index(drop=True)):
            raise AssertionError(f"同构校验失败({tag}): trades 列不一致")
        if not np.isclose(prod_result['final_value'], timing_result['final_value'], rtol=1e-9, atol=1e-12):
            raise AssertionError(f"同构校验失败({tag}): final_value 不一致")
    print(f"  [OK] 同构校验通过({tag}): equity {len(pe)} 行 / trades {len(pt)} 条 / final={timing_result['final_value']:.6f} "
          f"== 生产 run_backtest")


# ── 对拍 corr 报告 ────────────────────────────────────────


def factor_corr(a_df: pd.DataFrame, b_df: pd.DataFrame,
                cols: list = None, subset_start: str = None) -> dict:
    """两特征序列逐因子 corr（非 NaN 对齐；全 NaN 或零方差异常列返回 None）"""
    cols = cols or FACTOR_COLS
    a = a_df.copy()
    b = b_df.copy()
    if subset_start is not None:
        a = a[a['date'] >= pd.Timestamp(subset_start)]
        b = b[b['date'] >= pd.Timestamp(subset_start)]
    merged = a.merge(b, on='date', suffixes=('_a', '_b'))
    out = {}
    for c in cols:
        if f'{c}_a' not in merged.columns or f'{c}_b' not in merged.columns:
            out[c] = None
            continue
        x = merged[f'{c}_a'].astype(float)
        y = merged[f'{c}_b'].astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= 3 and x[m].std() > 1e-12:
            out[c] = float(x[m].corr(y[m]))
        else:
            out[c] = None
    return out


def print_corr_report(title: str, corr: dict):
    print(f"  ── {title} ──")
    for c in FACTOR_COLS:
        v = corr.get(c)
        s = f"{v:.6f}" if v is not None else "   n/a "
        print(f"    {c:<18s} corr = {s}")


# ── 回测运行与指标 ────────────────────────────────────────


def run_group(features: dict, config: dict, start: str, end: str, cost: float,
              group_key: str) -> dict:
    """跑一组回测，返回指标 + 结果"""
    gf = GROUPS[group_key]
    feat_df = features[gf['feature']]
    result = run_backtest_timing(feat_df, config, start, end, cost, gf['exec'])
    eq = result['equity_curve']
    if len(eq) == 0:
        raise ValueError(f"组 {group_key} 回测无权益曲线")
    metrics = compute_metrics(eq['equity'], trades_df=result['trades'], n_days=len(eq))
    return {'metrics': metrics, 'result': result}


def metrics_row(metrics: dict) -> dict:
    return {
        '总收益%': metrics['total_return'],
        '年化%': metrics['annual_return'],
        '夏普': metrics['sharpe'],
        '回撤%': metrics['max_drawdown'],
        '交易数': metrics['trade_count'],
        '胜率%': metrics['win_rate'],
    }


def print_group_table(rows: list, header_note: str):
    """rows: list of (组标识, 含义, metrics dict)"""
    print(f"\n  {header_note}")
    print(f"  {'组':<4s} {'含义':<30s} {'总收益%':>8s} {'年化%':>7s} {'夏普':>6s} "
          f"{'回撤%':>7s} {'交易数':>6s} {'胜率%':>6s}")
    print(f"  {'─' * (4 + 30 + 8 + 7 + 6 + 7 + 6 + 6 + 9)}")
    for gk, label, m in rows:
        print(f"  {gk:<4s} {label:<30s} {m['total_return']:>7.2f}% {m['annual_return']:>6.2f}% "
              f"{m['sharpe']:>6.2f} {m['max_drawdown']:>6.2f}% {m['trade_count']:>6d} {m['win_rate']:>5.1f}%")


def build_conclusion(full: dict, sub: dict) -> list:
    """按任务给定的结论标准生成结论文本（返回行列表）"""
    lines = []
    lines.append("")
    lines.append("═" * 70)
    lines.append("  结论：实盘应 T 日尾盘执行 还是 T+1 执行？")
    lines.append("═" * 70)

    m1, m2 = full['①'], full['②']
    lines.append(f"\n  [全段 {full['_label']}] 纯时点差（②收盘特征+T收盘 vs ①基线）:")
    d_ret = m2['annual_return'] - m1['annual_return']
    d_sharpe = m2['sharpe'] - m1['sharpe']
    d_dd = m2['max_drawdown'] - m1['max_drawdown']
    lines.append(f"    年化 {m2['annual_return']:+.2f}% vs {m1['annual_return']:+.2f}%  (Δ{d_ret:+.2f}pp)")
    lines.append(f"    夏普 {m2['sharpe']:+.2f} vs {m1['sharpe']:+.2f}       (Δ{d_sharpe:+.2f})")
    lines.append(f"    回撤 {m2['max_drawdown']:+.2f}% vs {m1['max_drawdown']:+.2f}%  (Δ{d_dd:+.2f}pp)")

    m3, m4 = sub['③'], sub['④']
    s1 = sub['①']
    lines.append(f"\n  [子区间 {sub['_label']}] 实盘路径（③盘中特征+T收盘）vs 回测路径（①基线）:")
    d_total = m3['annual_return'] - s1['annual_return']
    lines.append(f"    年化 {m3['annual_return']:+.2f}% vs {s1['annual_return']:+.2f}%  (Δ{d_total:+.2f}pp)")
    lines.append(f"    夏普 {m3['sharpe']:+.2f} vs {s1['sharpe']:+.2f}")

    # 特征口径影响 = ④ - ③（盘中特征下成交时点不重要时，③≈④ 说明成交时点不重要）
    d_feat_t1 = m4['annual_return'] - s1['annual_return']
    d_feat_t = m3['annual_return'] - sub['②']['annual_return']
    d_time_intraday = m4['annual_return'] - m3['annual_return']
    lines.append(f"\n  [子区间] 特征口径影响（④ vs ①，T+1 成交下）: Δ年化 {d_feat_t1:+.2f}pp，Δ夏普 {m4['sharpe']-s1['sharpe']:+.2f}")
    lines.append(f"  [子区间] 特征口径影响（③ vs ②，T 成交下）: Δ年化 {d_feat_t:+.2f}pp")
    lines.append(f"  [子区间] 盘中口径下时点差（④ vs ③）: Δ年化 {d_time_intraday:+.2f}pp")

    # 判定
    thr = 2.0  # 年化 pp 差异显著阈值（子区间样本约 150 天，宽松阈值）
    feat_close = abs(m4['annual_return'] - s1['annual_return']) < thr
    time_close = abs(d_time_intraday) < thr
    early_ok = m2['annual_return'] >= m1['annual_return']  # 全段 T 收盘不劣于 T+1
    total_gap = m3['annual_return'] - s1['annual_return']  # 实盘路径 vs 回测路径

    lines.append("\n  判定逻辑：")
    lines.append(f"    · 盘中特征 vs 收盘特征（T+1 成交下 ④vs① 年化差 {d_feat_t1:+.2f}pp）"
                 f"{'→ 特征口径接近，盘中特征可代替收盘特征' if feat_close else '→ 特征口径差异明显'}")
    lines.append(f"    · 成交时点（盘中特征下 ④vs③ 年化差 {d_time_intraday:+.2f}pp）"
                 f"{'→ 时点影响小，T 日尾盘可执行' if time_close else '→ 时点影响显著'}")
    lines.append(f"    · 全段时点差（②vs① 年化差 {d_ret:+.2f}pp）"
                 f"{'→ T 日成交不劣于 T+1' if early_ok else '→ T+1 成交占优'}")
    lines.append(f"    · 实盘路径 vs 回测路径总差距（③vs① 年化差 {total_gap:+.2f}pp，交易 {m3['trade_count']} 笔）"
                 f"→ 回测对实盘有 {abs(total_gap):.1f}pp 乐观偏差，实盘预期收益应相应打折")

    if feat_close and time_close and early_ok:
        verdict = ("✅ 建议：实盘可 T 日尾盘执行——盘中特征(14:30)与收盘特征高度接近，"
                   "成交时点差无显著损失，且 T 日成交享受动量延续。")
    elif early_ok and not time_close:
        verdict = ("⚠️ 建议：倾向 T 日尾盘执行——时点差全段为正，但盘中特征口径下时点差异显著，"
                   "需注意盘中信号相对收盘信号的确认损耗。")
    elif not early_ok:
        verdict = ("⚠️ 建议：倾向 T+1 执行——全段 T 日成交劣于 T+1，等次日收盘确认更优；"
                   "若实盘必须当日执行，接受动量损耗。")
    else:
        verdict = ("➡️ 建议：T 日尾盘与 T+1 差异不显著（样本内），实盘按执行便捷性选择；"
                   "建议用纸面盘（paper_trade）跟踪 T 日尾盘执行偏差。")
    lines.append(f"\n  {verdict}")
    lines.append(f"\n  注：组③/④ 的成交价 = 特征序列 close（盘中口径为 14:30 快照价），"
                 f"未计 14:30~收盘 最后 30 分钟滑点与流动性成本；"
                 f"真实尾盘执行价应在 14:30 快照价基础上加 0~0.1% 偏差后再看结论稳健性。")
    lines.append("═" * 70)
    return lines


# ── 主流程 ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='决策点 F：成交时点 + 特征口径双对比（T+0 vs T+1）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--cost', type=float, default=0.00055, help='单边费率（默认万5.5，与生产一致）')
    parser.add_argument('--no-save', action='store_true', help='不写 CSV')
    args = parser.parse_args()
    cost = args.cost

    config = load_config()
    symbol = config['symbol']

    print(f"\n🔧 决策点 F — 成交时点 + 特征口径双对比（{symbol}）")
    print(f"{'=' * 70}")

    # ── 加载数据 ──
    print("\n📂 加载数据...")
    daily = load_daily(config)
    min15 = load_min15(config)
    bench = load_benchmark(config)
    prod = load_prod_features(config)
    print(f"  daily: {len(daily)} 行 ({daily['date'].min().date()} ~ {daily['date'].max().date()})")
    print(f"  min15: {len(min15)} 行 ({min15['date'].min().date()} ~ {min15['date'].max().date()})")
    print(f"  features_cache(生产收盘特征): {len(prod)} 行")

    full_end = daily['date'].max().date().isoformat()
    sub_end = min15['date'].max().date().isoformat()
    full_label = f"{daily['date'].min().date()} ~ {full_end}"
    sub_label = f"{SUB_START} ~ {sub_end}"
    print(f"  全段区间: {full_label} | 子区间(min15 覆盖): {sub_label}")

    # ── 1. 复用一致性对拍：用生产因子函数重算收盘特征 vs 生产缓存 ──
    print("\n🔬 对拍 1/2：生产因子函数重算（收盘口径） vs features_cache.csv（全段）")
    recomputed_close = compute_feature_frame(daily, bench, config)
    corr_consistency = factor_corr(prod, recomputed_close)
    print_corr_report("复用一致性（期望 corr=1.0）", corr_consistency)
    bad = [c for c, v in corr_consistency.items() if v is not None and v < 0.999999]
    if bad:
        raise SystemExit(f"[FATAL] 因子复用对拍不一致: {bad} —— 生产缓存口径与本脚本不同，终止实验")

    # ── 2. 盘中特征序列 + 对拍 2/2：盘中 vs 收盘特征（子区间）──
    print("\n🕒 构造盘中特征序列（daily 全历史 + min15 覆盖日替换为截至 14:30 bar）...")
    intraday = build_intraday_feature_series(daily, min15, bench, config)
    print(f"  盘中特征序列: {len(intraday)} 行")
    print("\n🔬 对拍 2/2：盘中特征 vs 收盘特征（子区间起逐因子 corr）")
    corr_intraday = factor_corr(recomputed_close, intraday, subset_start=SUB_START)
    print_corr_report(f"盘中(14:30) vs 收盘特征 [{sub_label}]", corr_intraday)

    # ── 3. 同构校验：t1_close 模式 == 生产 run_backtest ──
    print("\n🔧 同构校验：参数化回测(t1_close) vs 生产 run_backtest（全段，收盘特征）...")
    prod_result = run_backtest(prod, config, str(daily['date'].min().date()), full_end, cost)
    timing_result = run_backtest_timing(prod, config, str(daily['date'].min().date()), full_end, cost, 't1_close')
    assert_isomorphic(prod_result, timing_result, tag='全段①')

    # 与既有存档 metrics.json 对照
    out_dir = PROJECT_ROOT / config['data_dir'] / symbol / 'backtest'
    out_csv = out_dir / 'execution_timing_对比.csv'
    metrics_path = out_dir / 'metrics.json'
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding='utf-8'))
        eq = prod_result['equity_curve']
        cur = compute_metrics(eq['equity'], trades_df=prod_result['trades'], n_days=len(eq))
        diff = {k: round(cur[k] - saved[k], 4) for k in ['total_return', 'annual_return', 'sharpe',
                                                         'max_drawdown', 'trade_count'] if k in saved}
        close = all(abs(v) < 1e-6 or (isinstance(v, float) and abs(v) < 0.02) for v in diff.values())
        status = '[OK] 与生产存档 metrics.json 一致' if close else f'[WARN] 与存档有差: {diff}'
        print(f"  {status}（存档: {saved.get('total_return')}% / sharpe {saved.get('sharpe')} / 交易 {saved.get('trade_count')}）")

    features = {'close': prod, 'intraday': intraday}

    # ── 4. 四组回测 ──
    full_start = str(daily['date'].min().date())
    print(f"\n📊 回测（费率单边 {cost*10000:.1f}‱）...")

    print(f"\n  ▶ 全段 [{full_label}]：组①②（收盘特征，时点差）")
    full = {}
    for gk in ('①', '②'):
        r = run_group(features, config, full_start, full_end, cost, gk)
        full[gk] = r['metrics']
        print(f"    组{gk} {GROUPS[gk]['label']}: 年化 {r['metrics']['annual_return']:+.2f}% "
              f"夏普 {r['metrics']['sharpe']:.2f} 回撤 {r['metrics']['max_drawdown']:.2f}% "
              f"交易 {r['metrics']['trade_count']} 笔")
    full['_label'] = full_label

    print(f"\n  ▶ 子区间 [{sub_label}]：组①②③④")
    sub = {}
    for gk in ('①', '②', '③', '④'):
        r = run_group(features, config, SUB_START, sub_end, cost, gk)
        sub[gk] = r['metrics']
        print(f"    组{gk} {GROUPS[gk]['label']}: 年化 {r['metrics']['annual_return']:+.2f}% "
              f"夏普 {r['metrics']['sharpe']:.2f} 回撤 {r['metrics']['max_drawdown']:.2f}% "
              f"交易 {r['metrics']['trade_count']} 笔")
    sub['_label'] = sub_label

    # ── 5. 对比表 ──
    print("\n" + "═" * 70)
    print(f"  对比汇总 — 全段 [{full_label}]（样本 {len(prod):d} 天）")
    print("═" * 70)
    print_group_table(
        [(gk, GROUPS[gk]['label'], full[gk]) for gk in ('①', '②')],
        f"全段 [{full_label}] ① vs ②（纯时点差）",
    )

    print("\n" + "═" * 70)
    print(f"  对比汇总 — 子区间 [{sub_label}]（min15 覆盖）")
    print("═" * 70)
    print_group_table(
        [(gk, GROUPS[gk]['label'], sub[gk]) for gk in ('①', '②', '③', '④')],
        f"子区间 [{sub_label}] ①②③④ 全比",
    )

    # ── 6. 结论 ──
    conclusion = build_conclusion(full, sub)
    for line in conclusion:
        print(line)

    # ── 7. 保存 CSV ──
    if not args.no_save:
        rows = []
        for seg, label, groups in (('全段', full_label, ['①', '②']),
                                   ('子区间', sub_label, ['①', '②', '③', '④'])):
            pool = full if seg == '全段' else sub
            for gk in groups:
                m = pool[gk]
                rows.append({
                    'segment': seg, 'segment_range': label,
                    'group': gk, 'feature': GROUPS[gk]['feature'],
                    'exec_price': GROUPS[gk]['exec'],
                    **{k: v for k, v in metrics_row(m).items()},
                })
        # 差异行（子区间）
        s = sub
        rows.append({'segment': '子区间', 'segment_range': sub_label, 'group': 'Δ时点差(②-①)',
                     'feature': 'close', 'exec_price': '-',
                     '总收益%': s['②']['total_return'] - s['①']['total_return'],
                     '年化%': s['②']['annual_return'] - s['①']['annual_return'],
                     '夏普': round(s['②']['sharpe'] - s['①']['sharpe'], 4),
                     '回撤%': s['②']['max_drawdown'] - s['①']['max_drawdown'],
                     '交易数': s['②']['trade_count'] - s['①']['trade_count'], '胜率%': 0.0})
        rows.append({'segment': '子区间', 'segment_range': sub_label, 'group': 'Δ实盘路径(③-①)',
                     'feature': '-', 'exec_price': '-',
                     '总收益%': s['③']['total_return'] - s['①']['total_return'],
                     '年化%': s['③']['annual_return'] - s['①']['annual_return'],
                     '夏普': round(s['③']['sharpe'] - s['①']['sharpe'], 4),
                     '回撤%': s['③']['max_drawdown'] - s['①']['max_drawdown'],
                     '交易数': s['③']['trade_count'] - s['①']['trade_count'], '胜率%': 0.0})
        rows.append({'segment': '子区间', 'segment_range': sub_label, 'group': 'Δ特征口径(④-③)',
                     'feature': '-', 'exec_price': '-',
                     '总收益%': s['④']['total_return'] - s['③']['total_return'],
                     '年化%': s['④']['annual_return'] - s['③']['annual_return'],
                     '夏普': round(s['④']['sharpe'] - s['③']['sharpe'], 4),
                     '回撤%': s['④']['max_drawdown'] - s['③']['max_drawdown'],
                     '交易数': s['④']['trade_count'] - s['③']['trade_count'], '胜率%': 0.0})

        out_df = pd.DataFrame(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\n  [SAVE] 对比表 → {out_csv}")

        # 附加 corr 报告 CSV
        corr_path = out_dir / 'execution_timing_因子corr.csv'
        corr_rows = []
        for c in FACTOR_COLS:
            corr_rows.append({'factor': c,
                              'consistency_corr(重算vs生产缓存)': corr_consistency.get(c),
                              'intraday_vs_close_corr(子区间)': corr_intraday.get(c)})
        pd.DataFrame(corr_rows).to_csv(corr_path, index=False, encoding='utf-8-sig')
        print(f"  [SAVE] 因子 corr 报告 → {corr_path}")

    print(f"\n  [DONE] 决策点 F 实验完成。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
