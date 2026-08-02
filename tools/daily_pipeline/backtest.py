#!/usr/bin/env python3
"""
backtest.py — 588000 日线择时回测框架

用法：
  python backtest.py                               # 全量回测（默认）
  python backtest.py --start 2024-01-01             # 指定起始
  python backtest.py --end 2025-12-31               # 指定结束
  python backtest.py --cost 0.00055                 # 自定义单边费率
  python backtest.py --factor-attribution           # 因子归因分析
  python backtest.py --output ./backtest            # 输出目录

性能指标：
  年化收益率 / 年化波动率 / 夏普比率 / 最大回撤 / 卡玛比率
  胜率 / 盈亏比 / 交易次数 / 平均持仓天数
  超额收益 / 超额夏普（vs 持有不动、vs 双均线交叉）

信号对齐（防未来函数）：
  T 日特征 → T 日信号 → T+1 日收盘成交
"""

import argparse
import copy
import json
import math
import sys
from datetime import datetime, date
from pathlib import Path

import yaml

from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

sys.path.insert(0, str(SCRIPT_DIR))
from signal_rules import decide


# ── 配置 ─────────────────────────────────────────────

# 市场环境区间（用于分段归因）
END_DATE = date.today().strftime('%Y-%m-%d')
REGIME_SLICES = {
    '📉 下跌段': ('2023-01-03', '2024-09-23'),
    '📈 牛市段': ('2024-09-24', END_DATE),
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_features_df(symbol: str) -> pd.DataFrame:
    """读特征缓存，直接从 CSV 加载"""
    config = load_config()
    data_dir = PROJECT_ROOT / config["data_dir"]
    path = data_dir / symbol / "features_cache.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"特征缓存不存在: {path}\n"
            "请先运行 python daily_panel.py 生成特征数据"
        )
    df = pd.read_csv(path, parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)


# ── 回测引擎 ────────────────────────────────────────


def run_backtest(
    features_df: pd.DataFrame,
    config: dict,
    start: str,
    end: str,
    cost_rate: float,
) -> dict:
    """
    逐天模拟状态机回测。

    信号对齐：T 日特征 → T 日信号 → T+1 日收盘成交
    仓位：连续仓位模拟 0%~70%（复用 signal_rules 的 calc_position 逻辑）
    """
    # 过滤区间
    mask = (
        (features_df['date'] >= pd.Timestamp(start))
        & (features_df['date'] <= pd.Timestamp(end))
    )
    df = features_df[mask].sort_values('date').reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"回测区间 {start} ~ {end} 数据不足 ({len(df)} 条)")

    n_days = len(df)
    print(f"  [INFO] 回测区间: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()} "
          f"({n_days} 个交易日)")

    # 初始状态（空仓起步）
    state = {'state': '空仓', 'waiting_days': 0, 'last_decision_date': None}

    # 帐户跟踪
    # 用 T 日特征出信号，在 T+1 日收盘执行交易
    # 最后一天不产生交易（没有 T+1 了）
    cash = 1.0          # 初始净值
    holding_value = 0.0  # 持仓份额 × 收盘价
    holding_shares = 0.0
    last_close = 0.0

    prev_signal = '空仓'
    trades = []
    daily = []   # T+1 日跟踪

    for i in range(n_days - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        today_date = today['date']
        tomorrow_date = tomorrow['date']
        tomorrow_close = tomorrow['close']
        total_score = today.get('total_score', 0)

        # ── 决策 ──
        row_dict = today.to_dict()
        result = decide(
            row_dict, features_df,
            state_override=state, persist=False,
            config_override=config,
        )
        signal = result['decision']
        state = result['_new_state']  # 取出更新后的状态，用于下一轮

        # ── 交易执行：连续仓位调整 ──
        action = '不动'
        current_holdings_value = holding_shares * tomorrow_close
        total_equity = current_holdings_value + cash

        if total_equity > 1e-6:
            target_pct = float(result['position'].rstrip('%')) / 100.0
            current_pct = current_holdings_value / total_equity
            gap_pct = target_pct - current_pct

            if abs(gap_pct) > 0.05:
                gap_value = gap_pct * total_equity

                if gap_value > 0:  # 买入
                    cost = gap_value * cost_rate
                    total_needed = gap_value + cost
                    if total_needed <= cash:
                        holding_shares += gap_value / tomorrow_close
                        cash -= total_needed
                        holding_value = holding_shares * tomorrow_close

                        is_first_buy = abs(current_pct) < 1e-6
                        act_label = '买入' if is_first_buy else '加仓'
                        action = act_label
                        trades.append({
                            'action': act_label,
                            'entry_date': tomorrow_date,
                            'entry_price': tomorrow_close,
                            'entry_value': gap_value,
                            'entry_shares': gap_value / tomorrow_close,  # 批次份额（LIFO 扣减用）
                            'exit_date': None,
                            'exit_price': None,
                            'exit_value': None,
                            'return': None,
                            'signal_date': today_date,
                            'signal_score': total_score,
                        })
                    else:
                        action = '不动(现金不足)'

                else:  # 卖出
                    sell_value = -gap_value
                    cost = sell_value * cost_rate
                    if sell_value <= current_holdings_value:
                        sell_shares = sell_value / tomorrow_close
                        holding_shares -= sell_shares
                        cash += sell_value - cost
                        holding_value = holding_shares * tomorrow_close

                        is_full_close = target_pct < 1e-6
                        action = '清仓' if is_full_close else '减仓'

                        # LIFO 扣减未平仓批次：从最近一笔买入/加仓开始扣份额
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
                                # 整个批次卖完 → 闭合，收益按价格算
                                remaining -= lot
                                t['exit_date'] = tomorrow_date
                                t['exit_price'] = tomorrow_close
                                t['exit_value'] = lot * tomorrow_close
                                t['return'] = (tomorrow_close - t['entry_price']) / t['entry_price']
                                # 保留原始份额与 entry_value 自洽（exit_date 已置，后续循环跳过）
                            else:
                                # 部分卖出：批次剩余份额减少，不闭合
                                t['entry_shares'] = lot - remaining
                                # 保持记录自洽：entry_value 同步剩余份额的成本
                                t['entry_value'] = t['entry_shares'] * t['entry_price']
                                remaining = 0.0
                            if remaining <= 1e-12:
                                break

                        trades.append({
                            'action': action,
                            'entry_date': tomorrow_date,
                            'entry_price': tomorrow_close,
                            'entry_value': sell_value,
                            'exit_date': None,
                            'exit_price': None,
                            'exit_value': None,
                            'return': None,
                            'signal_date': today_date,
                            'signal_score': total_score,
                        })

        # ── 日末权益（T+1 收盘后） ──
        total_equity = cash + holding_shares * tomorrow_close
        last_close = tomorrow_close

        daily.append({
            'date': tomorrow_date,
            'equity': round(total_equity, 6),
            'signal': signal,
            'total_score': round(total_score, 4),
            'position': round(holding_shares * tomorrow_close / total_equity if total_equity > 0 else 0, 4),
            'cash': round(cash, 6),
            'hold_value': round(holding_shares * tomorrow_close, 6),
        })

        prev_signal = signal

    # 补上最后一笔权益（最后一天只有特征，没有交易）
    if n_days > 0:
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
    }


# ── 基准策略 ──────────────────────────────────────────


def run_buy_hold(features_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """持有不动基准"""
    mask = (
        (features_df['date'] >= pd.Timestamp(start))
        & (features_df['date'] <= pd.Timestamp(end))
    )
    df = features_df[mask].sort_values('date').reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame()
    init_price = df['close'].iloc[0]
    df = df.copy()
    df['benchmark_equity'] = df['close'] / init_price
    return df[['date', 'benchmark_equity']]


def run_ma_crossover(
    features_df: pd.DataFrame,
    start: str,
    end: str,
    cost_rate: float,
    fast: int = 5,
    slow: int = 20,
) -> dict:
    """双均线交叉基准（MA5/MA20）"""
    mask = (
        (features_df['date'] >= pd.Timestamp(start))
        & (features_df['date'] <= pd.Timestamp(end))
    )
    df = features_df[mask].sort_values('date').reset_index(drop=True).copy()

    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()
    # 金叉买入，死叉卖出
    df['ma_signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'ma_signal'] = 1
    df['ma_cross'] = df['ma_signal'].diff()

    cash = 1.0
    holding = 0.0
    in_position = False
    trades = []
    daily_equity = []

    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        cross = df['ma_cross'].iloc[i]

        if cross == 1 and not in_position:
            # 金叉买入
            buy = cash * (1 - cost_rate)
            holding = buy / price
            cash = 0.0
            in_position = True
            trades.append({'action': '买入', 'date': df['date'].iloc[i], 'price': price})
        elif cross == -1 and in_position:
            # 死叉卖出
            sell = holding * price * (1 - cost_rate)
            cash = sell
            holding = 0.0
            in_position = False
            trades.append({'action': '卖出', 'date': df['date'].iloc[i], 'price': price})

        equity = cash + holding * price
        daily_equity.append({'date': df['date'].iloc[i], 'equity': equity})

    # 末笔清仓
    if in_position and len(df) > 0:
        final_price = df['close'].iloc[-1]
        sell = holding * final_price * (1 - cost_rate)
        cash = sell
        holding = 0.0

    equity_df = pd.DataFrame(daily_equity) if daily_equity else pd.DataFrame()
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=['action', 'date', 'price']
    )

    return {
        'trades': trades_df,
        'equity_curve': equity_df,
        'final_value': cash + holding * (df['close'].iloc[-1] if len(df) > 0 else 1),
    }


# ── 绩效计算 ──────────────────────────────────────────


def compute_metrics(
    equity_series: pd.Series,
    risk_free_rate: float = 0.02,
    trades_df: Optional[pd.DataFrame] = None,
    n_days: int = 0,
) -> dict:
    """
    从日级权益曲线计算绩效指标

    Parameters
    ----------
    equity_series : pd.Series — 日均权益
    trades_df : pd.DataFrame — 交易记录，用于胜率/盈亏比
    n_days : int — 交易日数（非交易日按 252 计）
    """
    if len(equity_series) < 5:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_vol': 0,
            'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'calmar': 0,
            'omega': 0, 'max_dd_duration': 0,
            'win_rate': 0, 'profit_loss_ratio': 0, 'trade_count': 0,
            'avg_hold_days': 0,
        }

    # 日收益率
    daily_returns = equity_series.pct_change().dropna()

    # 交易日数
    if n_days == 0:
        n_days = len(equity_series)

    total_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1)
    years = n_days / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 0 else 0
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 1e-8 else 0

    # 下行波动率（仅负收益）→ Sortino
    downside = daily_returns[daily_returns < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 1e-8 else 0

    # Omega 比率：正收益加权和 / 负收益加权和（阈值 0）
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    omega = float(gains / losses) if losses > 1e-12 else 0

    # 最大回撤
    peak = equity_series.expanding().max()
    dd = (equity_series - peak) / peak
    max_drawdown = float(dd.min())

    # 卡玛比率
    calmar = abs(annual_return / max_drawdown) if max_drawdown < -1e-8 else 0

    # 最大回撤持续期（从回撤开始到净值创新高，交易日）
    max_dd_duration = 0
    if len(dd) > 0:
        in_dd = False
        dd_start = 0
        cur_duration = 0
        best_duration = 0
        for i, v in enumerate(dd.values):
            if v < -1e-12:
                if not in_dd:
                    in_dd = True
                    dd_start = i
                cur_duration = i - dd_start + 1
            else:
                if in_dd:
                    best_duration = max(best_duration, cur_duration)
                    in_dd = False
                    cur_duration = 0
        if in_dd:  # 未创新高（仍在回撤中）
            best_duration = max(best_duration, cur_duration)
        max_dd_duration = best_duration

    # 交易相关
    win_rate = 0.0
    profit_loss_ratio = 0.0
    trade_count = 0
    avg_hold_days = 0.0

    if trades_df is not None and len(trades_df) > 0:
        # 判断 trade df 的 schema（主策略 vs 基准可能有不同的列）
        has_entry_exit = all(c in trades_df.columns for c in ['exit_date', 'entry_date', 'return'])

        if has_entry_exit:
            # 统计所有完整闭合的建仓批次（买入/加仓，有 exit 记录）
            complete = trades_df[
                trades_df['action'].isin(['买入', '加仓'])
                & trades_df['exit_date'].notna()
            ]
            if len(complete) > 0:
                win_mask = complete['return'] > 0
                win_rate = float(win_mask.sum() / len(complete))
                wins = complete[win_mask]['return']
                losses = complete[~win_mask]['return']
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
                profit_loss_ratio = min(avg_win / avg_loss, 100.0) if avg_loss > 1e-3 else 0

            trade_count = len(complete) if len(complete) > 0 else len(trades_df[trades_df['action'].isin(['买入', '加仓'])])
            # 平均持仓天数（仅建仓批次）
            with_exit = trades_df[
                trades_df['action'].isin(['买入', '加仓'])
                & trades_df['exit_date'].notna()
                & trades_df['entry_date'].notna()
            ].copy()
            if len(with_exit) > 0:
                hold_days = (pd.to_datetime(with_exit['exit_date']) - pd.to_datetime(with_exit['entry_date'])).dt.days
                avg_hold_days = float(hold_days.mean())
        else:
            # 简单 schema（如 MA 基准：只有 action, date, price）
            trade_count = len(trades_df)

    return {
        'total_return': round(total_return * 100, 2),       # %
        'annual_return': round(annual_return * 100, 2),     # %
        'annual_vol': round(annual_vol * 100, 2),           # %
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'max_drawdown': round(max_drawdown * 100, 2),       # %
        'calmar': round(calmar, 4),
        'omega': round(omega, 4),
        'max_dd_duration': max_dd_duration,                 # 交易日
        'win_rate': round(win_rate * 100, 2),               # %
        'profit_loss_ratio': round(profit_loss_ratio, 4),
        'trade_count': trade_count,
        'avg_hold_days': round(avg_hold_days, 1),
    }


def extract_yearly(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    """按年分段计算绩效"""
    if len(equity_df) == 0:
        return {}

    equity_df = equity_df.copy()
    equity_df['year'] = pd.to_datetime(equity_df['date']).dt.year

    years = sorted(equity_df['year'].unique())
    result = {}
    for y in years:
        sub = equity_df[equity_df['year'] == y]
        sub_trades = trades_df[
            trades_df['entry_date'].notna()
            & (pd.to_datetime(trades_df['entry_date']).dt.year == y)
        ] if len(trades_df) > 0 else pd.DataFrame()

        m = compute_metrics(sub['equity'], trades_df=sub_trades, n_days=len(sub))
        result[str(y)] = m

    return result


# ── 因子归因分析 ──────────────────────────────────────



# ── 市场环境分段归因 ─────────────────────────────


def run_regime_analysis(
    features_df: pd.DataFrame,
    config: dict,
    full_start: str,
    full_end: str,
    cost_rate: float,
) -> dict:
    """
    分市场环境运行因子归因。
    分别跑下跌段和牛市段的因子归因，对比因子贡献差异。
    """
    factors = list(config['weights'].keys())

    print("\n  ── 市场环境分段归因 ──")

    results = {}
    for regime_name, (r_start, r_end) in REGIME_SLICES.items():
        # 确保区间不超出全量范围
        if r_end > full_end:
            continue

        print(f"\n  [{regime_name}] {r_start} ~ {r_end}")
        regime_results = {}

        # 全模型
        try:
            base = run_backtest(features_df, config, r_start, r_end, cost_rate)
            base_df = base['equity_curve']
            if len(base_df) > 0:
                m = compute_metrics(base_df['equity'], trades_df=base['trades'], n_days=len(base_df))
                regime_results['full'] = m
        except Exception as e:
            print(f"  [ERR] 全模型回测失败: {e}")
            continue

        for factor in factors:
            adj = {k: v for k, v in config['weights'].items() if k != factor}
            remaining = sum(adj.values())
            if remaining > 0:
                adj = {k: v / remaining for k, v in adj.items()}

            config_copy = copy.deepcopy(config)
            config_copy['weights'] = adj

            feat_copy = features_df.copy()
            feat_copy['total_score'] = sum(
                feat_copy[f].fillna(0) * adj.get(f, 0) for f in factors if f in adj
            )

            try:
                sub = run_backtest(feat_copy, config_copy, r_start, r_end, cost_rate)
                sub_df = sub['equity_curve']
                if len(sub_df) > 0:
                    m2 = compute_metrics(sub_df['equity'], trades_df=sub['trades'], n_days=len(sub_df))
                    regime_results[factor] = m2
            except Exception as e:
                print(f"  [SKIP] 归因 {factor} 在 {regime_name}: {e}")

        results[regime_name] = regime_results

    return results


def print_regime_report(regime_results: dict, bh_metrics: dict):
    """打印市场环境分段归因报告"""
    if not regime_results:
        return

    print(f"\n{'═' * 60}")
    print(f"  市场环境分段归因分析")
    print(f"{'═' * 60}")

    # 表格头
    regime_names = list(regime_results.keys())
    cols = len(regime_names)
    regime_labels = [f"{rn}" for rn in regime_names]
    regime_dates = [f"{REGIME_SLICES[rn][0]}~{REGIME_SLICES[rn][1]}" for rn in regime_names]

    print(f"  {'':26s}", end="")
    for lbl in regime_labels:
        print(f"  {lbl:>18s}", end="")
    print()
    print(f"  {'':26s}", end="")
    for d in regime_dates:
        print(f"  {d:>18s}", end="")
    print()
    print(f"  {'─' * (26 + cols * 20)}")

    # 全模型 vs 持有
    def regime_val(name, key, fmt='pct'):
        row_data = []
        for rn in regime_names:
            rm = regime_results[rn]
            if name == '持有不动':
                row_data.append('   —    ')
            elif name == '全模型':
                fm = rm.get('full', {})
                v = fm.get(key, 0)
                row_data.append(format_pct(v) if fmt == 'pct' else str(round(v, 2)))
            else:
                fm = rm.get(name, {})
                v = fm.get(key, 0)
                full_v = rm.get('full', {}).get(key, 0)
                diff = v - full_v
                row_data.append(f"{format_pct(v)} ({format_val(diff,1)}%)" if fmt == 'pct' else f"{v:.2f} ({diff:+.2f})")
        return row_data

    # 输出全模型
    for key, label, fmt in [('annual_return', '全模型年化', 'pct'), ('sharpe', '全模型夏普', 'val'), ('max_drawdown', '全模型回撤', 'pct')]:
        vals = regime_val('全模型', key, fmt)
        line = "  ".join(f"{v:>18s}" if isinstance(v, str) else f"{v:>18.2%}" for v in vals)
        print(f"  {label:<24s}  {line}")

    print(f"  {'─' * (26 + cols * 20)}")
    print(f"  {'移除因子后年化收益变化':<24s}")

    for factor in ['momentum', 'trend', 'volume_price', 'rsrs', 'relative_strength']:
        f_label = f"  - {factor}"
        print(f"  {f_label:<24s}", end="")
        for rn in regime_names:
            rm = regime_results[rn]
            fm = rm.get(factor)
            full_m = rm.get('full', {})
            if fm and full_m:
                f_ret = fm.get('annual_return', 0)
                full_ret = full_m.get('annual_return', 0)
                diff = f_ret - full_ret
                label = f"{format_pct(f_ret)} ({diff:+.1f}%)"
            else:
                label = "     —    "
            print(f"  {label:>18s}", end="")
        print()

    print(f"{'═' * 60}\n")


def run_factor_attribution(
    features_df: pd.DataFrame,
    config: dict,
    start: str,
    end: str,
    cost_rate: float,
) -> dict:
    """逐个零化因子权重，观察绩效变化"""
    factors = list(config['weights'].keys())

    print("\n  ── 因子归因分析 ──")

    # 全模型
    base = run_backtest(features_df, config, start, end, cost_rate)
    base_df = base['equity_curve']
    results = {}

    if len(base_df) == 0:
        print("  [WARN] 全模型回测无数据，跳过归因")
        return {}

    base_metrics = compute_metrics(base_df['equity'], trades_df=base['trades'], n_days=len(base_df))
    results['full'] = base_metrics

    for factor in factors:
        # 调整权重：移除因子后，剩余权重归一化
        adj = {k: v for k, v in config['weights'].items() if k != factor}
        remaining = sum(adj.values())
        if remaining > 0:
            adj = {k: v / remaining for k, v in adj.items()}

        config_copy = copy.deepcopy(config)
        config_copy['weights'] = adj

        # 重算 total_score
        feat_copy = features_df.copy()
        feat_copy['total_score'] = sum(
            feat_copy[f].fillna(0) * adj.get(f, 0) for f in factors if f in adj
        )

        try:
            sub = run_backtest(feat_copy, config_copy, start, end, cost_rate)
            sub_df = sub['equity_curve']
            if len(sub_df) > 0:
                m = compute_metrics(sub_df['equity'], trades_df=sub['trades'], n_days=len(sub_df))
                results[factor] = m
            else:
                results[factor] = None
        except Exception as e:
            print(f"  [ERR] 归因 {factor}: {e}")
            results[factor] = None

    return results


# ── 输出 ──────────────────────────────────────────────


def format_pct(v: float) -> str:
    sign = '+' if v > 0 else ''
    return f"{sign}{v:.1f}%"


def format_val(v: float, decimals: int = 2) -> str:
    sign = '+' if v > 0 else ''
    return f"{sign}{v:.{decimals}f}"


def pad(v, w=6):
    """右对齐固定宽度"""
    s = str(v)
    return s.rjust(w)


def print_report(
    metrics: dict,
    bh_metrics: dict,
    ma_metrics: dict,
    yearly: dict,
    start: str,
    end: str,
    n_days: int,
    cost_rate: float,
    factor_attr: Optional[dict] = None,
):
    """打印控制台报告"""
    print(f"\n{'═' * 55}")
    print(f"  588000 日线择时回测报告")
    print(f"  区间: {start} ~ {end}  ({n_days} 个交易日)")
    print(f"  费率: 单边 {cost_rate*10000:.1f}‱")
    print(f"{'═' * 55}")

    # 表头
    m = metrics
    bh = bh_metrics if bh_metrics else {}
    ma_val = ma_metrics if ma_metrics else {}

    def row(label, key, fmt='pct'):
        val_dec = 4 if key in ('sharpe', 'sortino', 'omega', 'calmar', 'profit_loss_ratio') else 2
        sv = format_pct(m.get(key, 0)) if fmt == 'pct' else format_val(m.get(key, 0), val_dec)
        bhv = format_pct(bh.get(key, 0)) if fmt == 'pct' else format_val(bh.get(key, 0), val_dec)
        mav = format_pct(ma_val.get(key, 0)) if fmt == 'pct' else format_val(ma_val.get(key, 0), val_dec)
        if key in ('win_rate', 'profit_loss_ratio', 'trade_count', 'avg_hold_days', 'max_dd_duration'):
            bhv = mav = '   — '
        # 超额 = 策略 - 持有
        exc = m.get(key, 0) - bh.get(key, 0) if bh and fmt == 'pct' else 0
        if key in ('sharpe', 'sortino', 'omega', 'calmar'):
            exc = m.get(key, 0) - bh.get(key, 0)
        exc_s = format_pct(exc) if fmt == 'pct' else format_val(exc, val_dec)
        if key in ('win_rate', 'profit_loss_ratio', 'trade_count', 'avg_hold_days', 'max_dd_duration'):
            exc_s = '   — '
        print(f"  {label:<14s} {sv:>8s}  {bhv:>8s}  {mav:>8s}  {exc_s:>8s}")

    print(f"  {'─' * 55}")
    row('年化收益率', 'annual_return')
    row('年化波动率', 'annual_vol')
    row('夏普比率', 'sharpe', 'val')
    row('Sortino比率', 'sortino', 'val')
    row('Omega比率', 'omega', 'val')
    row('最大回撤', 'max_drawdown')
    row('卡玛比率', 'calmar', 'val')
    row('回撤持续期(d)', 'max_dd_duration', 'val')
    print(f"  {'─' * 35}")
    row('胜率', 'win_rate')
    row('盈亏比', 'profit_loss_ratio', 'val')
    row('交易次数', 'trade_count', 'val')
    row('平均持仓(d)', 'avg_hold_days', 'val')
    print(f"{'─' * 55}")

    # 分段绩效
    if yearly:
        print(f"\n  分段表现:")
        print(f"  {'─' * 40}")
        for y, ym in sorted(yearly.items(), reverse=True):
            y_note = ""
            if ym.get('trade_count', 0) < 3:
                y_note = "  ⚠ 样本少"
            print(f"  {y}:  {format_pct(ym.get('annual_return', 0)):>7s}  "
                  f"夏普 {ym.get('sharpe', 0):>5.2f}  "
                  f"回撤 {ym.get('max_drawdown', 0):>5.1f}%  "
                  f"交易 {ym.get('trade_count', 0)} 次{y_note}")

    # 因子归因
    if factor_attr and len(factor_attr) > 1:
        print(f"\n  因子归因分析:")
        print(f"  {'─' * 55}")
        full_ret = factor_attr.get('full', {}).get('annual_return', 0)
        full_sharpe = factor_attr.get('full', {}).get('sharpe', 0)
        full_dd = factor_attr.get('full', {}).get('max_drawdown', 0)
        print(f"  {'移除因子':<14s} {'年化收益':>8s}  {'夏普':>6s}  {'回撤':>7s}  {'收益差':>8s}")
        print(f"  {'─' * 50}")
        print(f"  {'全模型(6因子)':<14s} {format_pct(full_ret):>8s}  {full_sharpe:>6.2f}  {full_dd:>6.1f}%  {'—':>8s}")

        for fname in ['momentum', 'trend', 'volume_price', 'rsrs', 'relative_strength']:
            fm = factor_attr.get(fname)
            if fm is None:
                continue
            f_ret = fm.get('annual_return', 0)
            f_sharpe = fm.get('sharpe', 0)
            f_dd = fm.get('max_drawdown', 0)
            diff = f_ret - full_ret
            label = f"- {fname}"
            if abs(diff) < 0.3:
                label += "  "
            print(f"  {label:<14s} {format_pct(f_ret):>8s}  {f_sharpe:>6.2f}  {f_dd:>6.1f}%  {format_val(diff, 1):>7s}%")

    print(f"\n{'═' * 55}\n")


def save_outputs(result: dict, bh_equity: pd.DataFrame, output_dir: str, metrics: Optional[dict] = None):
    """保存 CSV + metrics.json"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades = result.get('trades')
    equity = result.get('equity_curve')

    if trades is not None and len(trades) > 0:
        trades.to_csv(out / 'trades.csv', index=False)
        print(f"  [SAVE] 交易明细 → {out / 'trades.csv'} ({len(trades)} 条)")

    if equity is not None and len(equity) > 0:
        eq = equity.copy()
        if bh_equity is not None and len(bh_equity) > 0:
            eq = eq.merge(bh_equity, on='date', how='left')
        eq.to_csv(out / 'equity_curve.csv', index=False)
        print(f"  [SAVE] 权益曲线 → {out / 'equity_curve.csv'} ({len(eq)} 条)")

    # 绩效指标持久化（dashboard / gen_data 消费）
    if metrics:
        m = {k: (None if v != v else v) for k, v in metrics.items()}  # NaN → None
        metrics_path = out / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"  [SAVE] 绩效指标 → {metrics_path}")


# ── 主入口 ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='588000 日线择时回测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python backtest.py                                     # 全量回测
  python backtest.py --start 2023-06-01 --end 2025-12-31 # 指定区间
  python backtest.py --cost 0.001                        # 单边万10费率
  python backtest.py --factor-attribution                # 因子归因
  python backtest.py --regime-analysis                  # 分段归因（下跌/牛市）
  python backtest.py --output ./bt_results               # 自定义输出目录
        """
    )
    parser.add_argument('--start', default='2023-01-01', help='回测起始 (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='回测结束 (YYYY-MM-DD)，默认今天')
    parser.add_argument('--cost', type=float, default=0.00055,
                        help='单边交易费率（含佣金+滑点），默认万5.5')
    parser.add_argument('--output', default=None, help='输出目录（默认 data/588000/backtest/）')
    parser.add_argument('--factor-attribution', action='store_true',
                        help='同时跑因子归因分析')
    parser.add_argument('--regime-analysis', action='store_true',
                        help='分段因子归因（下跌段/牛市段分别分析）')
    parser.add_argument('--strategy', default=None, help='策略配置文件名（strategies/*.yaml）')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    args = parser.parse_args()

    # ── 加载 ──
    config = load_config()

    # 策略配置覆盖
    if args.strategy:
        strategy_path = PROJECT_ROOT / 'strategies' / f'{args.strategy}.yaml'
        if strategy_path.exists():
            with open(strategy_path, encoding='utf-8') as f:
                sc = yaml.safe_load(f)
            sc.get('weights', {}) and config['weights'].update(sc['weights'])
            sc.get('thresholds', {}) and config['thresholds'].update(sc['thresholds'])
            sc.get('adaptive_thresholds', {}) and config['adaptive_thresholds'].update(sc.get('adaptive_thresholds', {}))
            sc.get('weekly_modifier', {}) and config['weekly_modifier'].update(sc.get('weekly_modifier', {}))
            print(f"  [STRATEGY] {args.strategy}")
        else:
            print(f"  [WARN] 策略文件不存在: {strategy_path}，使用默认配置")

    symbol = config['symbol']
    end = args.end or date.today().strftime('%Y-%m-%d')
    output_dir = args.output or str(Path(PROJECT_ROOT) / config['data_dir'] / symbol / 'backtest')

    print(f"\n🔧 trade-pulse 回测框架")
    print(f"{'=' * 45}")

    # 读特征缓存
    print("\n📂 加载特征数据...")
    features_df = load_features_df(symbol)
    print(f"  [OK] 特征缓存: {len(features_df)} 条 "
          f"({features_df['date'].min().date()} ~ {features_df['date'].max().date()})")

    # ── 主回测 ──
    print("\n📊 运行多因子择时回测...")
    result = run_backtest(features_df, config, args.start, end, args.cost)
    equity_df = result['equity_curve']
    trades_df = result['trades']

    n_days = len(equity_df)
    metrics = compute_metrics(equity_df['equity'] if len(equity_df) > 0 else pd.Series(),
                               trades_df=trades_df, n_days=n_days)
    yearly = extract_yearly(equity_df, trades_df) if len(equity_df) > 0 else {}

    # ── 基准回测 ──
    print("\n📈 运行持有不动基准...")
    bh_df = run_buy_hold(features_df, args.start, end)
    bh_metrics = compute_metrics(bh_df['benchmark_equity'] if len(bh_df) > 0 else pd.Series(),
                                  n_days=len(bh_df))

    print("\n📉 运行双均线交叉基准(MA5/20)...")
    ma_result = run_ma_crossover(features_df, args.start, end, args.cost)
    ma_equity = ma_result['equity_curve']
    ma_metrics = compute_metrics(ma_equity['equity'] if len(ma_equity) > 0 else pd.Series(),
                                  trades_df=ma_result['trades'],
                                  n_days=len(ma_equity))

    # ── 因子归因 ──
    factor_attr = None
    if args.factor_attribution:
        print("\n🔬 运行因子归因分析...")
        factor_attr = run_factor_attribution(features_df, config, args.start, end, args.cost)

    # ── 市场环境分段归因 ──
    regime_results = None
    if args.regime_analysis:
        print("\n🌍 运行市场环境分段归因...")
        regime_results = run_regime_analysis(features_df, config, args.start, end, args.cost)

    # ── 报告 ──
    print_report(metrics, bh_metrics, ma_metrics, yearly,
                 args.start, end, n_days, args.cost, factor_attr)

    if regime_results:
        print_regime_report(regime_results, bh_metrics)

    # ── 保存 ──
    save_outputs(result, bh_df if len(bh_df) > 0 else None, output_dir, metrics=metrics)

    print(f"  [DONE] 回测完成。最终净值: {result['final_value']:.6f}")
    if len(trades_df) > 0:
        print(f"         交易次数: {len(trades_df[trades_df['action'].isin(['买入','加仓'])])}")
        print(f"         详细记录: {output_dir}/trades.csv")


if __name__ == '__main__':
    main()
