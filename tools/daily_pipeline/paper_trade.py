#!/usr/bin/env python3
"""
paper_trade.py — 纸面虚拟盘（paper trading）

【口径】与回测完全一致（引擎与 backtest.run_backtest 同构）：
  - 信号对齐：T 日特征 → T 日信号（signal_rules.decide 状态机）→ T+1 日收盘成交
  - 连续仓位 0~70%：复用 signal_rules 内部 calc_position（round(0.3+score*0.4,2)，封顶 0.7）
  - 单边成本 0.00055（佣金+滑点），只对调仓差额部分计费
  - 调仓阈值：目标仓位与当前仓位偏差 |gap_pct| > 5% 才调仓（与回测相同）
  - 初始资金 1.0（净值口径）、空仓起步
  - 每日净值 = cash + shares × 当日收盘价（T+1 收盘后口径；最后一天只有特征无成交）
  - 使用生产 config.json 原样运行（adaptive_thresholds.enabled=true，与实盘同口径），
    不手工禁用自适应阈值

【与实盘的关系】执行质量对照
  纸面盘假设「完美执行信号」：T+1 收盘价成交、零延迟、零人为偏差。
  未来实盘（手动执行）从同一信号日起记录实际成交价/成交日期；
  实盘净值曲线与纸面盘净值曲线的偏差 = 执行质量（滑点、延迟、遗漏信号）。

【运行方式】
  python3 paper_trade.py                      # 默认 --update：增量重放（无状态文件时自动全量）
  python3 paper_trade.py --full               # 全量重放（2023-01-01 起）+ 一致性自检
  python3 paper_trade.py --tail 10            # 只读打印最近 10 行 equity
  python3 paper_trade.py --update --tail 10   # 先增量重放再打印最近 10 行

【数据文件】（data/588000/paper/，每次更新原子写：临时文件 + rename）
  paper_state.json   引擎状态：decide 返回的 _new_state + last_date（最后信号日 T）
                     + 完整精度 cash/shares（保证增量续跑与全量结果完全一致）
  paper_equity.csv   date, equity, position_pct(%), close, shares, cash
  paper_trades.csv   date(成交日 T+1), action(买入/加仓/减仓/清仓), price, shares, amount
                     amount 为成交金额（不含费用）

【一致性自检】--full 完成后自动执行：
  用 backtest.run_backtest(features_df, 生产config, '2023-01-01', 特征末日, 0.00055)
  重跑一遍，逐行对比两条 equity 曲线：行数一致且全曲线最大差异 < 1e-6 才为 PASS。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR))
from signal_rules import decide
from backtest import run_backtest, compute_metrics, load_features_df, load_config

START_DATE = '2023-01-01'      # --full 全量重放起点（与需求一致）
COST_RATE = 0.00055            # 单边费率（万5.5，与回测默认一致）
GAP_THRESHOLD = 0.05           # 调仓阈值（与 backtest.run_backtest 一致）
INIT_EQUITY = 1.0              # 初始净值

STATE_FILENAME = 'paper_state.json'
EQUITY_FILENAME = 'paper_equity.csv'
TRADES_FILENAME = 'paper_trades.csv'


# ── 原子写 ──────────────────────────────────────────


def _atomic_write_csv(df: pd.DataFrame, path: Path):
    """临时文件 + rename，保证原子写"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ── 重放引擎（与 backtest.run_backtest 同构） ──


def _replay_engine(
    features_df: pd.DataFrame,
    config: dict,
    cost_rate: float,
    start: str,
    end: str,
    first_signal_idx: int = 0,
    init_state: dict = None,
    init_cash: float = INIT_EQUITY,
    init_shares: float = 0.0,
) -> dict:
    """逐日重放：T 日信号 → T+1 日收盘成交。

    数值逻辑与 backtest.run_backtest 完全相同（同引擎同数据 → 同净值），
    decide 的上下文参数同样传完整 features_df。
    first_signal_idx 支持增量续跑：全量=0；增量=过滤后 df 中 last_date 的下一行(1)。
    """
    if init_state is None:
        init_state = {'state': '空仓', 'waiting_days': 0, 'last_decision_date': None}

    mask = (
        (features_df['date'] >= pd.Timestamp(start))
        & (features_df['date'] <= pd.Timestamp(end))
    )
    df = features_df[mask].sort_values('date').reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"重放区间 {start} ~ {end} 数据不足 ({len(df)} 条)")

    n_days = len(df)
    state = dict(init_state)
    cash = init_cash
    holding_shares = init_shares

    daily = []   # equity 行（执行日 T+1）
    trades = []  # paper_trades 行

    for i in range(first_signal_idx, n_days - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        tomorrow_date = tomorrow['date']
        tomorrow_close = tomorrow['close']

        # ── 决策（与 backtest.run_backtest 同参调用） ──
        row_dict = today.to_dict()
        result = decide(
            row_dict, features_df,
            state_override=state, persist=False,
            config_override=config,
        )
        state = result['_new_state']  # 取出更新后的状态，用于下一轮

        # ── 交易执行：连续仓位调整（与回测同逻辑） ──
        current_holdings_value = holding_shares * tomorrow_close
        total_equity = current_holdings_value + cash

        if total_equity > 1e-6:
            target_pct = float(result['position'].rstrip('%')) / 100.0
            current_pct = current_holdings_value / total_equity
            gap_pct = target_pct - current_pct

            if abs(gap_pct) > GAP_THRESHOLD:
                gap_value = gap_pct * total_equity

                if gap_value > 0:  # 买入
                    cost = gap_value * cost_rate
                    total_needed = gap_value + cost
                    if total_needed <= cash:
                        delta_shares = gap_value / tomorrow_close
                        holding_shares += delta_shares
                        cash -= total_needed

                        is_first_buy = abs(current_pct) < 1e-6
                        act_label = '买入' if is_first_buy else '加仓'
                        trades.append({
                            'date': tomorrow_date,
                            'action': act_label,
                            'price': round(float(tomorrow_close), 6),
                            'shares': round(delta_shares, 6),
                            'amount': round(gap_value, 6),
                        })
                    # else: 现金不足 → 不动（与回测一致，不记录）

                else:  # 卖出
                    sell_value = -gap_value
                    cost = sell_value * cost_rate
                    if sell_value <= current_holdings_value:
                        sell_shares = sell_value / tomorrow_close
                        holding_shares -= sell_shares
                        cash += sell_value - cost

                        is_full_close = target_pct < 1e-6
                        act_label = '清仓' if is_full_close else '减仓'
                        trades.append({
                            'date': tomorrow_date,
                            'action': act_label,
                            'price': round(float(tomorrow_close), 6),
                            'shares': round(sell_shares, 6),
                            'amount': round(sell_value, 6),
                        })

        # ── 日末权益（T+1 收盘后） ──
        total_equity = cash + holding_shares * tomorrow_close
        daily.append(_equity_row(tomorrow_date, total_equity, holding_shares,
                                 tomorrow_close, cash))

    # 补上最后一笔权益（最后一天只有特征，没有交易）
    if n_days > 0:
        last_row = df.iloc[-1]
        last_equity = cash + holding_shares * last_row['close']
        daily.append(_equity_row(last_row['date'], last_equity, holding_shares,
                                 last_row['close'], cash))

    last_signal_date = df['date'].iloc[n_days - 2] if n_days - 2 >= first_signal_idx else None

    return {
        'daily': daily,
        'trades': trades,
        'state': state,
        'cash': cash,
        'shares': holding_shares,
        'last_signal_date': last_signal_date,
        'n_days': n_days,
    }


def _equity_row(date_ts, equity: float, shares: float, close: float, cash: float) -> dict:
    """equity 行：date, equity, position_pct(%), close, shares, cash"""
    pos = (shares * close / equity if equity > 0 else 0.0) * 100.0
    return {
        'date': date_ts,
        'equity': round(equity, 6),
        'position_pct': round(pos, 2),
        'close': round(float(close), 6),
        'shares': round(float(shares), 6),
        'cash': round(float(cash), 6),
    }


# ── 全量 / 增量 ──────────────────────────────────────


def _paper_dir(config: dict) -> Path:
    return PROJECT_ROOT / config['data_dir'] / config['symbol'] / 'paper'


def _state_path(paper_dir: Path) -> Path:
    return paper_dir / STATE_FILENAME


def _load_state(paper_dir: Path) -> dict:
    with open(_state_path(paper_dir), encoding='utf-8') as f:
        return json.load(f)


def replay_full(features_df: pd.DataFrame, config: dict) -> tuple:
    """全量重放：2023-01-01 起，空仓起步，重放至特征末日 + 一致性自检"""
    end = features_df['date'].max().strftime('%Y-%m-%d')
    print(f"\n📼 纸面盘全量重放（{START_DATE} 起，空仓起步）")

    out = _replay_engine(features_df, config, COST_RATE, START_DATE, end, first_signal_idx=0)
    _persist(_paper_dir(config), out, append=False)
    print(f"  [OK] 重放 {out['n_days']} 个交易日（信号日 {out['n_days'] - 1} 个）")

    equity_df = _read_equity(_paper_dir(config))
    ok = self_check(features_df, config, equity_df)
    print_summary(equity_df, _read_trades(_paper_dir(config)))
    return equity_df, ok


def replay_update(features_df: pd.DataFrame, config: dict) -> tuple:
    """增量重放：从 paper_state.json 的 last_date 续跑至特征末日"""
    paper_dir = _paper_dir(config)
    if not _state_path(paper_dir).exists():
        print("[WARN] 无 paper_state.json，自动转全量重放")
        return replay_full(features_df, config)

    saved = _load_state(paper_dir)
    last_date = str(saved['last_date'])
    last_ts = pd.Timestamp(last_date)

    # 定位最后信号日在完整特征中的位置
    matches = features_df.index[features_df['date'] == last_ts]
    if len(matches) == 0:
        raise ValueError(
            f"paper_state.json 的 last_date={last_date} 不在特征数据中"
            "（数据被重算/回滚？）。请用 --full 重建纸面盘。"
        )
    idx = int(matches[0])
    n_days = len(features_df)
    if idx > n_days - 2:
        raise ValueError(
            f"last_date={last_date} 已超出特征范围（特征末日 "
            f"{features_df['date'].max().date()}）。请用 --full 重建。"
        )

    new_signal_days = (n_days - 2) - idx  # 待处理信号日数量
    if new_signal_days <= 0:
        print(f"\n📼 纸面盘增量重放：已是最新（信号日截至 {last_date}），无新增交易日")
        return _read_equity(paper_dir), None

    end = features_df['date'].max().strftime('%Y-%m-%d')
    print(f"\n📼 纸面盘增量重放（从信号日 {last_date} 之后继续，新增 {new_signal_days} 个信号日）")
    # 过滤区间从 last_date 起：过滤后 df[0]=last_date（已处理），first_signal_idx=1 从下一行续跑
    out = _replay_engine(
        features_df, config, COST_RATE, last_date, end,
        first_signal_idx=1,
        init_state=saved['state'],
        init_cash=float(saved['cash']),
        init_shares=float(saved['shares']),
    )
    _persist(paper_dir, out, append=True)

    equity_df = _read_equity(paper_dir)
    print_summary(equity_df, _read_trades(paper_dir))
    return equity_df, None


def _fmt_date(ts) -> str:
    """Timestamp/日期 → 'YYYY-MM-DD' 字符串（state.json 可读性）"""
    return pd.Timestamp(ts).strftime('%Y-%m-%d') if ts is not None else None


def _persist(paper_dir: Path, out: dict, append: bool = False):
    """原子写出 equity.csv / trades.csv / state.json"""
    equity_path = paper_dir / EQUITY_FILENAME
    trades_path = paper_dir / TRADES_FILENAME

    new_equity = pd.DataFrame(out['daily'])
    new_trades = pd.DataFrame(out['trades'])

    if append:
        old_equity = _read_equity(paper_dir)
        # 旧 CSV 最后一行是旧特征末日的补行（无成交的收尾行），
        # 丢弃它，由新 daily 的最后一笔（执行日 + 新末日补行）取代，
        # 保证增量结果与全量重跑的行数与日期序列完全一致
        if len(old_equity) > 0:
            old_equity = old_equity.iloc[:-1]
        old_trades = _read_trades(paper_dir)
        new_equity = pd.concat([old_equity, new_equity], ignore_index=True)
        new_trades = pd.concat([old_trades, new_trades], ignore_index=True)

    _atomic_write_csv(new_equity, equity_path)
    _atomic_write_csv(new_trades, trades_path)

    state_obj = {
        'state': out['state'],                       # decide 返回的 _new_state
        'last_date': _fmt_date(out['last_signal_date']),  # 最后信号日（T）
        'cash': out['cash'],                         # 完整精度，保证增量与全量一致
        'shares': out['shares'],
        'equity_last_date': _fmt_date(new_equity['date'].iloc[-1]),
    }
    _atomic_write_json(state_obj, _state_path(paper_dir))
    print(f"  [SAVE] {equity_path} ({len(new_equity)} 行)")
    print(f"  [SAVE] {trades_path} ({len(new_trades)} 条)")
    print(f"  [SAVE] {_state_path(paper_dir)}")


def _read_equity(paper_dir: Path) -> pd.DataFrame:
    path = paper_dir / EQUITY_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=['date', 'equity', 'position_pct', 'close', 'shares', 'cash'])
    try:
        return pd.read_csv(path, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['date', 'equity', 'position_pct', 'close', 'shares', 'cash'])


def _read_trades(paper_dir: Path) -> pd.DataFrame:
    path = paper_dir / TRADES_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=['date', 'action', 'price', 'shares', 'amount'])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['date', 'action', 'price', 'shares', 'amount'])


# ── 一致性自检 ──────────────────────────────────────


def self_check(features_df: pd.DataFrame, config: dict, paper_equity: pd.DataFrame) -> bool:
    """与 backtest.run_backtest 逐行对比：行数一致且全曲线最大差异 < 1e-6 为 PASS"""
    end = features_df['date'].max().strftime('%Y-%m-%d')
    print(f"\n🔍 一致性自检：vs backtest.run_backtest({START_DATE} ~ {end}, 费率 {COST_RATE})")
    bt = run_backtest(features_df, config, START_DATE, end, COST_RATE)
    bt_equity = bt['equity_curve']

    paper_last = float(paper_equity['equity'].iloc[-1])
    bt_last = float(bt_equity['equity'].iloc[-1])
    diff = abs(paper_last - bt_last)

    full_diff = None
    pv = paper_equity['equity'].astype(float).values
    bv = bt_equity['equity'].astype(float).values
    if len(pv) == len(bv):
        full_diff = float(np.max(np.abs(pv - bv)))  # 全曲线逐行最大差异

    print(f"  paper 末端净值:    {paper_last:.9f}")
    print(f"  backtest 末端净值: {bt_last:.9f}")
    print(f"  末端差异: {diff:.3e}")
    if full_diff is None:
        # 行数不一致直接 FAIL，不能只靠末端净值对比
        print(f"  曲线行数不一致: paper {len(pv)} 行 vs backtest {len(bv)} 行 → 直接 FAIL")
        ok = False
    else:
        print(f"  全曲线最大差异: {full_diff:.3e}")
        # 行数一致时同时 enforce 末端差异与全曲线最大差异 < 1e-6 才算 PASS
        ok = diff < 1e-6 and full_diff < 1e-6
    print(f"  结果: {'PASS ✓ 重放与回测同口径' if ok else 'FAIL ✗ 重放逻辑与回测不一致，需修复'}")
    return ok


# ── 摘要 / tail ──────────────────────────────────────


def print_summary(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    if len(equity_df) == 0:
        print("  [WARN] 无权益数据")
        return
    last = equity_df.iloc[-1]
    m = compute_metrics(equity_df['equity'], trades_df=trades_df, n_days=len(equity_df))
    print(f"\n  ── 纸面盘摘要 ──")
    print(f"  最后日期: {pd.Timestamp(last['date']).date()}   当前仓位: {last['position_pct']:.2f}%   "
          f"净值: {last['equity']:.6f}")
    print(f"  累计年化: {m['annual_return']:+.2f}%   夏普: {m['sharpe']:.4f}   "
          f"最大回撤: {m['max_drawdown']:.2f}%")
    if len(trades_df) > 0:
        print(f"  交易记录: {len(trades_df)} 条（{TRADES_FILENAME}）")


def print_tail(equity_df: pd.DataFrame, n: int):
    if len(equity_df) == 0:
        print("  [WARN] 无权益数据，请先运行 --full / --update")
        return
    k = min(n, len(equity_df))
    print(f"\n  最近 {k} 行 equity（{EQUITY_FILENAME}）:")
    print(f"  {'date':<12} {'equity':>10} {'pos%':>7} {'close':>9} {'shares':>12} {'cash':>10}")
    for _, r in equity_df.tail(k).iterrows():
        print(f"  {str(pd.Timestamp(r['date']).date()):<12} {r['equity']:>10.6f} {r['position_pct']:>7.2f} "
              f"{r['close']:>9.4f} {r['shares']:>12.4f} {r['cash']:>10.4f}")


# ── CLI ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='588000 纸面虚拟盘：T 日信号 → T+1 收盘成交，与回测同口径',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 paper_trade.py                  # 增量重放（默认；无状态文件时自动全量）
  python3 paper_trade.py --full           # 全量重放（2023-01-01 起）+ 一致性自检
  python3 paper_trade.py --tail 10        # 只读打印最近 10 行 equity
  python3 paper_trade.py --update --tail 10
        """,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='全量重放（2023-01-01 起）并做一致性自检')
    mode.add_argument('--update', action='store_true', help='增量重放（默认，从 last_date 续跑）')
    parser.add_argument('--tail', type=int, default=None, metavar='N',
                        help='打印最近 N 行 equity（可与 --full/--update 组合，单独使用为只读）')
    args = parser.parse_args()

    config = load_config()
    features_df = load_features_df(config['symbol'])
    print(f"  [INFO] 特征缓存: {len(features_df)} 条 "
          f"({features_df['date'].min().date()} ~ {features_df['date'].max().date()})")

    paper_dir = _paper_dir(config)

    if args.full:
        equity_df, _ = replay_full(features_df, config)
    elif args.update or args.tail is None:
        # 默认增量；--tail 单独使用时跳过更新（只读）
        if not _state_path(paper_dir).exists():
            equity_df, _ = replay_full(features_df, config)  # 首次自动全量
        else:
            equity_df, _ = replay_update(features_df, config)
    else:
        equity_df = _read_equity(paper_dir)

    if args.tail is not None:
        print_tail(equity_df, args.tail)

    if args.full or args.update or args.tail is None:
        print("\n  [DONE] 纸面盘更新完成")


if __name__ == '__main__':
    main()
