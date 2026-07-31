#!/usr/bin/env python3
"""
trade_log.py — 交易记录管理

功能：
  记录每笔交易（决策留痕），提供 CSV 存储 + 统计 + 追加接口。
  实盘前开始用，每笔操作记一行。

用法：
  python trade_log.py --list                    # 查看全部记录
  python trade_log.py --stats                   # 统计（胜率/盈亏比/月度）
  python trade_log.py --add --date 2026-08-01 --symbol 588000 \
      --action 买入 --score -0.3 --position 40 --entry 1.85 \
      --note "信号+行业判断"                    # 追加一笔
  python trade_log.py --add-simple "买入 588000 40% @1.85"  # 简单模式

CSV: data/trade_log.csv
  列: date,symbol,action,score,position_pct,entry_price,exit_price,pnl_pct,note
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "trade_log.csv"

COLUMNS = ["date", "symbol", "action", "score", "position_pct",
           "entry_price", "exit_price", "pnl_pct", "note"]


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, dtype={"date": str})
    return pd.DataFrame(columns=COLUMNS)


def save_log(df: pd.DataFrame):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"  [OK] {LOG_PATH} ({len(df)} 条)")


def add_entry(entry: dict):
    df = load_log()
    row = {k: entry.get(k, "") for k in COLUMNS}
    row["date"] = row["date"] or date.today().isoformat()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_log(df)
    print(f"  [LOG] {row['date']} {row['symbol']} {row['action']} "
          f"仓位{row['position_pct']}% @{row['entry_price']}")


def show_list(df: pd.DataFrame):
    if len(df) == 0:
        print("  [EMPTY] 暂无交易记录")
        return
    print(f"\n{'日期':<12}{'标的':<8}{'操作':<6}{'分数':>7}{'仓位%':>6}{'入场':>8}{'出场':>8}{'盈亏%':>7}  备注")
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"{r['date']:<12}{r['symbol']:<8}{r['action']:<6}"
              f"{r['score']:>7.2f}{r['position_pct']:>6.0f}"
              f"{r['entry_price']:>8.2f}{r['exit_price']:>8.2f}"
              f"{r['pnl_pct']:>7.2f}  {r['note']}")


def show_stats(df: pd.DataFrame):
    if len(df) == 0:
        print("  [EMPTY] 暂无交易记录")
        return
    # 已平仓的完整交易（有 exit）
    closed = df[df["exit_price"].notna() & (df["exit_price"] != "")]
    if len(closed) > 0:
        closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
        wins = closed[closed["pnl_pct"] > 0]
        losses = closed[closed["pnl_pct"] < 0]
        win_rate = len(wins) / len(closed) * 100 if len(closed) else 0
        avg_win = wins["pnl_pct"].mean() if len(wins) else 0
        avg_loss = losses["pnl_pct"].mean() if len(losses) else 0
        print(f"\n  交易统计（{len(closed)} 笔已平仓）:")
        print(f"    胜率: {win_rate:.1f}%  ({len(wins)} 胜 / {len(losses)} 负)")
        print(f"    平均盈利: +{avg_win:.2f}% | 平均亏损: {avg_loss:.2f}%")
        if avg_loss != 0:
            print(f"    盈亏比: {abs(avg_win / avg_loss):.2f}")
    else:
        print("\n  暂无已平仓交易")

    # 按标的
    if len(df) > 0:
        print("\n  按标的:")
        for sym, grp in df.groupby("symbol"):
            print(f"    {sym}: {len(grp)} 笔操作")


def main():
    parser = argparse.ArgumentParser(description="交易记录管理")
    parser.add_argument("--list", action="store_true", help="查看记录")
    parser.add_argument("--stats", action="store_true", help="统计")
    parser.add_argument("--add", action="store_true", help="追加记录")
    parser.add_argument("--date", default="")
    parser.add_argument("--symbol", default="588000")
    parser.add_argument("--action", default="")
    parser.add_argument("--score", type=float, default=0.0)
    parser.add_argument("--position", type=float, default=0.0, help="仓位%%")
    parser.add_argument("--entry", type=float, default=0.0)
    parser.add_argument("--exit", type=float, default=0.0)
    parser.add_argument("--pnl", type=float, default=0.0)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    if args.add:
        entry = {
            "date": args.date, "symbol": args.symbol, "action": args.action,
            "score": args.score, "position_pct": args.position,
            "entry_price": args.entry, "exit_price": args.exit,
            "pnl_pct": args.pnl, "note": args.note,
        }
        add_entry(entry)
    elif args.stats:
        show_stats(load_log())
    else:
        show_list(load_log())


if __name__ == "__main__":
    main()
