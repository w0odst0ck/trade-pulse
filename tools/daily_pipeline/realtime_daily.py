#!/usr/bin/env python3
"""
realtime_daily.py — 实时快照积累（方案 C：双线并行数据资产层）

每交易日三个时点写入 data/{symbol}/realtime_daily.csv 同一行：
  - 14:25 盘中预览 → close_1425 / volume_1425（daily_signal.sh 调用）
  - 14:50 尾盘确认 → close_1450 / volume_1450（daily_confirm.sh 调用）
  - 15:30 收盘后   → open_final / high_final / low_final / close_final / volume_final
    （daily_fetch.sh 调用，回填真值）

设计要点：
  - 列级幂等 upsert：三个时点各自独立写列，同日重复调用覆盖对应列（不互相覆盖）
  - 写入前 sanity check：价格 > 0 / volume > 0，脏数据宁可缺行不写脏行
  - 实时不可用当天 → 对应列留空（统计时按可用行算，不假装有数据）
  - 独立文件，绝不写入 daily.csv / features_cache（回测口径保持纯净）

CSV 列：
  date, open_1425, high_1425, low_1425, close_1425, volume_1425,
  open_1450, high_1450, low_1450, close_1450, volume_1450,
  open_final, high_final, low_final, close_final, volume_final

用法：
  python3 realtime_daily.py --snapshot-1425        # 14:25 预览时点（传 --bar 或自动拉）
  python3 realtime_daily.py --snapshot-1450        # 14:50 确认时点
  python3 realtime_daily.py --backfill-final       # 收盘后回填真值（从 daily.csv）
  python3 realtime_daily.py --bar '{"close":...}'  # 注入 bar dict（测试/调试）
  python3 realtime_daily.py --tail 5               # 查看最近几行
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from realtime_quote import fetch_realtime_bar, sanity_check

COLUMNS = [
    "date",
    "open_1425", "high_1425", "low_1425", "close_1425", "volume_1425",
    "open_1450", "high_1450", "low_1450", "close_1450", "volume_1450",
    "open_final", "high_final", "low_final", "close_final", "volume_final",
]


def snapshot_path(symbol: str) -> Path:
    config = json.loads((SCRIPT_DIR / "config.json").read_text(encoding="utf-8"))
    return PROJECT_ROOT / config["data_dir"] / symbol / "realtime_daily.csv"


def load_snapshots(symbol: str) -> pd.DataFrame:
    p = snapshot_path(symbol)
    if p.exists():
        df = pd.read_csv(p)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = float("nan")
        return df
    return pd.DataFrame(columns=COLUMNS)


def _atomic_save(df: pd.DataFrame, symbol: str) -> None:
    p = snapshot_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def _bar_to_row(bar: dict, prefix: str) -> dict:
    """bar dict → 带前缀的列 dict（缺失字段置 NaN）"""
    return {
        f"open_{prefix}": bar.get("open", float("nan")),
        f"high_{prefix}": bar.get("high", float("nan")),
        f"low_{prefix}": bar.get("low", float("nan")),
        f"close_{prefix}": bar.get("close", float("nan")),
        f"volume_{prefix}": bar.get("volume", float("nan")),
    }


def write_snapshot(symbol: str, prefix: str, bar: Optional[Dict] = None,
                   prev_close: Optional[float] = None) -> bool:
    """写入某时点快照（列级 upsert）

    prefix: '1425' / '1450'
    bar:    None 时自动拉实时（fetch_realtime_bar 双源互备）
    返回 True 表示成功写入（含覆盖），False 表示不可用/脏数据（留空）。
    """
    if bar is None:
        bar = fetch_realtime_bar(symbol, prev_close=prev_close)
    if bar is None:
        print(f"  [SNAP] {prefix} 实时不可用，{symbol} 该列留空")
        return False

    ok, issues = sanity_check(bar, prev_close)
    if not ok:
        print(f"  [SNAP] {prefix} 数据不合法（{'; '.join(issues)}），不写入")
        return False

    row = _bar_to_row(bar, prefix)
    today = str(bar.get("date", date.today()))[:10]

    df = load_snapshots(symbol)
    if len(df) > 0 and str(df["date"].iloc[-1]) == today:
        idx = df.index[-1]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        new_row = {c: float("nan") for c in COLUMNS}
        new_row["date"] = today
        new_row.update(row)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    _atomic_save(df, symbol)
    print(f"  [SNAP] {symbol} {prefix} 快照已写入（{today} close={row[f'close_{prefix}']:.4f}）")
    return True


def backfill_final(symbol: str) -> bool:
    """收盘后回填真值：从 daily.csv 取当日收盘 OHLCV 写入 final 列

    返回 True 表示当日数据已在 daily.csv 且已回填；False 表示盘后未发布（留待下次）。
    """
    config = json.loads((SCRIPT_DIR / "config.json").read_text(encoding="utf-8"))
    daily_path = PROJECT_ROOT / config["data_dir"] / symbol / "daily.csv"
    if not daily_path.exists():
        print("  [SNAP] daily.csv 不存在，无法回填")
        return False

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    today = date.today().strftime("%Y-%m-%d")
    today_rows = daily[daily["date"] == pd.Timestamp(today)]
    if len(today_rows) == 0:
        print(f"  [SNAP] {today} 收盘数据未发布（daily.csv 无当日行），final 留空待下次")
        return False

    row = today_rows.iloc[-1]
    final = {
        "open_final": row["open"], "high_final": row["high"],
        "low_final": row["low"], "close_final": row["close"],
        "volume_final": row["volume"],
    }

    df = load_snapshots(symbol)
    if len(df) > 0 and str(df["date"].iloc[-1]) == today:
        idx = df.index[-1]
        for k, v in final.items():
            df.at[idx, k] = v
        _atomic_save(df, symbol)
        print(f"  [SNAP] {symbol} final 已回填（{today} close_final={row['close']:.4f}）")
        return True

    # 当日无快照行（14:25/14:50 都没成功）→ 补一行只有 final 的数据（partial）
    new_row = {c: float("nan") for c in COLUMNS}
    new_row["date"] = today
    new_row.update(final)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _atomic_save(df, symbol)
    print(f"  [SNAP] {symbol} final 已回填（当日无实时快照，仅收盘真值行）")
    return True


def print_tail(symbol: str, n: int = 5) -> None:
    df = load_snapshots(symbol)
    if len(df) == 0:
        print("  [SNAP] 暂无快照数据")
        return
    print(f"  realtime_daily.csv 最近 {min(n, len(df))} 行（{symbol}）:")
    show_cols = ["date", "close_1425", "close_1450", "close_final",
                 "volume_1425", "volume_1450", "volume_final"]
    print(df[show_cols].tail(n).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="实时快照积累（双线并行数据层）")
    parser.add_argument("--snapshot-1425", action="store_true", help="写 14:25 预览时点快照")
    parser.add_argument("--snapshot-1450", action="store_true", help="写 14:50 确认时点快照")
    parser.add_argument("--backfill-final", action="store_true", help="收盘后回填真值")
    parser.add_argument("--bar", default=None, help='注入 bar dict JSON（测试/调试用）')
    parser.add_argument("--symbol", default=None, help='标的（默认 config.symbol）')
    parser.add_argument("--tail", type=int, default=None, help='查看最近 N 行')
    args = parser.parse_args()

    config = json.loads((SCRIPT_DIR / "config.json").read_text(encoding="utf-8"))
    symbol = args.symbol or config["symbol"]

    if args.tail is not None:
        print_tail(symbol, args.tail)
        return

    bar = json.loads(args.bar) if args.bar else None
    if args.snapshot_1425:
        write_snapshot(symbol, "1425", bar=bar)
    elif args.snapshot_1450:
        write_snapshot(symbol, "1450", bar=bar)
    elif args.backfill_final:
        backfill_final(symbol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
