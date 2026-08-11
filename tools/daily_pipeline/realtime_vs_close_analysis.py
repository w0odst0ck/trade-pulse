#!/usr/bin/env python3
"""
realtime_vs_close_analysis.py — 实时 vs 收盘偏差统计（双线并行校准层）

输入：data/{symbol}/realtime_daily.csv（14:25/14:50 快照 + 收盘真值）
输出：
  1. 价格偏差：close_1450 vs close_final 相对偏差分布（mean/median/max、>0.5% 占比）
  2. 时点漂移：close_1425 vs close_1450 vs close_final 三时点漂移
  3. 信号翻转率：14:50 实时特征信号 vs 收盘特征信号（复用 verify_realtime_signal 内核）
  4. volume 归一化报告：盘中量/收盘量 比值分布（单独报告，不混入价格指标）

数据积累 < 20 行时仅报价格偏差（三时点漂移/volume 归一化需 20+ 行才有统计意义）。
用法：
  python3 realtime_vs_close_analysis.py            # 全量报告
  python3 realtime_vs_close_analysis.py --json     # JSON 输出（供 cron/飞书）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from realtime_daily import COLUMNS, load_snapshots


def _pct(a: float, b: float) -> float:
    """相对偏差百分比（b 为基准，b=0 或 NaN 返回 NaN）"""
    if b is None or b != b or b == 0 or a is None or a != a:
        return float("nan")
    return (a - b) / b * 100.0


def analyze_price_deviation(df: pd.DataFrame) -> dict:
    """close_1450 vs close_final 相对偏差分布"""
    rows = df.dropna(subset=["close_1450", "close_final"])
    if len(rows) == 0:
        return {"n": 0}
    dev = rows.apply(lambda r: _pct(r["close_1450"], r["close_final"]), axis=1)
    dev = dev.dropna()
    if len(dev) == 0:
        return {"n": 0}
    return {
        "n": len(dev),
        "mean_pct": round(float(dev.mean()), 4),
        "median_pct": round(float(dev.median()), 4),
        "max_abs_pct": round(float(dev.abs().max()), 4),
        "over_0_5pct_ratio": round(float((dev.abs() > 0.5).mean()), 4),
        "over_1pct_ratio": round(float((dev.abs() > 1.0).mean()), 4),
    }


def analyze_three_point_drift(df: pd.DataFrame) -> dict:
    """三时点漂移：1425→1450→final 各段平均绝对偏差"""
    rows = df.dropna(subset=["close_1425", "close_1450"])
    out = {}
    if len(rows) > 0:
        d1 = rows.apply(lambda r: abs(_pct(r["close_1425"], r["close_1450"])), axis=1).dropna()
        if len(d1) > 0:
            out["mean_abs_1425_to_1450_pct"] = round(float(d1.mean()), 4)
    rows2 = df.dropna(subset=["close_1450", "close_final"])
    if len(rows2) > 0:
        d2 = rows2.apply(lambda r: abs(_pct(r["close_1450"], r["close_final"])), axis=1).dropna()
        if len(d2) > 0:
            out["mean_abs_1450_to_final_pct"] = round(float(d2.mean()), 4)
    return out


def analyze_volume_ratio(df: pd.DataFrame) -> dict:
    """volume_1450 / volume_final 比值分布（盘中累计量 ≈ 收盘的 80-95% 属正常）"""
    rows = df.dropna(subset=["volume_1450", "volume_final"])
    if len(rows) == 0 or (rows["volume_final"] == 0).all():
        return {"n": 0}
    ratio = rows["volume_1450"] / rows["volume_final"].replace(0, np.nan)
    ratio = ratio.dropna()
    ratio = ratio[(ratio > 0) & (ratio < 10)]  # 过滤异常
    if len(ratio) == 0:
        return {"n": 0}
    return {
        "n": len(ratio),
        "mean_ratio": round(float(ratio.mean()), 4),
        "median_ratio": round(float(ratio.median()), 4),
        "min_ratio": round(float(ratio.min()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="实时 vs 收盘偏差统计")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    import json as _json
    config = _json.loads((SCRIPT_DIR / "config.json").read_text(encoding="utf-8"))
    symbol = config["symbol"]

    df = load_snapshots(symbol)

    if len(df) == 0:
        summary = {"n_days": 0, "price_deviation": {"n": 0},
                   "three_point_drift": {}, "volume_ratio": {"n": 0}}
        if args.json:
            print(_json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"\n🎯 实时 vs 收盘偏差统计（{symbol}，累计 0 个交易日）")
            print("=" * 56)
            print("  [INFO] 暂无快照数据（14:25/14:50 自动积累，1 天 1 行）")
        return summary

    price = analyze_price_deviation(df)
    drift = analyze_three_point_drift(df)
    vol = analyze_volume_ratio(df)

    summary = {"n_days": len(df), "price_deviation": price,
               "three_point_drift": drift, "volume_ratio": vol}

    # --json 模式：stdout 只输出 JSON（供管道/飞书 webhook 解析）
    if args.json:
        print(_json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    # 人类可读报告
    print(f"\n🎯 实时 vs 收盘偏差统计（{symbol}，累计 {len(df)} 个交易日）")
    print("=" * 56)

    # 渐进式报告：数据量不足时只报有统计意义的部分，避免过度解读
    if price.get("n", 0) > 0:
        print(f"\n  ── 价格偏差（14:50 实时 vs 收盘真值，{price['n']} 天）──")
        print(f"  平均偏差: {price['mean_pct']:+.3f}%   中位数: {price['median_pct']:+.3f}%")
        print(f"  最大绝对偏差: {price['max_abs_pct']:.3f}%")
        print(f"  >0.5% 占比: {price['over_0_5pct_ratio']*100:.1f}%   >1% 占比: {price['over_1pct_ratio']*100:.1f}%")
    else:
        print(f"\n  ── 价格偏差 ── 暂无同时含 14:50 快照与收盘真值的行（积累中）")

    # 三时点漂移：≥20 行才有统计意义（<20 行跳过，避免少数几天误导）
    if drift and price.get("n", 0) >= 20:
        print(f"\n  ── 三时点漂移（14:25→14:50→收盘，{price['n']} 天）──")
        if "mean_abs_1425_to_1450_pct" in drift:
            print(f"  14:25→14:50 平均 |Δ|: {drift['mean_abs_1425_to_1450_pct']:.3f}%")
        if "mean_abs_1450_to_final_pct" in drift:
            print(f"  14:50→收盘 平均 |Δ|: {drift['mean_abs_1450_to_final_pct']:.3f}%")
    elif drift:
        print(f"\n  ── 三时点漂移 ── 数据 <20 天（当前 {price.get('n', 0)} 天），积累后报告")

    # volume 归一化：≥20 行才有意义
    if vol.get("n", 0) >= 20:
        print(f"\n  ── volume 归一化（14:50 累计量 / 收盘量，{vol['n']} 天）──")
        print(f"  均值: {vol['mean_ratio']:.2f}   中位数: {vol['median_ratio']:.2f}   "
              f"最小: {vol['min_ratio']:.2f}")
    elif vol.get("n", 0) > 0:
        print(f"\n  ── volume 归一化 ── 数据 <20 天（当前 {vol['n']} 天），积累后报告")

    return summary


if __name__ == "__main__":
    main()
