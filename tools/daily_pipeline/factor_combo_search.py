#!/usr/bin/env python3
"""
factor_combo_search.py — 因子组合搜索

移除负贡献因子（trend/relative_strength）后，对比不同因子组合的绩效。
用修复前视后的特征数据 + 新阈值（buy 0.1 / sell -0.07）。

用法：
  python factor_combo_search.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR))
from backtest import run_backtest, compute_metrics, load_config, load_features_df

FACTORS = ["momentum", "trend", "volume_price", "rsrs", "relative_strength"]

# 组合定义：保留的因子
COMBOS = {
    "全5因子": FACTORS,
    "去trend": [f for f in FACTORS if f != "trend"],
    "去rs": [f for f in FACTORS if f != "relative_strength"],
    "去trend+rs": [f for f in FACTORS if f not in ("trend", "relative_strength")],
    "量价+动量+rsrs": ["momentum", "volume_price", "rsrs"],
    "量价+动量": ["momentum", "volume_price"],
}


def run_combo(df: pd.DataFrame, config: dict, factors: list, start: str, end: str, cost: float):
    """用给定因子组合重算 total_score 并回测"""
    cfg = copy.deepcopy(config)
    # 权重：保留因子均分
    w = 1.0 / len(factors)
    weights = {f: w for f in factors}
    cfg["weights"] = weights
    cfg["thresholds"] = {"buy": 0.1, "sell": -0.07, "confirm_days": 2, "weekly_filter_percentile": 0.2}
    # 纯净对比：关掉自适应阈值
    cfg["adaptive_thresholds"] = {"enabled": False}

    df2 = df.copy()
    df2["total_score"] = sum(df2[f].fillna(0) * w for f in factors)

    res = run_backtest(df2, cfg, start, end, cost)
    eq = res["equity_curve"]
    m = compute_metrics(eq["equity"], trades_df=res["trades"], n_days=len(eq))
    return m


def main():
    config = load_config()
    df = load_features_df("588000")
    start, end, cost = "2023-01-01", "2026-07-31", 0.00055

    print(f"\n═══ 因子组合搜索（前视修正后 + 阈值 0.1/-0.07）═══")
    print(f"区间: {start} ~ {end}, 费率 {cost}\n")
    print(f"  {'组合':<16s}{'年化':>8s}{'夏普':>8s}{'回撤':>9s}{'交易':>6s}")
    print(f"  {'-'*52}")

    results = {}
    for name, factors in COMBOS.items():
        try:
            m = run_combo(df, config, factors, start, end, cost)
            results[name] = m
            print(f"  {name:<16s}{m['annual_return']:>+7.1f}%{m['sharpe']:>8.3f}{m['max_drawdown']:>8.1f}%{m['trade_count']:>6d}")
        except Exception as e:
            print(f"  {name:<16s} ERR: {e}")

    # 最优
    best = max(results.items(), key=lambda kv: kv[1]["sharpe"])
    print(f"\n🏆 最优组合: {best[0]} (夏普 {best[1]['sharpe']:.3f})")
    print(f"   weights = {{ {', '.join(f'{f}: {1.0/len(COMBOS[best[0]]):.3f}' for f in COMBOS[best[0]])} }}")


if __name__ == "__main__":
    main()
