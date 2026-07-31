#!/usr/bin/env python3
"""
parameter_search.py — 单参数留一年交叉验证

搜索买入阈值 {0.1, 0.2, 0.3}，用 2 折叠交叉验证选最优。

折叠：
  训练集                 验证集
  ────────────────       ────────────────
  2023-01 ~ 2024-12     2025-01 ~ 2025-12
  2024-01 ~ 2025-12     2026-01 ~ 今天

输出：每个阈值的验证集平均夏普 → 选最优
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

sys.path.insert(0, str(SCRIPT_DIR))
from backtest import load_features_df, run_backtest, compute_metrics


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


END = date.today().strftime('%Y-%m-%d')

FOLDS = [
    ('2023-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
    ('2024-01-01', '2025-12-31', '2026-01-01', END),
]

CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3]


def main():
    config = load_config()
    factors = ['momentum', 'trend', 'volatility', 'volume_price', 'rsrs', 'relative_strength']
    weights = config['weights']

    print(f"\n{'=' * 50}")
    print(f"  参数搜索：买入阈值")
    print(f"{'=' * 50}")

    features_df = load_features_df(config['symbol'])
    print(f"  [OK] 特征 {len(features_df)} 条")

    results = []

    for buy_th in CANDIDATES:
        sell_th = round(buy_th * (-0.67), 2)
        config_copy = config.copy()
        config_copy['thresholds'] = {
            'buy': buy_th,
            'sell': sell_th,
            'confirm_days': 2,
            'weekly_filter_percentile': 0.2,
        }

        feat_copy = features_df.copy()
        fold_scores = []

        for _, _, val_start, val_end in FOLDS:
            try:
                val = run_backtest(feat_copy, config_copy, val_start, val_end, 0.00055)
                val_df = val['equity_curve']
                if len(val_df) > 0:
                    m = compute_metrics(val_df['equity'], trades_df=val['trades'], n_days=len(val_df))
                    fold_scores.append({
                        'fold': f"{val_start}~{val_end}",
                        'sharpe': m['sharpe'],
                        'annual_return': m['annual_return'],
                        'max_drawdown': m['max_drawdown'],
                        'trade_count': m['trade_count'],
                    })
            except (ValueError, KeyError) as e:
                print(f"  [ERR] buy_th={buy_th}: {e}")

        if fold_scores:
            avg_sharpe = np.mean([f['sharpe'] for f in fold_scores])
            avg_return = np.mean([f['annual_return'] for f in fold_scores])
            avg_dd = np.mean([f['max_drawdown'] for f in fold_scores])
            total_trades = sum(f['trade_count'] for f in fold_scores)

            results.append({
                'buy_th': buy_th,
                'sell_th': sell_th,
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'avg_dd': avg_dd,
                'total_trades': total_trades,
                'folds': fold_scores,
            })

    print(f"\n  {'=' * 55}")
    print(f"  {'买入阈值':>8s}  {'卖出阈值':>8s}  {'夏普均值':>8s}  {'年化均值':>8s}  {'回撤均值':>8s}  {'交易总数':>8s}")
    print(f"  {'─' * 55}")
    for r in sorted(results, key=lambda x: -x['avg_sharpe']):
        print(f"  {r['buy_th']:>8.1f}  {r['sell_th']:>8.2f}  {r['avg_sharpe']:>8.4f}  "
              f"{r['avg_return']:>+7.2f}%  {r['avg_dd']:>7.2f}%  {r['total_trades']:>8d}")

    if results:
        best = max(results, key=lambda x: x['avg_sharpe'])
        print(f"  {'─' * 55}")
        print(f"\n  ✅ 最优: 买入阈值 = {best['buy_th']}, 卖出阈值 = {best['sell_th']}")
        print(f"     验证集夏普均值: {best['avg_sharpe']:.4f}")
        print(f"     验证集年化均值: {best['avg_return']:+.2f}%")
        print(f"     验证集回撤均值: {best['avg_dd']:.2f}%")
        print(f"     总交易次数: {best['total_trades']}")

        print(f"\n  各折叠详情:")
        for f in best['folds']:
            print(f"    {f['fold']}: 夏普 {f['sharpe']:.4f}, 年化 {f['annual_return']:+.2f}%, "
                  f"回撤 {f['max_drawdown']:.2f}%, 交易 {f['trade_count']} 次")
    print(f"\n{'=' * 50}\n")


if __name__ == '__main__':
    main()
