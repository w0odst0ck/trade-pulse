#!/usr/bin/env python3
"""position_map_search.py — 仓位映射候选验证（B 项）

遍历候选仓位映射，跑 5 折滚动 OOS（每折：前 80% 训练确定映射 → 后 20% 验证），
对比全段绩效 + OOS 稳定性。过门禁（OOS 全面占优且非过拟合）才建议采纳，
否则维持现状（基线同构是硬约束）。

候选映射：
  A 现状     linear   base=0.3 slope=0.4 cap=0.7 cap_score=0.7（历史分段）
  B 平方     square   base=0.3 slope=0.7 cap=0.7（低分更保守）
  C 高下限   linear   base=0.5 slope=0.2 cap=0.7 cap_score=1.0
  D 低上限   linear   base=0.2 slope=0.4 cap=0.6 cap_score=1.0
  E 保守     linear   base=0.2 slope=0.3 cap=0.5 cap_score=1.0

用法：
  python3 tools/daily_pipeline/position_map_search.py [--trials 105]
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

from backtest import compute_metrics, run_backtest
from compute_features import load_config, load_features_cache

# 候选映射（config['position_map'] 覆盖值）
CANDIDATES = {
    "A_现状_linear_03_04_07": {"type": "linear", "base": 0.3, "slope": 0.4, "cap": 0.7, "cap_score": 0.7},
    "B_平方_03_07_07":        {"type": "square", "base": 0.3, "slope": 0.7, "cap": 0.7, "cap_score": 1.0},
    "C_高下限_05_02_07":      {"type": "linear", "base": 0.5, "slope": 0.2, "cap": 0.7, "cap_score": 1.0},
    "D_低上限_02_04_06":      {"type": "linear", "base": 0.2, "slope": 0.4, "cap": 0.6, "cap_score": 1.0},
    "E_保守_02_03_05":        {"type": "linear", "base": 0.2, "slope": 0.3, "cap": 0.5, "cap_score": 1.0},
}


def _with_pm(config: dict, pm: dict) -> dict:
    """拷贝 config 并覆盖 position_map"""
    c = dict(config)
    c["position_map"] = dict(pm)
    return c


def run_fold(config, features_df, start, end, cost) -> dict:
    """单折回测 → metrics（run_backtest 返回 trades/equity，需 compute_metrics 汇总）"""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = run_backtest(features_df, config, start, end, cost)
    equity = result.get('equity_curve')
    if equity is None or len(equity) == 0:
        raise ValueError('run_backtest 无权益曲线')
    equity = equity.copy()
    equity['equity'] = pd.to_numeric(equity['equity'], errors='coerce').fillna(0.0)
    metrics = compute_metrics(equity['equity'], trades_df=result.get('trades'), n_days=len(equity))
    return metrics


def main():
    parser = argparse.ArgumentParser(description='仓位映射候选验证')
    parser.add_argument('--symbol', default='588000')
    parser.add_argument('--cost', type=float, default=0.00055)
    args = parser.parse_args()

    config = load_config()
    features_df = load_features_cache(args.symbol)
    if len(features_df) == 0:
        print(f"  [ERR] {args.symbol} 特征数据为空")
        sys.exit(1)

    full_start = str(features_df['date'].iloc[0].date())
    full_end = str(features_df['date'].iloc[-1].date())
    n = len(features_df)
    print(f"数据: {full_start} ~ {full_end}（{n} 交易日）")

    # ── 1. 全段绩效对比 ──
    print("\n===== 全段回测对比 =====")
    full_results = {}
    for name, pm in CANDIDATES.items():
        c = _with_pm(config, pm)
        m = run_fold(c, features_df, full_start, full_end, args.cost)
        full_results[name] = m
        print(f"  {name:<24} 夏普 {m['sharpe']:.4f} | 年化 {m['annual_return']:+.2f}% "
              f"| 回撤 {m['max_drawdown']:.2f}% | 交易 {m['trade_count']}")

    base_sharpe = full_results["A_现状_linear_03_04_07"]["sharpe"]
    winners = [n for n, m in full_results.items()
               if m['sharpe'] > base_sharpe + 0.02]  # 全段需明显优于基线
    print(f"\n  基线夏普 {base_sharpe:.4f}；全段占优(>+0.02): {winners or '无'}")

    # ── 2. 5 折滚动 OOS ──
    print("\n===== 5 折滚动 OOS =====")
    n_folds = 5
    fold_size = n // n_folds
    oos_results = {name: [] for name in CANDIDATES}

    for fold in range(n_folds):
        val_start_idx = fold * fold_size
        val_end_idx = n if fold == n_folds - 1 else (fold + 1) * fold_size
        # 训练 = 验证段之前的数据；fold0 无前序数据（val_start_idx=0）→ 跳过
        # （否则训练=验证段同一数据，数据泄漏，OOS 门禁被污染）
        if val_start_idx == 0:
            print(f"  fold{fold}: 跳过（无前序训练数据）")
            continue
        train_end_idx = val_start_idx - 1
        train_df = features_df.iloc[:train_end_idx]
        val_df = features_df.iloc[val_start_idx:val_end_idx]

        if len(train_df) < 100 or len(val_df) < 50:
            continue

        t_start = str(train_df['date'].iloc[0].date())
        t_end = str(train_df['date'].iloc[-1].date())
        v_start = str(val_df['date'].iloc[0].date())
        v_end = str(val_df['date'].iloc[-1].date())

        # 训练段选最优映射（在训练段内网格选夏普最高）
        train_sharpes = {}
        for name, pm in CANDIDATES.items():
            c = _with_pm(config, pm)
            try:
                m = run_fold(c, train_df, t_start, t_end, args.cost)
                train_sharpes[name] = m['sharpe']
            except Exception as e:
                train_sharpes[name] = -9.9
                print(f"    [WARN] fold{fold} {name} 训练失败: {e}")
        # 全候选训练失败 → 该折无有效选参依据，跳过（避免哨兵选中失败候选污染 OOS）
        if all(v <= -9.9 for v in train_sharpes.values()):
            print(f"  fold{fold}: 跳过（全部候选训练失败）")
            continue
        best_train = max(train_sharpes, key=train_sharpes.get)

        # 验证段：用训练选中的映射
        c = _with_pm(config, CANDIDATES[best_train])
        try:
            m = run_fold(c, val_df, v_start, v_end, args.cost)
            oos_results[best_train].append(m['sharpe'])
            print(f"  fold{fold}: 训练最优={best_train}（夏普 {train_sharpes[best_train]:.3f}）"
                  f" → OOS 夏普 {m['sharpe']:+.3f}")
        except Exception as e:
            print(f"  fold{fold}: OOS 失败 {e}")

    # ── 3. 汇总 + 门禁 ──
    print("\n===== OOS 汇总 =====")
    summary = []
    for name in CANDIDATES:
        vals = oos_results[name]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        summary.append((name, mean, std, len(vals)))
        print(f"  {name:<24} OOS 夏普均值 {mean:+.3f}（std {std:.3f}，{len(vals)} 折）")

    if not summary:
        print("  [FAIL] 无有效 OOS 结果")
        sys.exit(1)

    # 门禁（双条件，与 docstring 一致：全段占优 AND OOS 占优）：
    #   1. 全段夏普须 > 基线 + 0.02
    #   2. OOS 均值须 > 基线 OOS + 0.05
    base_oos = [v for n, v, _, _ in summary if n.startswith("A_")]
    if not base_oos:
        print("\n  ❌ 基线映射从未被训练段选中——验证无效，强制维持现状")
        sys.exit(1)
    base_oos_mean = base_oos[0]
    print(f"\n  基线 OOS 均值 {base_oos_mean:+.3f}")
    passed = []
    for name, v, _, _ in summary:
        if name == "A_现状_linear_03_04_07":
            continue
        full_sharpe = full_results[name]['sharpe']
        full_ok = full_sharpe > base_sharpe + 0.02
        oos_ok = v > base_oos_mean + 0.05
        mark = "✅" if (full_ok and oos_ok) else "❌"
        print(f"  {mark} {name}: 全段 {full_sharpe:+.3f}(需>{base_sharpe+0.02:+.3f}) "
              f"OOS {v:+.3f}(需>{base_oos_mean+0.05:+.3f})")
        if full_ok and oos_ok:
            passed.append((name, v))
    if passed:
        print(f"  ✅ 候选过门禁: {passed}")
    else:
        print("  ❌ 无候选过门禁（全段/OOS 未全面占优）→ 维持现状映射")

    # 输出 JSON 供报告
    out = PROJECT_ROOT / "data" / args.symbol / "position_map_search.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "full": {k: {"sharpe": v["sharpe"], "annual": v["annual_return"],
                     "dd": v["max_drawdown"], "trades": v["trade_count"]}
                 for k, v in full_results.items()},
        "oos": {n: {"mean": float(np.mean(v)) if v else None, "std": float(np.std(v)) if len(v) > 1 else None,
                    "folds": len(v)} for n, v in oos_results.items()},
        "passed": passed,
    }, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n  [SAVE] {out}")


if __name__ == "__main__":
    main()
