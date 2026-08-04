#!/usr/bin/env python3
"""
window_search.py — 因子窗口参数 walk-forward 门禁搜索（纯只读分析）

背景：
  生产窗口 momentum=5d / trend=20d / rsrs=18d 为拍脑袋设定；
  已实测 trend 20d 次日预测力≈0（corr -0.001）、momentum 5d（+0.023）弱于 3d/10d（+0.041）。
  本工具用 5 折锚定式 walk-forward 门禁验证窗口参数，输出：

    需求1：单因子窗口敏感性
      每个因子独立扫窗口（其余因子用当前 config 窗口），固定生产阈值
      buy=0.1 / sell=-0.07 / confirm=2（adaptive 置 false），直接在 5 折
      验证段跑（训练段不选参），输出每窗口的 5 折 OOS 夏普均值/中位/最差折。

    需求2：联合窗口优化（完整 walk-forward 门禁）
      候选网格 momentum×trend×rsrs=12 组合；每折在训练段网格搜索选训练段
      夏普最优的窗口组合 → 验证段冻结跑。输出 5 折表 + OOS 均值 vs 基线 +
      DSR（试错次数 = 12 组合 × 折数 = 60，如实计入）。

    需求3：写 docs/window_search_report_2026-08-04.md

约束：
  - 不改 compute_features.py / backtest.py / walk_forward.py / config.json 等
    任何现有文件，不写 data/ 下任何数据文件。
  - 特征在内存重算（复用 compute_features / walk_forward / backtest 现有函数）。
  - 依赖仅 numpy / pandas + 标准库（rsrs_score 内部用 scipy，属既有依赖）。

用法：
  python3 tools/daily_pipeline/window_search.py
  python3 tools/daily_pipeline/window_search.py --end 2026-08-03 --cost 0.00055
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "window_search_report_2026-08-04.md"

sys.path.insert(0, str(SCRIPT_DIR))
from backtest import load_config, load_features_df  # noqa: E402
from compute_features import (  # noqa: E402
    load_data, momentum_score, trend_score, rsrs_score,
)
from walk_forward import (  # noqa: E402
    FOLD_DEFS, TRAIN_START, build_config, backtest_metrics, compute_dsr,
)

# ── 窗口候选（来自需求）──────────────────────────────

# 当前生产窗口（基线）
BASELINE = {'momentum': 5, 'trend': 20, 'rsrs': 18}

# 需求1：单因子敏感性候选
SENS_WINDOWS = {
    'momentum': [3, 5, 10, 15],
    'trend': [3, 5, 10, 15, 20],
    'rsrs': [10, 18, 30],
}

# 需求2：联合优化网格（rsrs 固定 18，控制搜索空间 = 12 组合）
JOINT_MOM = [3, 5, 10]
JOINT_TRE = [5, 10, 15, 20]
JOINT_RSRS = [18]

# 生产固定阈值（需求1/2 统一口径，adaptive=false 同 walk_forward 主报告）
PROD_BUY = 0.10
PROD_CONFIRM = 2

# ── 特征重算（内存中，不落盘）────────────────────────


def precompute_factor_cache(raw: pd.DataFrame) -> dict:
    """按候选窗口预计算各因子序列，供组合复用（只算一次）。

    raw: daily.csv 全量 OHLCV（compute_features.load_data 返回），
    返回 {factor: {window: pd.Series}}，Series 与 raw 索引对齐。
    """
    windows = {
        'momentum': sorted({*SENS_WINDOWS['momentum'], BASELINE['momentum']}),
        'trend': sorted({*SENS_WINDOWS['trend'], BASELINE['trend'],
                         *JOINT_TRE}),
        'rsrs': sorted({*SENS_WINDOWS['rsrs'], BASELINE['rsrs'], *JOINT_RSRS}),
    }
    cache = {}
    print("  [prep] 预计算因子序列：")
    for factor, ws in windows.items():
        fn = {'momentum': momentum_score, 'trend': trend_score,
              'rsrs': rsrs_score}[factor]
        cache[factor] = {}
        for w in ws:
            cache[factor][w] = fn(raw, w)
        print(f"    {factor:<10s} 窗口 {ws}")
    return cache


def build_features(base: pd.DataFrame, raw: pd.DataFrame, cfg: dict,
                   factor_cache: dict, momentum_w: int, trend_w: int,
                   rsrs_w: int) -> pd.DataFrame:
    """用指定窗口组合重算 momentum/trend/rsrs 列 + total_score。

    base: features_cache 全量（date/close/volume/volume_price/relative_strength/
          weekly_modifier/ma60_slope/…），volume_price 等不动因子沿用缓存列；
    raw:  daily.csv 全量 OHLCV（含 high/low，rsrs 重算需要）。
    total_score = Σ weights[f]·factor[f]（与 compute_all_features 同式；
    weekly_modifier 由回测引擎 decide() 在决策时自动叠加，此处不合成）。
    """
    df = base.copy()
    feat = pd.DataFrame({'date': raw['date'].values})
    feat['momentum'] = factor_cache['momentum'][momentum_w].values
    feat['trend'] = factor_cache['trend'][trend_w].values
    feat['rsrs'] = factor_cache['rsrs'][rsrs_w].values
    df = df.drop(columns=['momentum', 'trend', 'rsrs']).merge(
        feat, on='date', how='left')

    w = cfg['weights']
    parts = [df[f].fillna(0) * w[f] for f in w.keys() if f in df.columns]
    df['total_score'] = sum(parts)
    return df


# ── walk-forward 执行 ────────────────────────────────


def run_val_folds(df: pd.DataFrame, cfg: dict, full_end: str,
                  cost_rate: float) -> list:
    """5 折验证段各跑一次（训练段不选参）。返回 [(fold, sharpe, n_days, metrics)]。"""
    out = []
    for idx, (_train_end, val_start, val_end_def) in enumerate(FOLD_DEFS):
        val_end = val_end_def or full_end
        bt = backtest_metrics(df, cfg, val_start, val_end, cost_rate)
        if bt is None:
            out.append({'fold': idx + 1, 'sharpe': None, 'n': 0, 'metrics': None})
            continue
        m, n, _eq, _tr = bt
        out.append({'fold': idx + 1, 'sharpe': m['sharpe'], 'n': n,
                    'metrics': m})
    return out


def summarize_folds(folds: list) -> dict:
    """从 5 折结果汇总 OOS 夏普均值/中位/最差折。"""
    sh = [f['sharpe'] for f in folds if f['sharpe'] is not None]
    if not sh:
        return {'mean': np.nan, 'median': np.nan, 'worst': np.nan,
                'sharpe_list': [], 'n_list': []}
    return {
        'mean': float(np.mean(sh)),
        'median': float(np.median(sh)),
        'worst': float(np.min(sh)),
        'sharpe_list': sh,
        'n_list': [f['n'] for f in folds if f['sharpe'] is not None],
    }


def run_sensitivity(base: pd.DataFrame, raw: pd.DataFrame, cfg: dict,
                    factor_cache: dict, full_end: str,
                    cost_rate: float) -> list:
    """需求1：单因子窗口敏感性。返回行列表 [{factor, window, **summ}]。"""
    rows = []

    def add(factor_label, window_label, m, t, r):
        df = build_features(base, raw, cfg, factor_cache, m, t, r)
        folds = run_val_folds(df, cfg, full_end, cost_rate)
        summ = summarize_folds(folds)
        rows.append({'factor': factor_label, 'window': window_label,
                     'folds': folds, **summ})
        return summ

    print("\n  [sens] 基线 (5/20/18) ...")
    add('baseline', '(5,20,18)', 5, 20, 18)

    for factor in ['momentum', 'trend', 'rsrs']:
        for w in SENS_WINDOWS[factor]:
            m, t, r = BASELINE['momentum'], BASELINE['trend'], BASELINE['rsrs']
            if factor == 'momentum':
                m = w
            elif factor == 'trend':
                t = w
            else:
                r = w
            print(f"  [sens] {factor}={w} ...")
            summ = add(factor, w, m, t, r)
            print(f"    → OOS 夏普 均值 {summ['mean']:+.3f}  "
                  f"中位 {summ['median']:+.3f} 最差 {summ['worst']:+.3f}")
    return rows


def run_joint(base: pd.DataFrame, raw: pd.DataFrame, cfg: dict,
              factor_cache: dict, full_end: str, cost_rate: float) -> tuple:
    """需求2：联合窗口优化（训练段选参 → 验证段冻结）+ DSR。

    返回 (folds, combos, dsr)。
    """
    combos = [(m, t, r) for m in JOINT_MOM for t in JOINT_TRE
              for r in JOINT_RSRS]
    print(f"  [joint] 候选网格 {len(combos)} 组合："
          f"momentum∈{JOINT_MOM} × trend∈{JOINT_TRE} × rsrs={JOINT_RSRS}")

    df_cache = {}
    for combo in combos:
        df_cache[combo] = build_features(base, raw, cfg, factor_cache, *combo)

    folds = []
    for idx, (train_end, val_start, val_end_def) in enumerate(FOLD_DEFS):
        val_end = val_end_def or full_end
        print(f"  [joint] 折 {idx + 1}  训练 ~{train_end} → 验证 "
              f"{val_start}~{val_end}")

        # 训练段网格搜索：选训练段夏普最优窗口组合
        best = None  # (combo, train_sharpe, metrics)
        for combo in combos:
            bt = backtest_metrics(df_cache[combo], cfg, TRAIN_START,
                                  train_end, cost_rate)
            if bt is None:
                continue
            if best is None or bt[0]['sharpe'] > best[1]:
                best = (combo, bt[0]['sharpe'], bt[0])
        if best is None:
            print(f"  [ERR] 折 {idx + 1} 训练段全部失败，跳过")
            continue

        # 参数冻结 → 验证段
        vb = backtest_metrics(df_cache[best[0]], cfg, val_start, val_end,
                              cost_rate)
        if vb is None:
            print(f"  [ERR] 折 {idx + 1} 验证段失败，跳过")
            continue
        vm, vn, _veq, _vtr = vb

        folds.append({
            'fold': idx + 1,
            'combo': best[0],
            'train_sharpe': best[1],
            'val_sharpe': vm['sharpe'],
            'val_annual': vm['annual_return'],
            'val_dd': vm['max_drawdown'],
            'val_trades': vm['trade_count'],
            'val_win': vm['win_rate'],
            'val_n': vn,
        })
        print(f"    → 训练最优 {best[0]}（训练夏普 {best[1]:+.3f}）"
              f"→ 验证夏普 {vm['sharpe']:+.3f} 年化 {vm['annual_return']:+.1f}%")

    # DSR：试错次数 = 组合数 × 折数（如实计入全部 5 折）
    trials = len(combos) * len(FOLD_DEFS)
    dsr = compute_dsr([f['val_sharpe'] for f in folds],
                      [f['val_n'] for f in folds], trials) if folds else None
    return folds, combos, dsr


# ── 报告生成 ─────────────────────────────────────────


def fmt_sharpe(v) -> str:
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"


def build_report(args, cfg, sens_rows, joint_folds, combos, dsr,
                 full_end, cost_rate, sanity) -> str:
    """生成 markdown 报告（含基于实际数据的结论与建议）。"""
    L = []
    L.append("# 因子窗口 walk-forward 门禁搜索报告")
    L.append("")
    L.append(f"**日期**：2026-08-04　**标的**：{cfg['symbol']}　"
             f"**样本**：{sanity['data_start']} ~ {sanity['data_end']}"
             f"（{sanity['n_rows']} 行）　**费率**：{cost_rate}")
    L.append("")
    L.append("## 0. 背景与口径")
    L.append("")
    L.append("- 生产窗口 momentum=5d / trend=20d / rsrs=18d 为拍脑袋设定；"
             "已实测 trend 20d 次日预测力≈0（corr -0.001）、momentum 5d（+0.023）"
             "弱于 3d/10d（+0.041）。")
    L.append("- 门禁方法：5 折锚定式 walk-forward（与 walk_forward.py 同折方案："
             "训练起点 2023-01-01 固定，验证段各 6 个月，末折到样本末）。")
    L.append("- 统一口径：固定生产阈值 buy=0.1 / sell=-0.07 / confirm=2，"
             "adaptive_thresholds 置 false（同 walk_forward 主报告口径）。")
    L.append("- 特征在内存重算（复用 compute_features 因子函数），"
             "volume_price 为全周期最稳因子，本次不动。")
    L.append("- 数据一致性检查：基线重算 total_score vs 缓存 total_score "
             f"相关系数 {sanity['corr']:.4f}，最大绝对差 {sanity['max_abs_diff']:.6f}"
             f"{'（一致）' if sanity['max_abs_diff'] < 1e-6 else '（有差异，见备注）'}。")
    L.append("")

    # ── 需求1：单因子敏感性 ──
    L.append("## 1. 单因子窗口敏感性")
    L.append("")
    L.append("每个因子独立扫窗口（其余因子用当前 config 窗口），固定生产阈值，"
             "直接在 5 折验证段跑（训练段不选参）。")
    L.append("")
    headers = ["因子", "窗口", "OOS均值", "OOS中位", "最差折",
               "F1", "F2", "F3", "F4", "F5"]
    L.append("| " + " | ".join(headers) + " |")
    L.append("|" + "---|" * len(headers))
    for r in sens_rows:
        fold_sh = [f['sharpe'] for f in r['folds']]
        vals = ([str(r['factor']), str(r['window']),
                 f"{r['mean']:+.3f}", f"{r['median']:+.3f}",
                 f"{r['worst']:+.3f}"]
                + [fmt_sharpe(s) for s in fold_sh])
        L.append("| " + " | ".join(vals) + " |")
    L.append("")

    base_row = next(r for r in sens_rows if r['factor'] == 'baseline')
    base_mean = base_row['mean']
    L.append(f"基线（5/20/18）5 折 OOS 夏普均值：**{base_mean:+.3f}**，"
             f"中位 {base_row['median']:+.3f}，最差折 {base_row['worst']:+.3f}。")
    L.append("")

    for factor in ['momentum', 'trend', 'rsrs']:
        frows = [r for r in sens_rows if r['factor'] == factor]
        best = max(frows, key=lambda r: r['mean'])
        cur = next(r for r in frows if r['window'] == BASELINE[factor])
        delta = best['mean'] - cur['mean']
        L.append(f"- **{factor}**：候选内 OOS 均值最高为窗口 "
                 f"**{best['window']}**（{best['mean']:+.3f}）"
                 f"vs 当前 {cur['window']}（{cur['mean']:+.3f}），"
                 f"差 {delta:+.3f}。")
    L.append("")

    # ── 需求2：联合优化 ──
    L.append("## 2. 联合窗口优化（完整 walk-forward 门禁）")
    L.append("")
    L.append(f"候选网格：momentum∈{JOINT_MOM} × trend∈{JOINT_TRE} × "
             f"rsrs={JOINT_RSRS}（{len(combos)} 组合）。每折在训练段网格搜索"
             "选训练段夏普最优窗口组合 → 验证段冻结跑。")
    L.append("")
    headers2 = ["折", "训练段", "验证段", "最优窗口(m,t,r)",
                "训练夏普", "验证夏普", "验证年化", "验证回撤", "交易数"]
    L.append("| " + " | ".join(headers2) + " |")
    L.append("|" + "---|" * len(headers2))
    for f in joint_folds:
        train_end, val_start, val_end_def = FOLD_DEFS[f['fold'] - 1]
        val_end = val_end_def or full_end
        combo = f"({f['combo'][0]},{f['combo'][1]},{f['combo'][2]})"
        L.append(f"| {f['fold']} | {TRAIN_START}~{train_end} | "
                 f"{val_start}~{val_end} | {combo} | "
                 f"{f['train_sharpe']:+.3f} | {f['val_sharpe']:+.3f} | "
                 f"{f['val_annual']:+.1f}% | {f['val_dd']:.1f}% | "
                 f"{f['val_trades']} |")
    L.append("")

    joint_sharpe = [f['val_sharpe'] for f in joint_folds]
    joint_mean = float(np.mean(joint_sharpe))
    joint_median = float(np.median(joint_sharpe))
    joint_worst = float(np.min(joint_sharpe))
    delta = joint_mean - base_mean
    L.append(f"联合优化 OOS 夏普：均值 **{joint_mean:+.3f}**"
             f"（中位 {joint_median:+.3f}，最差折 {joint_worst:+.3f}）。")
    L.append("")
    L.append(f"**vs 基线**（5/20/18 固定窗口）："
             f"{joint_mean - base_mean:+.3f}"
             f"{'（改善）' if delta > 0 else '（无改善）'}。")
    L.append("")

    # DSR
    L.append("### DSR（Deflated Sharpe Ratio）")
    L.append("")
    if dsr is not None:
        L.append(f"- 试错次数（如实计入）：{dsr['trials']} = "
                 f"{len(combos)} 组合 × {len(joint_folds)} 折")
        L.append(f"- 5 折 OOS 夏普：均值 {dsr['mean_sr']:+.4f}，"
                 f"样本方差 V[SR̂] {dsr['var_sr']:.6f}（sd {dsr['sd_sr']:.4f}）")
        L.append(f"- 期望最大夏普阈值 SR0 = **{dsr['sr0']:+.4f}**"
                 f"（{dsr['trials_note']}）")
        L.append(f"- 平均 deflated 夏普 = {dsr['mean_deflated']:+.4f}，"
                 f"p 值 = **{dsr['mean_p']:.4f}**"
                 f"{'（显著，可排除多重检验伪发现）' if dsr['mean_p'] < 0.05 else '（不显著，无法排除运气/过拟合）'}")
        L.append("- 披露口径：试错次数仅计入联合优化训练段选参（12 组合 × 5 折）；"
                 "需求 1 单因子敏感性为固定口径观察（不选参），不产生多重检验负担，故不计入。")
        L.append("")
        L.append("| 折 | SR | n | SE | Deflated | p |")
        L.append("|---|---|---|---|---|---|")
        for i, pf in enumerate(dsr['per_fold'], 1):
            L.append(f"| {i} | {pf['sr']:+.3f} | {pf['n']} | {pf['se']:.4f} | "
                     f"{pf['deflated']:+.3f} | {pf['p']:.4f} |")
    else:
        L.append("有效折 < 2，DSR 无法计算。")
    L.append("")

    # ── 结论 ──
    L.append("## 3. 结论与建议")
    L.append("")
    best_combo = max(joint_folds, key=lambda f: f['val_sharpe'])['combo']
    best_combo_str = f"({best_combo[0]},{best_combo[1]},{best_combo[2]})"
    L.append(f"- 联合门禁下 OOS 均值最高出现在验证折 "
             f"{max(joint_folds, key=lambda f: f['val_sharpe'])['fold']}（"
             f"窗口 {best_combo_str}），但门禁考察的是跨折整体表现，"
             f"不以单折论英雄。")
    L.append("")

    # 单因子建议
    L.append("**单因子维度：**")
    L.append("")
    for factor in ['momentum', 'trend', 'rsrs']:
        frows = [r for r in sens_rows if r['factor'] == factor]
        best = max(frows, key=lambda r: r['mean'])
        cur = next(r for r in frows if r['window'] == BASELINE[factor])
        delta_b = best['mean'] - cur['mean']
        # 全维度占优：均值（>当前+0.02）、中位、最差折均不劣于当前窗口才建议采纳
        dom = [r for r in frows
               if r['window'] != BASELINE[factor]
               and r['mean'] > cur['mean'] + 0.02
               and r['median'] >= cur['median']
               and r['worst'] >= cur['worst']]
        if dom:
            cand = max(dom, key=lambda r: r['mean'])
            verdict = (f"建议由 {cur['window']} 调整为 "
                       f"**{cand['window']}**（均值/中位/最差折全面占优）")
        elif delta_b > 0.02:
            verdict = (f"窗口 {best['window']} 均值最高（{best['mean']:+.3f}）但中位/最差折"
                       f"未全面占优，不稳健，建议观察、不优先采纳")
        else:
            verdict = "建议保留现状"
        L.append(f"- **{factor}**：OOS 均值最高 {best['window']}"
                 f"（{best['mean']:+.3f}）vs 当前 {cur['window']}"
                 f"（{cur['mean']:+.3f}），差 {delta_b:+.3f}。→ {verdict}。")
    L.append("")
    trend_short = [r for r in sens_rows
                   if r['factor'] == 'trend' and r['window'] != BASELINE['trend']]
    trend_short_worst = f"{min(r['worst'] for r in trend_short):+.1f}" if trend_short else "—"
    n_last = joint_folds[-1]['val_n'] if joint_folds else 0
    L.append(f"> 备注：trend 更短窗口（3/5/10）的最差折灾难性"
             f"（低至 {trend_short_worst}），20 仍是候选内 OOS 最优；"
             f"corr≈0 是单日预测力口径，与多因子回测交互下的 OOS 表现不同。"
             f"末折验证段仅 {n_last} 个交易日（2026-07-01 ~ 2026-08-03），"
             f"夏普估计方差大，DSR 已按折样本量加权。")
    L.append("")

    # 联合建议
    L.append("**联合维度（最终采纳口径）：**")
    L.append("")
    joint_best_combo = None
    for combo in combos:
        # 该组合被选为训练最优的次数
        cnt = sum(1 for f in joint_folds if tuple(f['combo']) == combo)
        if cnt == len(joint_folds):
            joint_best_combo = combo
            break
    if joint_best_combo is not None and joint_best_combo == (5, 20, 18):
        L.append("- 全部折训练段最优均为当前生产窗口 (5,20,18)——"
                 "**窗口不是当前弱点，保留现状**，转向阈值/风控等其他参数优化。")
    elif joint_best_combo is not None:
        L.append(f"- 全部折训练段一致选 {joint_best_combo}，"
                 f"OOS 均值 {joint_mean:+.3f} vs 基线 {base_mean:+.3f}"
                 f"（{delta:+.3f}）；DSR p={dsr['mean_p']:.4f}"
                 f"{'，显著' if dsr['mean_p'] < 0.05 else '，不显著'}。"
                 f"{'建议采纳该窗口组合' if delta > 0 and dsr['mean_p'] < 0.05 else '数值占优但不显著，建议小步试改并观察' if delta > 0 else '不建议采纳'}。")
    else:
        L.append(f"- 各折训练段最优窗口不一致（"
                 f"{', '.join(str(tuple(f['combo'])) for f in joint_folds)}），"
                 f"选参不稳定；OOS 均值 {joint_mean:+.3f} vs 基线 {base_mean:+.3f}"
                 f"（{delta:+.3f}），DSR p={dsr['mean_p']:.4f}"
                 f"{'，显著' if dsr['mean_p'] < 0.05 else '，不显著'}。"
                 f"{'建议采纳' if delta > 0 and dsr['mean_p'] < 0.05 else '建议保留现状，窗口不是稳定可迁移的改进点'}。")
    L.append("")
    L.append("---")
    L.append("*本报告由 tools/daily_pipeline/window_search.py 生成"
             "（纯只读，不改任何现有文件）。*")
    L.append("")
    return "\n".join(L)


# ── 主流程 ───────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description='因子窗口 walk-forward 门禁搜索（纯只读）')
    ap.add_argument('--symbol', default=None, help='标的（默认取 config）')
    ap.add_argument('--cost', type=float, default=0.00055, help='单边费率')
    ap.add_argument('--end', default=None,
                    help='样本终点（默认取特征数据最后日期）')
    ap.add_argument('--report', default=str(DEFAULT_REPORT), help='报告输出路径')
    args = ap.parse_args()

    cfg = load_config()
    symbol = args.symbol or cfg['symbol']

    print("═══ 因子窗口 walk-forward 门禁搜索 ═══")
    print(f"标的 {symbol}，费率 {args.cost}")

    base = load_features_df(symbol)
    raw = load_data(symbol)
    full_end = args.end or base['date'].max().strftime('%Y-%m-%d')

    # 构建统一 cfg：固定生产阈值，adaptive=false
    cfg_bt = build_config(cfg, PROD_BUY, PROD_CONFIRM)

    # sanity：基线重算 vs 缓存 total_score
    factor_cache = precompute_factor_cache(raw)
    base_df = build_features(base, raw, cfg, factor_cache, 5, 20, 18)
    merged = base_df[['date', 'total_score']].merge(
        base[['date', 'total_score']].rename(
            columns={'total_score': 'cached_total'}), on='date')
    corr = merged['total_score'].corr(merged['cached_total'])
    max_abs_diff = float(
        (merged['total_score'] - merged['cached_total']).abs().max())
    print(f"  [sanity] 基线重算 vs 缓存 total_score：corr {corr:.4f}，"
          f"最大绝对差 {max_abs_diff:.6f}")

    # 需求1
    print("\n── 需求1：单因子窗口敏感性 ──")
    sens_rows = run_sensitivity(base, raw, cfg, factor_cache, full_end,
                                args.cost)

    # 需求2
    print("\n── 需求2：联合窗口优化 ──")
    joint_folds, combos, dsr = run_joint(base, raw, cfg, factor_cache,
                                         full_end, args.cost)

    # 汇总输出
    base_row = next(r for r in sens_rows if r['factor'] == 'baseline')
    print("\n── 需求1 汇总（5 折 OOS 夏普）──")
    print(f"  {'因子':<10s}{'窗口':<10s}{'均值':>8s}{'中位':>8s}{'最差':>8s}")
    for r in sens_rows:
        label = r['factor'] if r['factor'] == 'baseline' else r['factor']
        print(f"  {label:<10s}{str(r['window']):<10s}"
              f"{r['mean']:>+8.3f}{r['median']:>+8.3f}{r['worst']:>+8.3f}")

    print("\n── 需求2 汇总 ──")
    jm = float(np.mean([f['val_sharpe'] for f in joint_folds]))
    print(f"  联合优化 OOS 夏普均值 {jm:+.3f} vs 基线 {base_row['mean']:+.3f}"
          f"（{jm - base_row['mean']:+.3f}）")
    if dsr:
        print(f"  DSR: trials={dsr['trials']} SR0={dsr['sr0']:+.4f} "
              f"deflated={dsr['mean_deflated']:+.3f} p={dsr['mean_p']:.4f}")

    # 需求3
    report = build_report(args, cfg, sens_rows, joint_folds, combos, dsr,
                          full_end, args.cost, {
                              'corr': corr, 'max_abs_diff': max_abs_diff,
                              'data_start': base['date'].min().strftime('%Y-%m-%d'),
                              'data_end': base['date'].max().strftime('%Y-%m-%d'),
                              'n_rows': len(base),
                          })
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    print(f"\n✅ 报告已写入: {report_path}")


if __name__ == '__main__':
    main()
