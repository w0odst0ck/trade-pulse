#!/usr/bin/env python3
"""
regime_weight_search.py — 市场状态自适应因子权重（攻守分层）walk-forward 门禁验证

纯只读分析工具：
  - 不改 compute_features.py / backtest.py / walk_forward.py / config.json 等
    任何现有文件，不写 data/ 下任何数据文件（特征在内存重算）。
  - 复用 backtest 的 run_backtest / compute_metrics / load_features_df、
    compute_features.compute_total_score、walk_forward 的 5 折方案与 DSR 实现。
  - 依赖仅 numpy / pandas + Python 标准库（scipy 属既有依赖，rsrs_score 内部用）。

背景（2026-08-04 因子解剖）：
  熊市段（2023-01~2024-09）rsrs 是唯一强正预测因子（corr +0.063），
  trend/momentum 反向（-0.075/-0.014）；牛市段 volume_price 最强（+0.094）。
  当前四因子等权 0.25 浪费了攻守分工——本工具验证「按 ma60_slope 市场状态
  分层换权重」能否通过 5 折 walk-forward 门禁。

验证协议（风控类规则验证口径，不选参）：
  1) 每套候选：内存重算特征（weights_by_regime.enabled=true 覆盖 config，
     仅重算 total_score 列，因子列沿用 features_cache）→
  2) 生产 config 原样（adaptive_thresholds 启用，与实盘同口径）→
  3) 直接在 5 折验证段跑（训练段不选参）。
  对照组：等权 + 生产 config（同口径 5 折）。
  DSR：trials = 3（候选套数，如实计入；对照组是基线，不产生试错）。

用法：
  python3 tools/daily_pipeline/regime_weight_search.py
  python3 tools/daily_pipeline/regime_weight_search.py --cost 0.00055 \
      --report docs/regime_weight_report_2026-08-04.md
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
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "regime_weight_report_2026-08-04.md"

sys.path.insert(0, str(SCRIPT_DIR))
from backtest import load_config, load_features_df  # noqa: E402
from compute_features import compute_total_score  # noqa: E402
from walk_forward import (  # noqa: E402
    FOLD_DEFS, TRAIN_START, backtest_metrics, compute_dsr,
)

# ── 候选方案（结构假设，非精细调参；权重和均为 1.0）────────────

# 每套 = 完整 weights_by_regime 配置（enabled=true）
CANDIDATES = [
    {
        'name': 'A 攻守均衡',
        'desc': ('与实证方向一致但幅度温和：uptrend 抬 volume_price 至 0.35'
                 '（牛市最强因子 +0.094）；downtrend 重押 rsrs 至 0.50'
                 '（熊市唯一强正因子 +0.063）并压 momentum/trend'
                 '（熊市反向 -0.075/-0.014）；sideways 无实证指向，保持等权。'),
        'weights_by_regime': {
            'enabled': True,
            'ma60_slope_uptrend': 0.005,
            'ma60_slope_downtrend': -0.005,
            'uptrend': {'momentum': 0.25, 'trend': 0.25,
                        'volume_price': 0.35, 'rsrs': 0.15},
            'downtrend': {'momentum': 0.10, 'trend': 0.10,
                          'volume_price': 0.30, 'rsrs': 0.50},
            'sideways': {'momentum': 0.25, 'trend': 0.25,
                         'volume_price': 0.25, 'rsrs': 0.25},
        },
    },
    {
        'name': 'B 激进攻守',
        'desc': ('放大幅度：uptrend 加 momentum 至 0.30 押进攻'
                 '（rsrs 降至 0.10，牛市段 rsrs 弱）；downtrend rsrs 提至 0.60、'
                 'momentum/trend 压到 0.05，更彻底的反转防守'
                 '（熊市趋势类因子反向最强）。'),
        'weights_by_regime': {
            'enabled': True,
            'ma60_slope_uptrend': 0.005,
            'ma60_slope_downtrend': -0.005,
            'uptrend': {'momentum': 0.30, 'trend': 0.25,
                        'volume_price': 0.35, 'rsrs': 0.10},
            'downtrend': {'momentum': 0.05, 'trend': 0.05,
                          'volume_price': 0.30, 'rsrs': 0.60},
            'sideways': {'momentum': 0.25, 'trend': 0.25,
                         'volume_price': 0.25, 'rsrs': 0.25},
        },
    },
    {
        'name': 'C 温和',
        'desc': ('小幅偏离等权：uptrend trend 保持 0.20（趋势确认不激进）、'
                 'rsrs 0.20；downtrend rsrs 0.40、momentum/trend 0.15'
                 '（防守但不极端，假设权重切换本身有成本/风险，'
                 '以最小干预换取攻守分工）。'),
        'weights_by_regime': {
            'enabled': True,
            'ma60_slope_uptrend': 0.005,
            'ma60_slope_downtrend': -0.005,
            'uptrend': {'momentum': 0.25, 'trend': 0.20,
                        'volume_price': 0.35, 'rsrs': 0.20},
            'downtrend': {'momentum': 0.15, 'trend': 0.15,
                          'volume_price': 0.30, 'rsrs': 0.40},
            'sideways': {'momentum': 0.25, 'trend': 0.25,
                         'volume_price': 0.25, 'rsrs': 0.25},
        },
    },
]

# 对照组：生产 config 原样（weights_by_regime.enabled=false → 等权）
BASELINE_NAME = '对照组（等权，生产 config）'


# ── 特征重算（内存中，不落盘）────────────────────────


def rebuild_total_score(base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """仅重算 total_score 列（其余因子列沿用 features_cache）。

    与 compute_all_features 同式（compute_total_score）；cfg 需已带
    weights_by_regime（enabled 视候选/对照组而定）。
    """
    df = base.copy()
    df['total_score'] = compute_total_score(df, cfg)
    return df


# ── walk-forward 验证段执行 ─────────────────────────


def run_val_folds(df: pd.DataFrame, cfg: dict, full_end: str,
                  cost_rate: float) -> list:
    """5 折验证段各跑一次（训练段不选参，生产 config 原样/adaptive 启用）。

    返回 [{fold, range, sharpe, annual, dd, trades, n, metrics}]，
    失败折的数值字段为 None。
    """
    out = []
    for idx, (_train_end, val_start, val_end_def) in enumerate(FOLD_DEFS):
        val_end = val_end_def or full_end
        bt = backtest_metrics(df, cfg, val_start, val_end, cost_rate)
        if bt is None:
            out.append({'fold': idx + 1, 'range': f"{val_start}~{val_end}",
                        'sharpe': None, 'annual': None, 'dd': None,
                        'trades': None, 'n': 0, 'metrics': None})
            continue
        m, n, _eq, _tr = bt
        out.append({'fold': idx + 1, 'range': f"{val_start}~{val_end}",
                    'sharpe': m['sharpe'], 'annual': m['annual_return'],
                    'dd': m['max_drawdown'], 'trades': m['trade_count'],
                    'n': n, 'metrics': m})
    return out


def summarize_folds(folds: list) -> dict:
    """从 5 折结果汇总 OOS 夏普均值/中位/最差折 + 年化/回撤均值。"""
    sh = [f['sharpe'] for f in folds if f['sharpe'] is not None]
    ann = [f['annual'] for f in folds if f['annual'] is not None]
    dd = [f['dd'] for f in folds if f['dd'] is not None]
    if not sh:
        return {'mean': np.nan, 'median': np.nan, 'worst': np.nan,
                'annual_mean': np.nan, 'dd_mean': np.nan,
                'sharpe_list': [], 'n_list': []}
    return {
        'mean': float(np.mean(sh)),
        'median': float(np.median(sh)),
        'worst': float(np.min(sh)),
        'annual_mean': float(np.mean(ann)) if ann else np.nan,
        'dd_mean': float(np.mean(dd)) if dd else np.nan,
        'sharpe_list': sh,
        'n_list': [f['n'] for f in folds if f['sharpe'] is not None],
    }


# ── 报告生成 ────────────────────────────────────────


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:+.1f}%"


def fmt_sharpe(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:+.3f}"


def build_report(args, cfg, sanity, baseline, results, dsr_list,
                 full_end, cost_rate) -> str:
    L = []
    L.append("# 市场状态自适应因子权重（攻守分层）walk-forward 门禁验证报告")
    L.append("")
    L.append(f"**日期**：2026-08-04　**标的**：{cfg['symbol']}　"
             f"**样本**：{sanity['data_start']} ~ {sanity['data_end']}"
             f"（{sanity['n_rows']} 行）　**费率**：{cost_rate}")
    L.append("")
    L.append("## 0. 背景与口径")
    L.append("")
    L.append("- 实证依据（2026-08-04 因子解剖）：熊市段（2023-01~2024-09）"
             "rsrs 是唯一强正预测因子（corr +0.063），trend/momentum 反向"
             "（-0.075/-0.014）；牛市段 volume_price 最强（+0.094）。"
             "当前四因子等权 0.25 浪费了攻守分工。")
    L.append("- 门禁方法：5 折锚定式 walk-forward（与 walk_forward.py 同折方案："
             "训练起点 2023-01-01 固定，验证段各 6 个月，末折到样本末）。")
    L.append("- 验证口径：**生产 config 原样（adaptive_thresholds 启用，与实盘同口径）**，"
             "每套候选直接在 5 折验证段跑，**训练段不选参**"
             "（风控类规则验证口径——权重是结构假设不是调参结果）。")
    L.append("- 特征在内存重算：仅重算 total_score 列（复用 "
             "`compute_features.compute_total_score`，与生产合成式同式），"
             "因子列沿用 features_cache，不写 data/ 下任何文件。")
    L.append("- 对照组：等权 + 生产 config（weights_by_regime.enabled=false），同口径 5 折。")
    L.append(f"- 数据一致性检查：基线（enabled=false）重算 total_score vs "
             f"缓存 total_score 相关系数 {sanity['corr']:.6f}，"
             f"最大绝对差 {sanity['max_abs_diff']:.2e}"
             f"{'（逐位一致）' if sanity['max_abs_diff'] < 1e-9 else '（有差异，见备注）'}。")
    L.append("")
    L.append(f"> 市场状态口径与 signal_rules.get_adaptive_thresholds 一致："
             f"ma60_slope > {cfg['weights_by_regime']['ma60_slope_uptrend']} → uptrend；"
             f"< {cfg['weights_by_regime']['ma60_slope_downtrend']} → downtrend；"
             f"否则 sideways。全样本行分布：uptrend {sanity['regime_dist']['up']}、"
             f"downtrend {sanity['regime_dist']['dn']}、"
             f"sideways {sanity['regime_dist']['side']}。")
    L.append("")

    # ── 候选方案 ──
    L.append("## 1. 候选方案（结构假设，非精细调参）")
    L.append("")
    L.append("| 方案 | 状态 | momentum | trend | volume_price | rsrs | 设计理由 |")
    L.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
    for cand in CANDIDATES:
        rb = cand['weights_by_regime']
        for reg in ['uptrend', 'downtrend', 'sideways']:
            w = rb[reg]
            row = f"| {cand['name'] if reg == 'uptrend' else ''} | {reg} | "
            row += f"{w['momentum']:.2f} | {w['trend']:.2f} | " \
                   f"{w['volume_price']:.2f} | {w['rsrs']:.2f} | "
            row += cand['desc'] if reg == 'uptrend' else ''
            L.append(row)
    L.append("| 对照组 | 全状态 | 0.25 | 0.25 | 0.25 | 0.25 | 生产等权（现状） |")
    L.append("")
    L.append("> sideways 无实证指向，三套候选均保持等权 0.25；"
             "差异仅体现在 uptrend（进攻）与 downtrend（防守）的权重分配。")
    L.append("")

    # ── 5 折 OOS 结果 ──
    L.append("## 2. 5 折 OOS 结果（验证段，生产口径）")
    L.append("")
    L.append("每套方案在 5 折验证段独立回测（空仓起步，与 walk_forward 同引擎限制）：")
    L.append("")
    headers = ["方案", "夏普均值", "夏普中位", "最差折", "年化均值", "回撤均值",
               "F1", "F2", "F3", "F4", "F5"]
    L.append("| " + " | ".join(headers) + " |")
    L.append("|" + "---|" * len(headers))
    all_rows = [('对照', baseline['summary'])] + [(r['name'], r['summary']) for r in results]
    for label, s in all_rows:
        folds = baseline['folds'] if label == '对照' else \
            next(r for r in results if r['name'] == label)['folds']
        fold_sh = [fmt_sharpe(f['sharpe']) for f in folds]
        L.append(f"| {label} | {s['mean']:+.3f} | {s['median']:+.3f} | "
                 f"{s['worst']:+.3f} | {fmt_pct(s['annual_mean'])} | "
                 f"{fmt_pct(s['dd_mean'])} | " + " | ".join(fold_sh) + " |")
    L.append("")
    L.append("> 年化/回撤均为各折验证段独立回测指标（验证段长度约 6 个月），"
             "均值口径仅作横向参考；跨折不连续持仓。")
    L.append("")

    # vs 对照组
    L.append("### 2.1 vs 对照组")
    L.append("")
    L.append("| 方案 | 夏普均值差 | 中位差 | 最差折差 | 年化均值差 | 回撤均值差 |")
    L.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        s = r['summary']
        L.append(f"| {r['name']} | {s['mean'] - baseline['summary']['mean']:+.3f} | "
                 f"{s['median'] - baseline['summary']['median']:+.3f} | "
                 f"{s['worst'] - baseline['summary']['worst']:+.3f} | "
                 f"{fmt_pct(s['annual_mean'] - baseline['summary']['annual_mean'])} | "
                 f"{fmt_pct(s['dd_mean'] - baseline['summary']['dd_mean'])} |")
    L.append("")

    # 分折明细表
    L.append("### 2.2 分折明细")
    L.append("")
    L.append("| 方案 | 折 | 验证段 | 夏普 | 年化 | 回撤 | 交易数 |")
    L.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for label, s in all_rows:
        folds = baseline['folds'] if label == '对照' else \
            next(r for r in results if r['name'] == label)['folds']
        for f in folds:
            L.append(f"| {label} | {f['fold']} | {f['range']} | "
                     f"{fmt_sharpe(f['sharpe'])} | {fmt_pct(f['annual'])} | "
                     f"{fmt_pct(f['dd'])} | {f['trades'] if f['trades'] is not None else '—'} |")
    L.append("")

    # ── DSR ──
    L.append("## 3. Deflated Sharpe Ratio（trials = 3 候选套数，如实计入）")
    L.append("")
    if not dsr_list:
        L.append("有效折 < 2，无法估计 V[SR̂]，DSR 不适用。")
    else:
        L.append(f"- 试错次数 N = **{dsr_list[0]['trials']}**（3 套候选权重方案；"
                 f"对照组是基线，不产生试错负担）。每套候选独立对自身 5 折 OOS 夏普"
                 f"计算 deflated = (SR̄ − SR0) / SE，SE = √((1 + 0.5·SR̄²)/n̄)。")
        L.append("")
        L.append("| 方案 | SR 均值 | n 均值 | V[SR̂] | SR0 | SE | Deflated | p 值(单侧) |")
        L.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for (label, _s), d in zip(all_rows[1:], dsr_list):
            L.append(f"| {label} | {d['mean_sr']:+.3f} | {round(float(np.mean([p['n'] for p in d['per_fold']])))} | "
                     f"{d['var_sr']:.5f} | {d['sr0']:+.4f} | "
                     f"{d['se_mean']:.4f} | {d['mean_deflated']:+.3f} | "
                     f"{d['mean_p']:.4f}{'⚠' if d['mean_p'] < 0.05 else ''} |")
        L.append("")
        best_idx = int(np.argmax([r['summary']['mean'] for r in results]))
        best_d = dsr_list[best_idx]
        L.append(f"- **候选最优（按 OOS 夏普均值，{results[best_idx]['name']}）的 deflated "
                 f"显著性：p = {best_d['mean_p']:.4f}**"
                 f"{'（<0.05，扣除 3 套候选多重检验后仍显著）' if best_d['mean_p'] < 0.05 else '（未达显著，无法排除运气成分）'}")
        L.append("")
        L.append("> DSR 口径与主报告一致（简化版：SR 标准误正态近似，"
                 "V[SR̂] 用各折夏普样本方差，未做 rolling 重叠相关性修正）。"
                 "三套候选与对照组共用同一批 5 折验证段，折间样本非独立，"
                 "deflated 夏普带有乐观倾向，应视为本验证协议下的上限估计。")
    L.append("")

    # ── 结论 ──
    L.append("## 4. 结论与建议")
    L.append("")
    best = max(results, key=lambda r: r['summary']['mean'])
    bs = best['summary']
    bm = baseline['summary']
    delta = bs['mean'] - bm['mean']
    L.append(f"- 三套候选 OOS 夏普均值："
             + "；".join(f"{r['name']} {r['summary']['mean']:+.3f}"
                        for r in results)
             + f"；对照组 **{bm['mean']:+.3f}**。")
    L.append(f"- 最优候选 **{best['name']}**：夏普均值 {bs['mean']:+.3f}"
             f"（vs 对照 {delta:+.3f}），中位 {bs['median']:+.3f}，"
             f"最差折 {bs['worst']:+.3f}，年化均值 {fmt_pct(bs['annual_mean'])}，"
             f"回撤均值 {fmt_pct(bs['dd_mean'])}。")
    L.append("")

    # 采纳判定
    improve = delta > 0.05
    median_ok = bs['median'] >= bm['median'] - 0.05
    worst_ok = bs['worst'] >= bm['worst'] - 0.05
    sig = dsr_list and dsr_list[int(np.argmax([r['summary']['mean'] for r in results]))]['mean_p'] < 0.05
    if improve and median_ok and worst_ok and sig:
        L.append(f"**判定：建议采纳 {best['name']}** —— OOS 夏普均值较对照改善 "
                 f"{delta:+.3f}（≥+0.05），中位/最差折不劣于对照，"
                 f"且 DSR p = {dsr_list[int(np.argmax([r['summary']['mean'] for r in results]))]['mean_p']:.4f} < 0.05（扣除 3 套候选试错后仍显著）。")
    elif improve and median_ok and worst_ok:
        L.append(f"**判定：数值占优但不显著，建议小步试改并观察** —— {best['name']} "
                 f"OOS 夏普均值较对照改善 {delta:+.3f}，中位/最差折不劣于对照，"
                 f"但 DSR p = {dsr_list[int(np.argmax([r['summary']['mean'] for r in results]))]['mean_p']:.4f} ≥ 0.05，"
                 f"无法排除运气成分；可将该套权重置入 config 试跑 1~2 个月实盘口径跟踪。")
    else:
        L.append(f"**判定：不建议采纳任一候选** —— 最优 {best['name']} 相对对照"
                 f"{'夏普均值无 ≥+0.05 改善' if not improve else '中位/最差折存在劣化风险'}，"
                 f"攻守分层未通过 walk-forward 门禁；建议保留等权现状，"
                 f"转向其他改进维度（如阈值/窗口/风控参数，见既有 walk-forward 报告）。")
    L.append("")
    L.append("> 若主要目标是**防御价值**（熊市折少亏/回撤更小）而非夏普均值，"
             "请直接比较 2.2 分折明细中 downtrend 占比高的折（如折 2、折 5）"
             "各方案的夏普与回撤——防守权重可能在夏普均值上不显著、"
             "但在尾部折上有结构性改善。")
    L.append("")
    L.append("---")
    L.append("*本报告由 tools/daily_pipeline/regime_weight_search.py 生成"
             "（纯只读分析，未修改任何生产代码或数据文件；"
             "生产 config 的 weights_by_regime.enabled 保持 false，"
             "等权现状不受影响）。*")
    L.append("")
    return "\n".join(L)


# ── 主流程 ───────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description='市场状态自适应因子权重 walk-forward 门禁验证（纯只读）')
    ap.add_argument('--symbol', default=None, help='标的（默认取 config）')
    ap.add_argument('--cost', type=float, default=0.00055, help='单边费率')
    ap.add_argument('--end', default=None,
                    help='样本终点（默认取特征数据最后日期）')
    ap.add_argument('--report', default=str(DEFAULT_REPORT), help='报告输出路径')
    args = ap.parse_args()

    cfg = load_config()
    symbol = args.symbol or cfg['symbol']

    print("═══ 市场状态自适应因子权重 walk-forward 门禁验证 ═══")
    print(f"标的 {symbol}，费率 {args.cost}")

    base = load_features_df(symbol)
    full_end = args.end or base['date'].max().strftime('%Y-%m-%d')

    # sanity：基线（enabled=false）重算 vs 缓存 total_score
    rebuilt = rebuild_total_score(base, cfg)
    merged = rebuilt[['date', 'total_score']].merge(
        base[['date', 'total_score']].rename(
            columns={'total_score': 'cached_total'}), on='date')
    corr = float(merged['total_score'].corr(merged['cached_total']))
    max_abs_diff = float(
        (merged['total_score'] - merged['cached_total']).abs().max())
    print(f"  [sanity] 基线重算 vs 缓存 total_score：corr {corr:.6f}，"
          f"最大绝对差 {max_abs_diff:.2e}")

    # 市场状态行分布（与 compute_total_score 生效口径一致：
    # ma60_slope 为 NaN 的行落入 sideways 分支）
    rb = cfg['weights_by_regime']
    slope = base['ma60_slope']
    n_up = int((slope > rb['ma60_slope_uptrend']).sum())
    n_dn = int((slope < rb['ma60_slope_downtrend']).sum())
    dist = {
        'up': n_up,
        'dn': n_dn,
        'side': len(slope) - n_up - n_dn,  # 含 NaN 兜底行
    }
    print(f"  [regime] 全样本状态分布：uptrend {dist['up']} / "
          f"downtrend {dist['dn']} / sideways {dist['side']}")

    # 1) 对照组：生产 config 原样（等权）
    print("\n── 对照组（等权，生产 config）5 折验证段 ──")
    base_folds = run_val_folds(base, cfg, full_end, args.cost)
    base_summary = summarize_folds(base_folds)
    print(f"  OOS 夏普 均值 {base_summary['mean']:+.3f}  "
          f"中位 {base_summary['median']:+.3f}  "
          f"最差 {base_summary['worst']:+.3f}  "
          f"年化均值 {fmt_pct(base_summary['annual_mean'])}  "
          f"回撤均值 {fmt_pct(base_summary['dd_mean'])}")

    # 2) 候选套数：内存重算特征 → 生产口径 5 折验证段
    results = []
    for cand in CANDIDATES:
        print(f"\n── {cand['name']} ──")
        cfg_cand = copy.deepcopy(cfg)
        cfg_cand['weights_by_regime'] = copy.deepcopy(
            cand['weights_by_regime'])
        df_cand = rebuild_total_score(base, cfg_cand)
        folds = run_val_folds(df_cand, cfg_cand, full_end, args.cost)
        summ = summarize_folds(folds)
        results.append({'name': cand['name'], 'desc': cand['desc'],
                        'cfg': cfg_cand, 'folds': folds, 'summary': summ})
        print(f"  OOS 夏普 均值 {summ['mean']:+.3f}  "
              f"中位 {summ['median']:+.3f}  最差 {summ['worst']:+.3f}  "
              f"年化均值 {fmt_pct(summ['annual_mean'])}  "
              f"回撤均值 {fmt_pct(summ['dd_mean'])}")

    # 3) DSR：每套候选对自身 5 折 OOS 夏普独立计算，trials = 3（候选套数）
    print("\n── DSR ──")
    trials = len(CANDIDATES)
    dsr_list = []
    for r in results:
        d = compute_dsr(r['summary']['sharpe_list'],
                        [float(x) if x and x > 0 else 1.0 for x in r['summary']['n_list']],
                        trials)
        if d is not None:
            d['name'] = r['name']
            dsr_list.append(d)
            print(f"  {r['name']:<12s} SR̄={d['mean_sr']:+.3f} SR0={d['sr0']:+.4f} "
                  f"deflated={d['mean_deflated']:+.3f} p={d['mean_p']:.4f}")
        else:
            print(f"  {r['name']:<12s} [WARN] 有效折 < 2，DSR 无法计算")

    # 4) 报告
    print("\n[生成报告]")
    report = build_report(args, cfg, {
        'corr': corr, 'max_abs_diff': max_abs_diff,
        'data_start': base['date'].min().strftime('%Y-%m-%d'),
        'data_end': base['date'].max().strftime('%Y-%m-%d'),
        'n_rows': len(base), 'regime_dist': dist,
    }, {'summary': base_summary, 'folds': base_folds}, results, dsr_list,
        full_end, args.cost)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    print(f"✅ 报告已写入: {report_path}")
    print("\n" + "=" * 62)
    print(report)
    print("=" * 62)


if __name__ == '__main__':
    main()
