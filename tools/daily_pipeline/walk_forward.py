#!/usr/bin/env python3
"""
walk_forward.py — walk-forward 多折滚动回测 + Deflated Sharpe Ratio + 参数高原分析

纯只读分析工具：
  - 不修改 backtest.py / compute_features.py / daily_panel.py / fetch_data.py /
    config.json 等任何现有文件，不写 data/ 下任何数据文件。
  - 复用 backtest 模块的 run_backtest / compute_metrics / load_features_df，
    不重复实现回测引擎。
  - 依赖仅 numpy / pandas + Python 标准库。

用法：
  python3 tools/daily_pipeline/walk_forward.py
  python3 tools/daily_pipeline/walk_forward.py --symbol 588000 --cost 0.00055 --trials 105

重要假设（adaptive_thresholds 处理）：
  生产 config.json 中 adaptive_thresholds.enabled=true，signal_rules.get_adaptive_thresholds()
  会在运行期用 uptrend/sideways/downtrend 阈值覆盖 config['thresholds'] 的 buy/sell，
  导致网格搜索的 buy_th 参数实际不生效。为使本工具的 walk-forward 选参、DSR 与
  高原分析真正衡量 buy_th 的影响，本工具在每次回测前将配置拷贝的
  adaptive_thresholds.enabled 置为 False（其余字段照抄 config.json）。
  报告末尾附「生产配置（adaptive 启用）」的全段表现作参照，两者口径不同。
"""

import argparse
import contextlib
import copy
import io
import json
import math
import sys
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "walk_forward_report_2026-08-04.md"

sys.path.insert(0, str(SCRIPT_DIR))
from backtest import load_features_df, run_backtest, compute_metrics  # noqa: E402

# ── 常量 ────────────────────────────────────────────────

TRAIN_START = '2023-01-01'                  # 锚定式滚动：训练起点固定
BUY_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
CONFIRM_CANDIDATES = [1, 2, 3]
SELL_RATIO = -0.67
PLATEAU_BUYS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PLATEAU_CONFIRM = 2
PLATEAU_CENTER = 0.10                        # 当前生产 buy 阈值（高原分析中心）
PLATEAU_TOL = 0.15                           # 相邻参数绩效相对差异 < 15% 视为高原

# 折定义：(训练终点, 验证起点, 验证终点)；验证终点 None = 今天
FOLD_DEFS = [
    ('2024-06-30', '2024-07-01', '2024-12-31'),
    ('2024-12-31', '2025-01-01', '2025-06-30'),
    ('2025-06-30', '2025-07-01', '2025-12-31'),
    ('2025-12-31', '2026-01-01', '2026-06-30'),
    ('2026-06-30', '2026-07-01', None),
]

EULER_GAMMA = 0.5772156649015329            # γ：期望最大夏普阈值中的欧拉常数
NORMAL = NormalDist()                        # 标准正态分布（Z^{-1} / CDF）


# ── 配置工具 ────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def build_config(base: dict, buy_th: float, confirm_days: int) -> dict:
    """拷贝生产 config，覆盖 thresholds 的 buy/sell/confirm_days，禁用 adaptive。

    仅临时用于本次回测（内存中拷贝），不落盘、不改生产文件。
    """
    cfg = copy.deepcopy(base)
    sell_th = round(buy_th * SELL_RATIO, 2)
    if sell_th == 0:
        sell_th = 0.0
    cfg['thresholds'] = {
        'buy': float(buy_th),
        'sell': sell_th,
        'confirm_days': int(confirm_days),
        'weekly_filter_percentile': base['thresholds'].get('weekly_filter_percentile', 0.2),
    }
    # 关键假设：禁用 adaptive，使 buy/sell 网格真正生效（见模块 docstring）
    cfg.setdefault('adaptive_thresholds', {})
    cfg['adaptive_thresholds'] = copy.deepcopy(cfg['adaptive_thresholds'])
    cfg['adaptive_thresholds']['enabled'] = False
    return cfg


# ── 回测包装 ────────────────────────────────────────────

def run_silent(features_df: pd.DataFrame, config: dict, start: str, end: str,
               cost_rate: float) -> dict:
    """调用 run_backtest，但吞掉其内部 INFO 打印（大量网格回测时保持输出整洁）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_backtest(features_df, config, start, end, cost_rate)
    return result


def backtest_metrics(features_df: pd.DataFrame, config: dict, start: str, end: str,
                     cost_rate: float) -> tuple:
    """跑一次回测并返回 (metrics, n_days, equity_df, trades_df)。失败返回 None。"""
    try:
        res = run_silent(features_df, config, start, end, cost_rate)
        eq = res['equity_curve']
        if len(eq) < 5:
            return None
        m = compute_metrics(eq['equity'], trades_df=res['trades'], n_days=len(eq))
        return m, len(eq), eq, res['trades']
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] 回测失败 {start}~{end}: {e}")
        return None


# ── 需求 1：walk-forward 多折滚动 ──────────────────────

def grid_search(features_df: pd.DataFrame, config: dict, start: str, end: str,
                cost_rate: float) -> dict:
    """训练段网格搜索：buy_th × confirm_days，选训练段夏普最高组合。"""
    grid = []
    for buy_th in BUY_CANDIDATES:
        for cd in CONFIRM_CANDIDATES:
            cfg = build_config(config, buy_th, cd)
            bt = backtest_metrics(features_df, cfg, start, end, cost_rate)
            if bt is None:
                continue
            m = bt[0]
            grid.append({'buy_th': buy_th, 'sell_th': round(buy_th * SELL_RATIO, 2),
                         'confirm_days': cd, 'sharpe': m['sharpe'],
                         'annual_return': m['annual_return'], 'max_drawdown': m['max_drawdown'],
                         'trade_count': m['trade_count']})
    if not grid:
        raise RuntimeError(f"训练段 {start}~{end} 网格搜索全部失败")
    return max(grid, key=lambda g: g['sharpe'])   # 并列时取先遇到的（最小 buy_th）


def run_walk_forward(features_df: pd.DataFrame, config: dict, full_end: str,
                     cost_rate: float) -> tuple:
    """执行 5 折锚定式 walk-forward，返回 (folds, is_scores)。"""
    folds = []
    is_scores = []          # 每折所选参数的全段 sharpe（in-sample）
    is_cache = {}           # 参数 → 全段指标缓存

    for idx, (train_end, val_start, val_end_def) in enumerate(FOLD_DEFS):
        val_end = val_end_def or full_end
        print(f"\n  ── 折 {idx + 1}/{len(FOLD_DEFS)}  训练 ~{train_end} → 验证 {val_start}~{val_end} ──")

        # 训练段网格搜索（21 组合）
        best = grid_search(features_df, config, TRAIN_START, train_end, cost_rate)
        print(f"  [grid] 训练段最优: buy={best['buy_th']:.2f} sell={best['sell_th']:.2f} "
              f"confirm={best['confirm_days']} → 训练夏普 {best['sharpe']:.4f}")

        # 参数冻结 → 验证段
        cfg = build_config(config, best['buy_th'], best['confirm_days'])
        vb = backtest_metrics(features_df, cfg, val_start, val_end, cost_rate)
        if vb is None:
            print(f"  [ERR] 折 {idx + 1} 验证段无有效结果，跳过")
            continue
        vm, vn, veq, vtr = vb

        folds.append({
            'fold': idx + 1,
            'train_range': f"{TRAIN_START}~{train_end}",
            'val_range': f"{val_start}~{val_end}",
            'val_actual': f"{veq['date'].iloc[0].date()}~{veq['date'].iloc[-1].date()}",
            'buy_th': best['buy_th'], 'sell_th': best['sell_th'],
            'confirm_days': best['confirm_days'],
            'train_sharpe': best['sharpe'],
            'train_annual': best['annual_return'],
            'val_sharpe': vm['sharpe'],
            'val_annual': vm['annual_return'],
            'val_dd': vm['max_drawdown'],
            'val_trades': vm['trade_count'],
            'val_win': vm['win_rate'],
            'val_n': vn,
        })
        print(f"  [val]   验证段: 夏普 {vm['sharpe']:.4f}  年化 {vm['annual_return']:+.2f}%  "
              f"回撤 {vm['max_drawdown']:.2f}%  交易 {vm['trade_count']}")

        # in-sample：该折所选参数在全段跑一次
        key = (best['buy_th'], best['confirm_days'])
        if key not in is_cache:
            icfg = build_config(config, key[0], key[1])
            ibt = backtest_metrics(features_df, icfg, TRAIN_START, full_end, cost_rate)
            is_cache[key] = ibt[0] if ibt else None
        im = is_cache[key]
        is_scores.append(im['sharpe'] if im else None)

    return folds, is_scores


# ── 需求 2：DSR（Deflated Sharpe Ratio，López de Prado 2018 简化） ──

def compute_dsr(sharpe_list: list, n_list: list, trials: int) -> dict | None:
    """
    SR0 = sqrt(V[SR_hat]) * ((1-γ)·Z^{-1}(1-1/N) + γ·Z^{-1}(1-1/(N·e)))
    每折 deflated 夏普 = (SR_i - SR0) / SE_i，SE_i = sqrt((1 + 0.5·SR_i²) / n_i)
    p 值 = 单侧正态 1 - Φ(deflated)
    N=1 时退化为普通显著性（SR0=0）。
    """
    # 有效折 < 2 时无法估计样本方差 V[SR̂]（ddof=1 至少需 2 个样本），
    # 返回 None 作为哨兵值，由调用方决定如何处理
    if len(sharpe_list) < 2:
        return None

    sr = np.asarray(sharpe_list, dtype=float)
    n = np.asarray(n_list, dtype=float)
    n = np.where(n > 0, n, 1.0)

    mean_sr = float(sr.mean())
    var_sr = float(sr.var(ddof=1)) if len(sr) > 1 else 0.0
    sd_sr = math.sqrt(var_sr)

    if trials <= 1:
        sr0 = 0.0
        trials_note = "N=1 → 退化为普通显著性（SR0=0）"
    else:
        z1 = NORMAL.inv_cdf(1 - 1.0 / trials)
        z2 = NORMAL.inv_cdf(1 - 1.0 / (trials * math.e))
        sr0 = sd_sr * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
        trials_note = f"SR0 = √V·[{(1 - EULER_GAMMA):.4f}·Z⁻¹(1-1/{trials}) + {EULER_GAMMA:.4f}·Z⁻¹(1-1/({trials}·e))]"

    per_fold = []
    for i in range(len(sr)):
        se = math.sqrt((1 + 0.5 * sr[i] ** 2) / n[i])
        d = (sr[i] - sr0) / se
        per_fold.append({'sr': float(sr[i]), 'n': int(n[i]), 'se': se,
                         'deflated': d, 'p': 1.0 - NORMAL.cdf(d)})

    se_mean = math.sqrt((1 + 0.5 * mean_sr ** 2) / float(n.mean()))
    d_mean = (mean_sr - sr0) / se_mean
    p_mean = 1.0 - NORMAL.cdf(d_mean)

    return {
        'trials': trials, 'mean_sr': mean_sr, 'var_sr': var_sr, 'sd_sr': sd_sr,
        'sr0': sr0, 'trials_note': trials_note,
        'per_fold': per_fold, 'se_mean': se_mean,
        'mean_deflated': d_mean, 'mean_p': p_mean,
    }


# ── 需求 3：参数高原分析 ──────────────────────────────

def plateau_analysis(features_df: pd.DataFrame, config: dict, full_end: str,
                     cost_rate: float) -> dict:
    """全段数据，buy_th ∈ {0.0..0.30}（sell 联动 -0.67，confirm=2）逐一全段回测。"""
    rows = []
    for buy_th in PLATEAU_BUYS:
        cfg = build_config(config, buy_th, PLATEAU_CONFIRM)
        bt = backtest_metrics(features_df, cfg, TRAIN_START, full_end, cost_rate)
        if bt is None:
            continue
        m = bt[0]
        rows.append({'buy_th': buy_th, 'sell_th': round(buy_th * SELL_RATIO, 2),
                     'sharpe': m['sharpe'], 'annual': m['annual_return'],
                     'dd': m['max_drawdown'], 'trades': m['trade_count']})

    pairs = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        denom = max(abs(a['sharpe']), abs(b['sharpe']), 1e-9)
        rel = abs(a['sharpe'] - b['sharpe']) / denom
        flat = rel < PLATEAU_TOL
        pairs.append({'lo': a['buy_th'], 'hi': b['buy_th'], 'rel_diff': rel, 'flat': flat})

    flat_n = sum(1 for p in pairs if p['flat'])
    overall_flat = flat_n >= math.ceil(len(pairs) / 2)
    verdict = '参数高原（稳健，对阈值选择不敏感）' if overall_flat else '参数尖峰（过拟合敏感，阈值微调即显著变差）'

    # 当前生产参数（buy=0.10）两侧相邻对是否平滑
    center_ok = True
    center_notes = []
    for p in pairs:
        if p['lo'] == PLATEAU_CENTER or p['hi'] == PLATEAU_CENTER:
            center_notes.append(p)
            center_ok = center_ok and p['flat']
    center_verdict = ('当前生产 buy=0.10 位于高原（稳健）'
                      if center_ok else '当前生产 buy=0.10 处于敏感区（相邻参数绩效差异大）')

    return {'rows': rows, 'pairs': pairs, 'flat_n': flat_n,
            'total_pairs': len(pairs), 'overall_flat': overall_flat,
            'verdict': verdict, 'center_ok': center_ok,
            'center_verdict': center_verdict}


# ── 报告输出 ────────────────────────────────────────────

def fmt_pct(v: float, sign: bool = True) -> str:
    s = f"{v:+.2f}%" if sign else f"{v:.2f}%"
    return s


def md_table(headers: list, rows: list) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join([':---:'] * len(headers)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(lines)


def build_report(args, config, features_df, folds, is_scores, dsr, plateau,
                 prod_ref, full_end) -> str:
    L = []
    A = L.append
    A(f"# walk-forward 多折滚动回测 + Deflated Sharpe Ratio + 参数高原分析")
    A(f"")
    A(f"- 生成日期：{date.today().isoformat()}（数据截至 {features_df['date'].max().date()}，"
      f"共 {len(features_df)} 个交易日）")
    A(f"- 标的：{args.symbol}　费率：单边 {args.cost}（{args.cost * 10000:.1f}‱）")
    A(f"- DSR 试错次数 N = {args.trials}（7 buy_th × 3 confirm_days × 5 折）")
    A(f"- 运行命令：`python3 tools/daily_pipeline/walk_forward.py"
      f" --symbol {args.symbol} --cost {args.cost} --trials {args.trials}`")
    A(f"")
    A(f"> **口径说明**：网格搜索与高原分析在内存中拷贝 config 并置 "
      f"`adaptive_thresholds.enabled=false`，使 buy/sell 网格参数真正生效"
      f"（生产 config 的 adaptive 机制会在运行期覆盖 buy/sell，见 `signal_rules.get_adaptive_thresholds`）。"
      f"其余字段照抄 config.json。")

    # ── 需求 1 ──
    A(f"")
    A(f"## 1. Walk-Forward 多折滚动（锚定式）")
    A(f"")
    A(f"训练起点固定 {TRAIN_START}，终点每折推进；验证段紧随训练终点、长度约 6 个月。")
    A(f"每折仅在训练段网格搜索（buy_th ∈ {BUY_CANDIDATES}，sell_th = round(buy_th×{SELL_RATIO},2)，"
      f"confirm_days ∈ {CONFIRM_CANDIDATES}），按训练段夏普最高选参，参数冻结后在验证段回测。")
    A(f"")
    rows = []
    for f in folds:
        win = f"{f['val_win']:.1f}%" if f['val_trades'] > 0 else '—'
        rows.append([f"折 {f['fold']}", f['train_range'], f['val_actual'],
                     f"{f['buy_th']:.2f}", f"{f['sell_th']:.2f}", f"{f['confirm_days']}",
                     f"{f['train_sharpe']:.3f}", f"{f['val_sharpe']:.3f}",
                     fmt_pct(f['val_annual']), fmt_pct(f['val_dd'], sign=False),
                     f"{f['val_trades']}", win, f"{f['val_n']}"])
    A(md_table(['折', '训练段', '验证段', 'buy', 'sell', '确认', '训练夏普',
                '验证夏普', '验证年化', '验证回撤', '交易数', '胜率', '验证天数'], rows))
    A(f"")
    A(f"注：验证段为独立回测（空仓起步，不延续上一折持仓——run_backtest 引擎限制）；"
      f"折 5 验证段仅约 1 个月（数据截至 {features_df['date'].max().date()}），样本最少。")
    neg_train = [f"折 {f['fold']}（{f['train_sharpe']:.3f}）" for f in folds if f['train_sharpe'] < 0]
    if neg_train:
        A(f"")
        A(f"⚠ 训练段最优夏普为负的折：{', '.join(neg_train)}——该训练段内全部 21 个参数组合均无"
          f"正收益（多为单边下跌市），此时选出的『最优』参数接近于噪声，验证段表现更多反映行情而非参数质量。")

    # 汇总
    v_sharpes = [f['val_sharpe'] for f in folds]
    v_annuals = [f['val_annual'] for f in folds]
    v_dds = [f['val_dd'] for f in folds]
    A(f"")
    A(f"**OOS（验证段）绩效分布：**")
    A(f"")
    A(md_table(['指标', '均值', '中位数', '最差折', '最佳折'], [
        ['夏普', f"{np.mean(v_sharpes):.3f}", f"{np.median(v_sharpes):.3f}",
         f"{min(v_sharpes):.3f}", f"{max(v_sharpes):.3f}"],
        ['年化', fmt_pct(np.mean(v_annuals)), fmt_pct(np.median(v_annuals)),
         fmt_pct(min(v_annuals)), fmt_pct(max(v_annuals))],
        ['回撤', fmt_pct(np.mean(v_dds), sign=False), fmt_pct(np.median(v_dds), sign=False),
         fmt_pct(max(v_dds), sign=False), fmt_pct(min(v_dds), sign=False)],
    ]))
    A(f"")
    # IS/OOS 比值只在「两者都有效的折」上计算：某折 IS 回测失败（is_scores=None）时，
    # 同折的 val_sharpe 不再进入 OOS 均值，保证 IS 与 OOS 人口一致，比值不因人口错位失真
    paired = [(f['val_sharpe'], s) for f, s in zip(folds, is_scores) if s is not None]
    if paired:
        oos_v, is_v = zip(*paired)
        is_mean = float(np.mean(is_v))
        oos_mean = float(np.mean(oos_v))
        ratio = oos_mean / is_mean if is_mean > 1e-9 else float('nan')
        ratio_s = f"{ratio:.2f}" if ratio == ratio else "N/A（IS≤0）"
        A(f"**in-sample vs out-of-sample 衰减：**")
        A(f"")
        A(f"- IS 平均（每折所选参数在全段回测的夏普均值）：`{is_mean:.3f}`（{len(is_v)} 折有效）")
        A(f"- OOS 平均（与 IS 同折的验证段夏普均值）：`{oos_mean:.3f}`（{len(oos_v)} 折有效）")
        A(f"- **OOS/IS 比值：`{ratio_s}`**（<1 表示样本外衰减；≤0 表示样本外无正超额）")
    A(f"")

    # ── 需求 2 ──
    A(f"## 2. Deflated Sharpe Ratio（López de Prado 2018 简化实现）")
    A(f"")
    A(f"- 试错次数 N = {dsr['trials']}")
    A(f"- 各折夏普样本方差 V[SR̂] = {dsr['var_sr']:.5f}（√V = {dsr['sd_sr']:.4f}）")
    A(f"- 期望最大夏普阈值 SR0 = **{dsr['sr0']:.4f}**")
    A(f"- 公式：{dsr['trials_note']}")
    A(f"- 每折 deflated = (SR_i − SR0) / SE_i，SE_i = √((1 + 0.5·SR_i²)/n_i)，p 为单侧正态")
    A(f"")
    d_rows = []
    for i, (f, pf) in enumerate(zip(folds, dsr['per_fold'])):
        d_rows.append([f"折 {f['fold']}", f"{pf['sr']:.3f}", f"{pf['n']}",
                       f"{pf['se']:.4f}", f"{pf['deflated']:+.3f}",
                       f"{pf['p']:.4f}" + ("⚠" if pf['p'] < 0.05 else "")])
    d_rows.append(['均值(合并)', f"{dsr['mean_sr']:.3f}", f"{round(float(np.mean([f['val_n'] for f in folds])))}",
                   f"{dsr['se_mean']:.4f}", f"{dsr['mean_deflated']:+.3f}",
                   f"{dsr['mean_p']:.4f}" + ("⚠" if dsr['mean_p'] < 0.05 else "")])
    A(md_table(['折', 'SR_i', 'n_i', 'SE_i', 'Deflated', 'p 值(单侧)'], d_rows))
    A(f"")
    neg_folds = [f"折 {f['fold']}（SR={f['val_sharpe']:.3f}）" for f in folds if f['val_sharpe'] < 0]
    A(f"- 负夏普折：{', '.join(neg_folds) if neg_folds else '无'}")
    A(f"- **均值夏普的 deflated 显著性：p = {dsr['mean_p']:.4f}**"
      f"{'（<0.05，扣除多重检验后仍显著）' if dsr['mean_p'] < 0.05 else '（未达显著，存在过拟合/运气成分的可能）'}")
    A(f"")
    A(f"> 说明：本实现为简化版——SR 标准误用正态近似（López de Prado 原文含偏度/峰度修正项），"
      f"V[SR̂] 用各折夏普样本方差，SR0 未考虑各折非独立（rolling 窗口重叠）带来的相关性修正。")
    A(f"")

    # ── 需求 3 ──
    A(f"## 3. 参数高原分析（全段 {TRAIN_START} ~ {full_end}）")
    A(f"")
    A(f"以生产 buy=0.10 为中心，confirm_days=2，sell 联动 -0.67。"
      f"相邻参数绩效相对差异 < {PLATEAU_TOL:.0%} 判为高原（平滑）。")
    A(f"")
    p_rows = [[f"{r['buy_th']:.2f}", f"{r['sell_th']:.2f}", f"{r['sharpe']:.4f}",
               fmt_pct(r['annual']), fmt_pct(r['dd'], sign=False), f"{r['trades']}"]
              for r in plateau['rows']]
    A(md_table(['buy_th', 'sell_th', '夏普', '年化', '回撤', '交易数'], p_rows))
    A(f"")
    A(md_table(['相邻参数对', '夏普相对差异', '判定'], [
        [f"{p['lo']:.2f}↔{p['hi']:.2f}", f"{p['rel_diff']:.1%}",
         '高原（差异<15%）' if p['flat'] else '尖峰（差异≥15%）']
        for p in plateau['pairs']]))
    A(f"")
    A(f"- 平滑相邻对 {plateau['flat_n']}/{plateau['total_pairs']}")
    A(f"- **整体判定：{plateau['verdict']}**")
    A(f"- **{plateau['center_verdict']}**")
    A(f"")

    # ── 生产配置参照 ──
    if prod_ref:
        A(f"## 附：生产配置（adaptive 启用）全段表现（口径参照）")
        A(f"")
        A(md_table(['口径', '夏普', '年化', '回撤', '交易数'], [[
            '生产 config（adaptive_thresholds.enabled=true）',
            f"{prod_ref['sharpe']:.4f}", fmt_pct(prod_ref['annual']),
            fmt_pct(prod_ref['dd'], sign=False), f"{prod_ref['trades']}",
        ]]))
        A(f"")
        A(f"> 生产口径下 buy/sell 在运行期被 adaptive 阈值覆盖（uptrend/sideways=0.1、downtrend=0.15），"
          f"与本工具网格搜索口径（adaptive 禁用）不可直接对比。")
        A(f"")

    A(f"---")
    A(f"*本报告由 walk_forward.py 自动生成（纯只读分析，未修改任何生产代码或数据文件）。*")
    return '\n'.join(L)


# ── 主入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='walk-forward 多折回测 + DSR + 参数高原分析')
    parser.add_argument('--symbol', default='588000')
    parser.add_argument('--cost', type=float, default=0.00055, help='单边费率，默认万5.5')
    parser.add_argument('--trials', type=int, default=105,
                        help='DSR 试错次数 N（默认 7×3×5=105）')
    parser.add_argument('--report', default=str(DEFAULT_REPORT), help='报告输出路径')
    args = parser.parse_args()

    print("=" * 62)
    print("  walk-forward 多折滚动回测 + Deflated Sharpe Ratio + 参数高原")
    print(f"  标的 {args.symbol} | 费率 {args.cost} | DSR trials={args.trials}")
    print("=" * 62)

    config = load_config()
    features_df = load_features_df(args.symbol)
    full_end = date.today().strftime('%Y-%m-%d')
    print(f"  [OK] 特征 {len(features_df)} 条 "
          f"({features_df['date'].min().date()} ~ {features_df['date'].max().date()})")
    print("  [i] 网格搜索/高原分析置 adaptive_thresholds.enabled=false（见 docstring）")

    # 1) walk-forward
    print("\n[1/3] walk-forward 多折滚动...")
    folds, is_scores = run_walk_forward(features_df, config, full_end, args.cost)
    if not folds:
        print("  [ERR] 所有折验证段均无有效结果，无法计算 DSR 与汇总，程序终止。")
        return
    if len(folds) < 5:
        print(f"  [WARN] 有效折数 {len(folds)}/5，DSR 与汇总基于现有折")

    # 2) DSR
    print("\n[2/3] Deflated Sharpe Ratio...")
    sharpe_list = [f['val_sharpe'] for f in folds]
    n_list = [f['val_n'] for f in folds]
    dsr = compute_dsr(sharpe_list, n_list, args.trials)
    if dsr is None:
        print("  [ERR] 有效折数 < 2，无法计算 DSR（样本方差 V[SR̂] 需 ≥2 折），程序终止。")
        return
    print(f"  SR0 = {dsr['sr0']:.4f} (√V={dsr['sd_sr']:.4f}, N={dsr['trials']})，"
          f"均值 deflated = {dsr['mean_deflated']:+.3f}，p = {dsr['mean_p']:.4f}")

    # 3) 高原分析
    print("\n[3/3] 参数高原分析...")
    plateau = plateau_analysis(features_df, config, full_end, args.cost)
    print(f"  平滑相邻对 {plateau['flat_n']}/{plateau['total_pairs']} → {plateau['verdict']}")

    # 生产配置参照（adaptive 启用，原样 config）
    print("\n  生产配置（adaptive 启用）全段参照...")
    prod_ref = None
    pb = backtest_metrics(features_df, config, TRAIN_START, full_end, args.cost)
    if pb:
        pm = pb[0]
        prod_ref = {'sharpe': pm['sharpe'], 'annual': pm['annual_return'],
                    'dd': pm['max_drawdown'], 'trades': pm['trade_count']}
        print(f"  [ref] 生产口径: 夏普 {pm['sharpe']:.4f}  年化 {pm['annual_return']:+.2f}%  "
              f"回撤 {pm['max_drawdown']:.2f}%")

    # 报告
    print("\n[生成报告]")
    report = build_report(args, config, features_df, folds, is_scores, dsr, plateau,
                          prod_ref, full_end)
    print("\n" + "=" * 62)
    print(report)
    print("=" * 62)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding='utf-8')
    print(f"\n  [SAVE] 报告 → {report_path}")


if __name__ == '__main__':
    main()
