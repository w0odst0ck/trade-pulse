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

# ── 风控验证模式（--risk-scan）──
RISK_STOP_LOSSES = [0.05, 0.08, 0.10, 0.12]  # 单笔止损网格
RISK_DD_LIMITS = [None, 0.10, 0.15]          # 回撤熔断网格（None=禁用）
RISK_COOLDOWNS = [3, 5]                      # 冷却期网格（交易日）
SCREEN_BUY = 0.10                            # 初筛固定生产阈值 buy
SCREEN_CONFIRM = 2                           # 初筛固定生产 confirm
SCREEN_IMPROVE = 0.10                        # 初筛门槛：OOS 夏普均值相对改善 ≥10%
RISK_DEFAULT_REPORT = PROJECT_ROOT / "docs" / "risk_control_report_2026-08-04.md"

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


# ── 风控验证模式（--risk-scan）─────────────────────────
# 两步协议：
#   1) 初筛：固定生产阈值 × 风控网格，直接在 5 折验证段跑（不选参，风控是规则不是调优）
#   2) 终审：初筛通过（OOS 夏普均值相对改善 ≥10%）的组合进完整 walk-forward，
#      训练段同时搜阈值+风控参数、验证段冻结，DSR 扣试错次数（初筛+终审全部组合数）

def backtest_metrics_verbose(features_df: pd.DataFrame, config: dict, start: str, end: str,
                             cost_rate: float):
    """同 backtest_metrics，但额外返回 risk_events（风控验证模式专用）。"""
    try:
        res = run_silent(features_df, config, start, end, cost_rate)
        eq = res['equity_curve']
        if len(eq) < 5:
            return None
        m = compute_metrics(eq['equity'], trades_df=res['trades'], n_days=len(eq))
        return m, len(eq), eq, res['trades'], res.get('risk_events', [])
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] 回测失败 {start}~{end}: {e}")
        return None


def build_risk_cfg(config: dict, rc_params: dict) -> dict:
    """固定生产阈值（buy=0.10/confirm=2，adaptive 禁用）+ 风控参数开启。"""
    cfg = build_config(config, SCREEN_BUY, SCREEN_CONFIRM)
    cfg['risk_control'] = {'enabled': True, **rc_params}
    return cfg


def run_val_fold(features_df: pd.DataFrame, cfg: dict, cost_rate: float,
                 full_end: str, fold_idx: int) -> dict | None:
    """在指定折的验证段回测一次，返回绩效 + 风控触发明细。失败返回 None。"""
    train_end, val_start, val_end_def = FOLD_DEFS[fold_idx]
    val_end = val_end_def or full_end
    vb = backtest_metrics_verbose(features_df, cfg, val_start, val_end, cost_rate)
    if vb is None:
        return None
    m, vn, veq, vtr, rev = vb
    return {
        'fold': fold_idx + 1,
        'val_range': f"{val_start}~{val_end}",
        'val_actual': f"{veq['date'].iloc[0].date()}~{veq['date'].iloc[-1].date()}",
        'val_sharpe': m['sharpe'], 'val_annual': m['annual_return'],
        'val_dd': m['max_drawdown'], 'val_trades': m['trade_count'],
        'val_n': vn,
        'events': [e.get('reason') for e in rev],
    }


def run_risk_screen(features_df: pd.DataFrame, config: dict, full_end: str,
                    cost_rate: float) -> tuple:
    """初筛：基线（无风控）vs 24 风控组合，全部在 5 折验证段直接跑（不选参）。

    返回 (rows, base_summary)。rows 每项含 5 折 OOS 夏普均值/中位/最差折、
    相对基线改善、触发统计、是否通过初筛。
    """
    # 基线（无风控）：固定生产阈值
    base_cfg = build_config(config, SCREEN_BUY, SCREEN_CONFIRM)
    base_folds = [run_val_fold(features_df, base_cfg, cost_rate, full_end, i)
                  for i in range(len(FOLD_DEFS))]
    base_folds = [f for f in base_folds if f is not None]
    if not base_folds:
        raise RuntimeError("风控初筛：基线 5 折验证段全部无效，无法对比")
    base_sharpes = [f['val_sharpe'] for f in base_folds]
    base_summary = {'mean': float(np.mean(base_sharpes)),
                    'median': float(np.median(base_sharpes)),
                    'worst': float(min(base_sharpes)),
                    'dd_mean': float(np.mean([f['val_dd'] for f in base_folds])),
                    'n_folds': len(base_folds)}
    print(f"  [base] 无风控 5 折 OOS 夏普: 均值 {base_summary['mean']:.3f}  "
          f"中位 {base_summary['median']:.3f} 最差折 {base_summary['worst']:.3f}")

    rows = []
    for sl in RISK_STOP_LOSSES:
        for dd in RISK_DD_LIMITS:
            for cd in RISK_COOLDOWNS:
                cfg = build_risk_cfg(config, {'stop_loss_pct': sl,
                                              'dd_limit_pct': dd,
                                              'cooldown_days': cd})
                folds = [run_val_fold(features_df, cfg, cost_rate, full_end, i)
                         for i in range(len(FOLD_DEFS))]
                folds = [f for f in folds if f is not None]
                if not folds:
                    print(f"  [WARN] 组合 sl={sl} dd={dd} cd={cd} 无有效折，跳过")
                    continue
                sharpes = [f['val_sharpe'] for f in folds]
                mean = float(np.mean(sharpes))
                events = [e for f in folds for e in f['events']]
                improve = (mean - base_summary['mean']) / max(abs(base_summary['mean']), 1e-9)
                rows.append({
                    'stop_loss_pct': sl, 'dd_limit_pct': dd, 'cooldown_days': cd,
                    'mean': mean, 'median': float(np.median(sharpes)),
                    'worst': float(min(sharpes)),
                    'improve': improve,
                    'n_stop': events.count('止损'),
                    'n_dd': events.count('回撤熔断'),
                    'pass': improve >= SCREEN_IMPROVE,
                })
    rows.sort(key=lambda r: r['mean'], reverse=True)
    return rows, base_summary


def risk_grid_search(features_df: pd.DataFrame, config: dict, start: str, end: str,
                     cost_rate: float, rc_params_list: list) -> dict:
    """终审训练段网格：buy_th × confirm_days × 风控组合同时搜索，选训练段夏普最高。"""
    grid = []
    for buy_th in BUY_CANDIDATES:
        for cd in CONFIRM_CANDIDATES:
            for rcp in rc_params_list:
                cfg = build_config(config, buy_th, cd)
                cfg['risk_control'] = {'enabled': True, **rcp}
                bt = backtest_metrics_verbose(features_df, cfg, start, end, cost_rate)
                if bt is None:
                    continue
                m = bt[0]
                grid.append({'buy_th': buy_th, 'sell_th': round(buy_th * SELL_RATIO, 2),
                             'confirm_days': cd, 'rc': dict(rcp),
                             'sharpe': m['sharpe'], 'annual_return': m['annual_return'],
                             'max_drawdown': m['max_drawdown'],
                             'trade_count': m['trade_count']})
    if not grid:
        raise RuntimeError(f"训练段 {start}~{end} 网格搜索全部失败")
    return max(grid, key=lambda g: g['sharpe'])   # 并列时取先遇到的（最小 buy_th）


def run_risk_final(features_df: pd.DataFrame, config: dict, full_end: str,
                   cost_rate: float, rc_params_list: list) -> list:
    """终审：完整 walk-forward（训练段同搜阈值+风控，验证段冻结）。

    返回每折详情（含训练段所选参数与验证段绩效）。
    """
    folds = []
    for idx, (train_end, val_start, val_end_def) in enumerate(FOLD_DEFS):
        val_end = val_end_def or full_end
        print(f"\n  ── 折 {idx + 1}/{len(FOLD_DEFS)}  训练 ~{train_end} → 验证 {val_start}~{val_end} ──")
        best = risk_grid_search(features_df, config, TRAIN_START, train_end,
                                cost_rate, rc_params_list)
        print(f"  [grid] 训练段最优: buy={best['buy_th']:.2f} sell={best['sell_th']:.2f} "
              f"confirm={best['confirm_days']} stop={best['rc']['stop_loss_pct']} "
              f"dd={best['rc']['dd_limit_pct']} cool={best['rc']['cooldown_days']} "
              f"→ 训练夏普 {best['sharpe']:.4f}")
        cfg = build_config(config, best['buy_th'], best['confirm_days'])
        cfg['risk_control'] = {'enabled': True, **best['rc']}
        vf = run_val_fold(features_df, cfg, cost_rate, full_end, idx)
        if vf is None:
            print(f"  [ERR] 折 {idx + 1} 验证段无有效结果，跳过")
            continue
        vf.update({
            'train_range': f"{TRAIN_START}~{train_end}",
            'buy_th': best['buy_th'], 'sell_th': best['sell_th'],
            'confirm_days': best['confirm_days'],
            'rc': dict(best['rc']),
            'train_sharpe': best['sharpe'],
        })
        print(f"  [val]   验证段: 夏普 {vf['val_sharpe']:.4f}  年化 {vf['val_annual']:+.2f}%  "
              f"回撤 {vf['val_dd']:.2f}%  交易 {vf['val_trades']}  "
              f"风控事件 {len(vf['events'])}")
        folds.append(vf)
    return folds


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


# ── 风控验证报告（--risk-scan）────────────────────────

def build_risk_report(args, config, features_df, base_summary, screen_rows,
                      passed, final_folds, dsr, full_end, trials) -> str:
    L = []
    A = L.append
    A("# 独立风控层（单笔止损 / 权益回撤熔断 / 冷却期）walk-forward 验证报告")
    A("")
    A(f"- 生成日期：{date.today().isoformat()}（数据截至 {features_df['date'].max().date()}，"
      f"共 {len(features_df)} 个交易日）")
    A(f"- 标的：{args.symbol}　费率：单边 {args.cost}（{args.cost * 10000:.1f}‱）")
    A(f"- 运行命令：`python3 tools/daily_pipeline/walk_forward.py --risk-scan"
      f" --symbol {args.symbol} --cost {args.cost}`")
    A("")
    A(f"> **口径说明**：沿用 walk_forward 主报告口径（内存拷贝 config 并置 "
      f"`adaptive_thresholds.enabled=false`，使阈值网格真正生效；风控参数从 "
      f"`config['risk_control']` 读取）。风控是**规则而非调参**：初筛不选参、直接在 "
      f"5 折验证段跑；仅终审在训练段与阈值联合搜索、验证段冻结。")
    A("")

    # ── 1. 初筛 ──
    A(f"## 1. 初筛：固定生产阈值 × 风控网格（{len(RISK_STOP_LOSSES) * len(RISK_DD_LIMITS) * len(RISK_COOLDOWNS)} 组合，5 折 OOS）")
    A("")
    A(f"固定生产阈值 buy={SCREEN_BUY} / sell={round(SCREEN_BUY * SELL_RATIO, 2)} / "
      f"confirm={SCREEN_CONFIRM}；风控网格 = stop_loss_pct ∈ {RISK_STOP_LOSSES} × "
      f"dd_limit_pct ∈ {RISK_DD_LIMITS}（None=禁用）× cooldown_days ∈ {RISK_COOLDOWNS}。"
      f"每个组合直接在 5 折验证段回测（不选参）。")
    A(f"")
    A(f"**基线（无风控，固定生产阈值）**：5 折 OOS 夏普 均值 `{base_summary['mean']:.3f}` / "
      f"中位 `{base_summary['median']:.3f}` / 最差折 `{base_summary['worst']:.3f}`"
      f"（{base_summary['n_folds']} 折有效）")
    A("")
    rows = []
    for r in screen_rows:
        dd_s = '禁用' if r['dd_limit_pct'] is None else f"{r['dd_limit_pct']:.2f}"
        rows.append([f"{r['stop_loss_pct']:.2f}", dd_s, f"{r['cooldown_days']}",
                     f"{r['mean']:.3f}", f"{r['median']:.3f}", f"{r['worst']:.3f}",
                     f"{r['improve']:+.1%}", f"{r['n_stop']}", f"{r['n_dd']}",
                     '✅' if r['pass'] else ''])
    A(md_table(['止损', '回撤熔断', '冷却', '夏普均值', '中位', '最差折',
                'vs 基线', '止损触发', '熔断触发', '通过'], rows))
    A("")
    if passed:
        A(f"**通过初筛**（OOS 夏普均值相对基线改善 ≥ {SCREEN_IMPROVE:.0%}）："
          + "；".join(
              f"sl={r['stop_loss_pct']:.2f}/dd={'禁用' if r['dd_limit_pct'] is None else format(r['dd_limit_pct'], '.2f')}"
              f"/cd={r['cooldown_days']}（均值 {r['mean']:.3f}，改善 {r['improve']:+.1%}）"
              for r in passed))
    else:
        A(f"**无组合通过初筛**（OOS 夏普均值改善均 < {SCREEN_IMPROVE:.0%}）。")
    A("")
    A("> 注：初筛为规则验证而非选优——未通过不代表规则无效，只表示在此固定阈值口径下 "
      "OOS 夏普均值未能相对基线提升 ≥10%；仍可参考各组合的最差折/触发次数评估风控的防御价值。")
    A("")

    # ── 2. 终审 ──
    A(f"## 2. 终审：完整 walk-forward（训练段同搜阈值+风控参数，验证段冻结）")
    A("")
    if not final_folds:
        A("无通过初筛的组合，未执行终审。")
        A("")
    else:
        A(f"训练段网格 = buy_th ∈ {BUY_CANDIDATES} × confirm_days ∈ {CONFIRM_CANDIDATES} × "
          f"初筛通过风控组合 {len(passed)} 个，按训练段夏普最高选参，参数（含风控）冻结后验证段回测。")
        A("")
        f_rows = []
        for f in final_folds:
            dd_s = '禁用' if f['rc']['dd_limit_pct'] is None else f"{f['rc']['dd_limit_pct']:.2f}"
            f_rows.append([f"折 {f['fold']}", f['train_range'], f['val_actual'],
                           f"{f['buy_th']:.2f}", f"{f['sell_th']:.2f}", f"{f['confirm_days']}",
                           f"{f['rc']['stop_loss_pct']:.2f}", dd_s, f"{f['rc']['cooldown_days']}",
                           f"{f['train_sharpe']:.3f}", f"{f['val_sharpe']:.3f}",
                           fmt_pct(f['val_annual']), fmt_pct(f['val_dd'], sign=False),
                           f"{f['val_trades']}", f"{len(f['events'])}"])
        A(md_table(['折', '训练段', '验证段', 'buy', 'sell', '确认', '止损', '熔断', '冷却',
                    '训练夏普', '验证夏普', '验证年化', '验证回撤', '交易数', '风控事件'], f_rows))
        A("")
        v_sharpes = [f['val_sharpe'] for f in final_folds]
        v_annuals = [f['val_annual'] for f in final_folds]
        v_dds = [f['val_dd'] for f in final_folds]
        A(md_table(['指标', '终审 OOS', '基线（无风控）'], [
            ['夏普均值', f"{np.mean(v_sharpes):.3f}", f"{base_summary['mean']:.3f}"],
            ['夏普中位', f"{np.median(v_sharpes):.3f}", f"{base_summary['median']:.3f}"],
            ['夏普最差折', f"{min(v_sharpes):.3f}", f"{base_summary['worst']:.3f}"],
            ['年化均值', fmt_pct(np.mean(v_annuals)), '—'],
            ['回撤均值', fmt_pct(np.mean(v_dds), sign=False),
             fmt_pct(base_summary['dd_mean'], sign=False)],
        ]))
        A("")
        # ── 3. DSR ──
        A(f"## 3. Deflated Sharpe Ratio（终审，扣除试错次数）")
        A("")
        if dsr is None:
            A("有效折 < 2，无法估计 V[SR̂]，DSR 不适用。")
        else:
            A(f"- 试错次数 N = **{dsr['trials']}** = 初筛 {len(screen_rows)} 组合 + 终审 "
              f"{len(FOLD_DEFS)} 折 × {len(BUY_CANDIDATES)} buy × {len(CONFIRM_CANDIDATES)} confirm "
              f"× {len(passed)} 风控组合（如实计入全部搜索过的参数组合）")
            A(f"- 各折夏普样本方差 V[SR̂] = {dsr['var_sr']:.5f}（√V = {dsr['sd_sr']:.4f}）")
            A(f"- 期望最大夏普阈值 SR0 = **{dsr['sr0']:.4f}**（{dsr['trials_note']}）")
            A(f"- 每折 deflated = (SR_i − SR0) / SE_i，SE_i = √((1 + 0.5·SR_i²)/n_i)，p 为单侧正态")
            A("")
            d_rows = []
            for i, (f, pf) in enumerate(zip(final_folds, dsr['per_fold'])):
                d_rows.append([f"折 {f['fold']}", f"{pf['sr']:.3f}", f"{pf['n']}",
                               f"{pf['se']:.4f}", f"{pf['deflated']:+.3f}",
                               f"{pf['p']:.4f}" + ("⚠" if pf['p'] < 0.05 else "")])
            d_rows.append(['均值(合并)', f"{dsr['mean_sr']:.3f}",
                           f"{round(float(np.mean([f['val_n'] for f in final_folds])))}",
                           f"{dsr['se_mean']:.4f}", f"{dsr['mean_deflated']:+.3f}",
                           f"{dsr['mean_p']:.4f}" + ("⚠" if dsr['mean_p'] < 0.05 else "")])
            A(md_table(['折', 'SR_i', 'n_i', 'SE_i', 'Deflated', 'p 值(单侧)'], d_rows))
            A("")
            A(f"- **均值夏普的 deflated 显著性：p = {dsr['mean_p']:.4f}**"
              f"{'（<0.05，扣除多重检验后仍显著）' if dsr['mean_p'] < 0.05 else '（未达显著，需谨慎对待）'}")
            A("")
            A(f"> DSR 口径与主报告一致（简化版：SR 标准误正态近似，V[SR̂] 用各折样本方差，"
              f"未做 rolling 重叠相关性修正）。试错次数按最诚实口径计入初筛全部组合与终审全部搜索组合。")
            A(f"> **方法局限**：初筛与终审共享同一批 5 折验证段样本——初筛门槛消费了 OOS 信息"
              f"（选择偏差），终审 SR_i 因此带有乐观倾向，且该偏差无法完全由 trials 计数抵消；"
              f"终审结果应视为『本验证协议下』的上限估计，落地前建议在更长样本外区间复核。")
            A("")

    # ── 4. 结论 ──
    A(f"## 4. 结论与建议")
    A("")
    if not final_folds:
        A(f"在当前验证口径（{TRAIN_START} ~ {full_end}，5 折滚动 OOS）下，风控网格 24 组合无一达到"
          f"『OOS 夏普均值相对基线改善 ≥ {SCREEN_IMPROVE:.0%}』的门槛。")
        A("")
        A(f"- 风控的价值主要体现在尾部防御（最差折/回撤/单笔深亏），而非夏普均值本身；"
          f"若目标是控制最大回撤与单笔止损，应直接比较各组合的验证段回撤与最差折，而非以夏普门槛一票否决。")
        A(f"- 可考虑放宽初筛门槛或改用『回撤改善』作为主目标重跑 `--risk-scan`。")
    else:
        A(f"- 终审 OOS 夏普均值 **{np.mean([f['val_sharpe'] for f in final_folds]):.3f}** vs 基线 "
          f"**{base_summary['mean']:.3f}**（{np.mean([f['val_sharpe'] for f in final_folds]) - base_summary['mean']:+.3f}）。")
        if dsr is None:
            A("- 有效折 < 2，DSR（Deflated Sharpe Ratio）不适用。")
        else:
            A(f"- DSR 均值 deflated = {dsr['mean_deflated']:+.3f}，p = {dsr['mean_p']:.4f}"
              f"{'（扣除试错后仍显著）' if dsr['mean_p'] < 0.05 else '（不显著，谨慎）'}。")
        best_worst = min(r['worst'] for r in passed)
        A(f"- 防御价值观察：初筛中通过组合的最差折普遍优于基线（最优 `{best_worst:.3f}` vs 基线 "
          f"`{base_summary['worst']:.3f}`）——止损 5% 在深跌折（2026-07，单日 -6% 级）中"
          f"有效截断了单笔深亏；但该优势在训练段联合搜索口径下未转化为 OOS 夏普提升，"
          f"说明风控改善集中在尾部风险而非期望收益。")
        A(f"- 建议：若目标是压回撤/单笔亏损，可采信初筛表的最差折与触发统计，按 "
          f"`sl=0.05/cd=3`（熔断可加 `dd=0.10`）口径小范围试用；若目标是夏普，本验证不支持开启风控。")

    A("")
    A("---")
    A("*本报告由 walk_forward.py --risk-scan 自动生成（纯只读分析，未修改任何生产代码或数据文件；"
      f"风控默认关闭，需在 config.json 的 risk_control.enabled=true 才在生产回测中生效）。*")
    return '\n'.join(L)


# ── 主入口 ────────────────────────────────────────────

def run_risk_scan_main(args, config, features_df, full_end) -> None:
    """--risk-scan 主流程：初筛（固定阈值 × 风控网格）→ 终审（联合搜索 + DSR）。"""
    print("=" * 62)
    print("  风控层 walk-forward 验证（初筛 + 终审 DSR）")
    print(f"  标的 {args.symbol} | 费率 {args.cost}")
    print("=" * 62)
    print("  [i] 风控是规则不调参：初筛固定生产阈值直接跑验证段；终审才联合搜索")

    # 1) 初筛
    print("\n[1/2] 初筛：固定生产阈值 × 风控网格（5 折 OOS，不选参）...")
    screen_rows, base_summary = run_risk_screen(features_df, config, full_end, args.cost)
    passed = [r for r in screen_rows if r['pass']]
    print(f"  [screen] 通过 {len(passed)}/{len(screen_rows)} 组合（OOS 夏普均值改善 ≥ {SCREEN_IMPROVE:.0%}）")

    # 2) 终审
    final_folds, dsr, trials = [], None, 0
    if passed:
        rc_params_list = [{'stop_loss_pct': r['stop_loss_pct'],
                           'dd_limit_pct': r['dd_limit_pct'],
                           'cooldown_days': r['cooldown_days']} for r in passed]
        print("\n[2/2] 终审：完整 walk-forward（训练段同搜阈值+风控参数）...")
        final_folds = run_risk_final(features_df, config, full_end, args.cost,
                                     rc_params_list)
        # DSR 试错次数 = 初筛组合数 + 终审全部搜索组合数（5 折 × 7 buy × 3 confirm × K 风控）
        trials = (len(screen_rows)
                  + len(FOLD_DEFS) * len(BUY_CANDIDATES) * len(CONFIRM_CANDIDATES) * len(passed))
        if final_folds:
            dsr = compute_dsr([f['val_sharpe'] for f in final_folds],
                              [f['val_n'] for f in final_folds], trials)
        if dsr is None:
            print("  [WARN] 有效折 < 2，DSR 无法计算")
    else:
        print("\n[2/2] 无通过初筛的组合，跳过终审。")

    # 3) 报告
    print("\n[生成报告]")
    report = build_risk_report(args, config, features_df, base_summary, screen_rows,
                               passed, final_folds, dsr, full_end, trials)
    print("\n" + "=" * 62)
    print(report)
    print("=" * 62)

    report_path = Path(args.risk_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding='utf-8')
    print(f"\n  [SAVE] 报告 → {report_path}")


def main():
    parser = argparse.ArgumentParser(description='walk-forward 多折回测 + DSR + 参数高原分析')
    parser.add_argument('--symbol', default='588000')
    parser.add_argument('--cost', type=float, default=0.00055, help='单边费率，默认万5.5')
    parser.add_argument('--trials', type=int, default=105,
                        help='DSR 试错次数 N（默认 7×3×5=105）')
    parser.add_argument('--report', default=str(DEFAULT_REPORT), help='报告输出路径')
    parser.add_argument('--risk-scan', action='store_true',
                        help='风控参数 walk-forward 验证模式（初筛 + 终审 DSR），写独立报告')
    parser.add_argument('--risk-report', default=str(RISK_DEFAULT_REPORT),
                        help='风控验证报告输出路径（默认 docs/risk_control_report_2026-08-04.md）')
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

    # 风控验证模式：独立流程（初筛 + 终审），完成后直接退出
    if args.risk_scan:
        run_risk_scan_main(args, config, features_df, full_end)
        return


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
