#!/usr/bin/env python3
"""rotation_backtest.py — 单边相对强弱轮动回测 · 阶段 2（S2，套利方向）

背景（阶段 1 已定论）：候选池 42 只 ETF + 基准 588000，与 588000 稳定协整对
= 0 → S1 配对套利不成立。S2 改为单边相对强弱轮动：每期横截面因子打分 →
持有最强 Top N（A 股 ETF 无做空、T+1，只能持强避弱），不需做空、无配对腿。

方法：
  1. backtest    全段回测：5 个横截面因子（动量 5/20 日、趋势 c/MA20 与
                  MA20/MA60、20 日波动率负向）rank 归一化等权合成打分 →
                  周频/日频 × Top N ∈ {3,5} × 空仓阈值小网格 → 对比臂
                  （588000 基线【引用 daily_pipeline metrics.json】、
                   候选池 43 只等权持有）→ 成本敏感性（5 档单边费率）
  2. walkforward 5 折锚定式 walk-forward：前段训练（小网格选 N/频率/空仓
                  阈值）→ 每折 OOS 夏普 → DSR（试错次数 = 网格组合数）
                  → 参数高原（Top N ∈ {2..5} × 空仓阈值邻域）
  3. report      生成 data/etf_scan/rotation_report.md + rotation_nav.png
                  + rotation_metrics.json
  4. all         上述全流程

用法（用 kronos venv 的 python，已装 pandas/numpy/scipy/matplotlib）：
  tools/kronos/.venv/bin/python tools/etf_scan/rotation_backtest.py --step all
  tools/kronos/.venv/bin/python tools/etf_scan/rotation_backtest.py --step backtest
  tools/kronos/.venv/bin/python tools/etf_scan/rotation_backtest.py --step walkforward
  tools/kronos/.venv/bin/python tools/etf_scan/rotation_backtest.py --step report

约束：只新增本文件，不改任何现有生产代码；无随机过程（seed 固定备用）。
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "etf_scan"
BASELINE_METRICS = PROJECT_ROOT / "data" / "588000" / "backtest" / "metrics.json"

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------
BASE_SYMBOL = "588000"          # 基准：科创50 ETF（可交易，纳入轮动候选池）
WARMUP = 60                     # 因子预热天数（MA60 需要），预热期内不参与回测
TRADING_DAYS = 252              # 年化交易日数

# 因子定义：(名称, 方向)。方向 +1 = 高值加分，-1 = 低值加分（如波动率）
FACTORS: List[Tuple[str, int]] = [
    ("ret5", 1),            # 5 日动量
    ("ret20", 1),           # 20 日动量
    ("trend_c_ma20", 1),    # 收盘 / MA20 - 1
    ("trend_ma20_ma60", 1), # MA20 / MA60 - 1
    ("vol20", -1),          # 20 日收益波动率（负向：低波动加分）
]

# 成本敏感性档位（单边费率：买入与卖出各按成交额计一次 r）
COST_RATES = [0.0, 0.00015, 0.0003, 0.00055, 0.001]

# 全段主网格：频率 × Top N × 空仓阈值（最强标的绝对趋势 收盘/MA20−1 ≤ 阈值则空仓）
GRID_FREQS = ["weekly", "daily"]
GRID_TOP_N = [3, 5]
GRID_CASH = [0.0, 0.02, 0.05, 0.10]

# walk-forward：小网格（DSR 试错次数 = 组合数，覆盖主网格关键阈值点）
WF_TOP_N = [3, 5]
WF_FREQS = ["weekly", "daily"]
WF_CASH = [0.0, 0.05, 0.10]

# 参数高原：Top N ∈ {2..5} × 空仓阈值邻域
PLATEAU_TOP_N = [2, 3, 4, 5]
PLATEAU_CASH = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
PLATEAU_TOL = 0.15           # 相邻参数夏普相对差异 < 15% 视为平坦

# 5 折验证段（锚定式：训练段 = 回测起点 → 验证段起点）
FOLD_VALS = [
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-06-30"),
    ("2026-07-01", "2026-08-05"),
]

NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329   # 欧拉常数 γ（DSR 期望最大阈值公式用）
SEED = 42

# ---------------------------------------------------------------------------
# 数据加载（pool.csv 存在则读长表，否则从各标的 CSV 现拼——两者等价）
# ---------------------------------------------------------------------------
def load_panels() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取候选池 42 只 + 基准 588000 的收盘/成交额面板（date × symbol）

    优先读合并长表 data/etf_scan/pool.csv（date,...,symbol）；不存在时从
    data/etf_scan/{symbol}.csv 现拼（当前磁盘状态即为此情形）。
    交易日对齐：只保留全标的都有数据的交易日。
    返回 (close 面板, volume 面板)，两者索引与列一致。
    """
    symbols = _pool_symbols()
    pool_csv = OUT_DIR / "pool.csv"
    if pool_csv.exists():
        long = pd.read_csv(pool_csv, parse_dates=["date"])
        close = long.pivot_table(index="date", columns="symbol", values="close")
        volume = long.pivot_table(index="date", columns="symbol", values="volume")
    else:
        closes, volumes = {}, {}
        for s in symbols:
            path = OUT_DIR / f"{s}.csv"
            if not path.exists():
                raise FileNotFoundError(f"缺少 {path}，请先运行 scan_pool.py --step fetch")
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.dropna(subset=["close"]).sort_values("date")
            closes[s] = df.set_index("date")["close"]
            volumes[s] = df.set_index("date")["volume"]
        close = pd.DataFrame(closes)
        volume = pd.DataFrame(volumes)

    # 对齐：只保留全标的都有数据的交易日（对齐后每只行数一致）
    # 注意：若某标的整段缺失（上市晚/停牌贯穿全窗），dropna(how="any") 会截掉
    # 整个窗口。当前池内 43 只数据齐全（869 条同日），无实际影响；单标的局部
    # NaN 由 compute_factors 的横截面 rank 与 run_rotation 的 valid_idx 自动跳过。
    valid = close.dropna(how="any").index
    close = close.loc[valid]
    volume = volume.loc[valid]
    return close, volume


def _pool_symbols() -> List[str]:
    """候选池 42 只 + 基准 588000（与 scan_pool 列表一致的 43 只全集）"""
    import scan_pool as sp  # 仅复用列表常量，不执行其 main
    return [s for s, _, _ in sp.ETF_POOL] + [BASE_SYMBOL]


# ---------------------------------------------------------------------------
# 横截面因子打分
# ---------------------------------------------------------------------------
def compute_factors(closes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """5 因子横截面 rank 归一化（0~1）等权合成打分

    返回 (score 面板, vol20 面板, market_trend 序列)。score 为 date×symbol，
    行内（横截面）pct-rank 后按因子方向加权平均（默认等权）。因子全 NaN
    （上市晚/停牌）的标的该日自动得 NaN → 不参与打分/选股。前 WARMUP 行为 NaN。

    market_trend = 当日全池 收盘/MA20−1 的最大值（最强标的相对其 20 日线的
    绝对偏离），用于空仓开关：rank 归一化分承载不了绝对阈值（Top1 恒≈1.0），
    故空仓阈值作用在绝对趋势上——最强标的都未站上 MA20 达阈值时视为整体走弱。
    """
    ma20 = closes.rolling(20).mean()
    ma60 = closes.rolling(60).mean()
    raw: Dict[str, pd.DataFrame] = {
        "ret5": closes.pct_change(5),
        "ret20": closes.pct_change(20),
        "trend_c_ma20": closes / ma20 - 1.0,
        "trend_ma20_ma60": ma20 / ma60 - 1.0,
        "vol20": closes.pct_change().rolling(20).std(),
    }
    ranked: Dict[str, pd.DataFrame] = {}
    for name, sign in FACTORS:
        r = raw[name].rank(axis=1, pct=True)   # 0~1，NaN 保持
        ranked[name] = r if sign > 0 else (1.0 - r)
    score = sum(ranked[n] for n, _ in FACTORS) / len(FACTORS)
    score = score.clip(lower=0.0, upper=1.0)
    market_trend = raw["trend_c_ma20"].max(axis=1)
    return score, raw["vol20"], market_trend


# ---------------------------------------------------------------------------
# 回测引擎（单标的信号回测不适用多标的轮动，这里自写轻量轮动引擎）
# ---------------------------------------------------------------------------
def run_rotation(scores: pd.DataFrame, closes: pd.DataFrame, freq: str = "weekly",
                 top_n: int = 3, cash_th: float = 0.0, cost_rate: float = 0.0003,
                 vol_weight: bool = False, vol20: Optional[pd.DataFrame] = None,
                 market: Optional[pd.Series] = None) -> Dict:
    """单边轮动回测

    freq: 'weekly'（每周最后交易日收盘调仓）| 'daily'（每日收盘调仓）
    cash_th: 空仓阈值。当日最强标的绝对趋势（全池 收盘/MA20−1 最大值）
        ≤ 阈值 → 空仓持现金（防系统性下跌）。默认 0.0 = 最强标的收盘跌破
        其 MA20 即空仓；阈值可配置。
    cost_rate: 单边费率，买入/卖出各按成交额计一次
    vol_weight: 波动率倒数加权（默认等权）
    market: market_trend 序列（compute_factors 返回），与 scores 同索引
    调仓：卖出不在新 Top N 的持仓、买入新进 Top N 的，按收盘价成交，
    同日先卖后买。T+1 约束天然满足（今日收盘买入，最早明日收盘卖出）。
    返回含 nav 序列、调仓明细、指标 dict。
    """
    assert freq in ("weekly", "daily"), f"freq 必须是 weekly/daily，收到 {freq}"
    syms = list(closes.columns)
    n_sym = len(syms)
    rets = closes.pct_change().fillna(0.0)
    dates = list(scores.index)
    n_days = len(dates)

    # 调仓日位置（对齐 scores 索引）
    pos = pd.Series(np.arange(n_days), index=dates)
    if freq == "weekly":
        rebal_pos = set(pos.resample("W-FRI").last().dropna().astype(int).tolist())
    else:
        rebal_pos = set(range(n_days))

    # 预取 numpy 数组（快）
    score_arr = scores.values.astype(float)
    ret_arr = rets.values
    vol_arr = vol20.values if vol20 is not None else None
    market_arr = market.reindex(dates).values if market is not None else None

    weights = np.zeros(n_sym)          # 占净值比例
    nav = 1.0
    navs = np.ones(n_days)
    trades = 0
    turnover_total = 0.0               # 累计单边换手（= 成交额变动/总资产 单边）
    cash_days = 0
    rebal_log: List[Dict] = []

    for t in range(n_days):
        r_t = float(np.dot(weights, ret_arr[t])) if t > 0 else 0.0
        nav *= (1.0 + r_t)
        navs[t] = nav
        if t not in rebal_pos:
            continue
        # 调仓日：先卖后买，按收盘价（当日收益已计入 navs[t]）
        row = score_arr[t]
        valid_idx = np.where(np.isfinite(row))[0]
        if len(valid_idx) == 0:
            target = np.zeros(n_sym)
            cash_days += 1
        elif market_arr is not None and np.isfinite(market_arr[t]) \
                and market_arr[t] <= cash_th:
            target = np.zeros(n_sym)
            cash_days += 1
        else:
            cand = valid_idx
            k = min(top_n, len(cand))
            top = cand[np.argsort(row[cand])[::-1][:k]]
            if vol_weight and vol_arr is not None:
                v = np.array([vol_arr[t, s_i] if np.isfinite(vol_arr[t, s_i]) else np.nan
                              for s_i in top])
                if np.all(np.isfinite(v)):
                    v = np.maximum(v, 1e-12)
                    w = (1.0 / v) / np.sum(1.0 / v)
                else:
                    w = np.full(k, 1.0 / k)   # 波动率缺数据 → 退化为等权
            else:
                w = np.full(k, 1.0 / k)
            target = np.zeros(n_sym)
            for j, s_i in enumerate(top):
                target[s_i] = w[j]
        dw = target - weights
        turnover = float(np.abs(dw).sum() / 2.0)          # 单边换手
        cost = float(np.abs(dw).sum()) * cost_rate        # 双边成交额 × 单边费率
        nav *= (1.0 - cost)
        navs[t] = nav
        n_buy = int(np.sum(dw > 0))
        n_sell = int(np.sum(dw < 0))
        trades += (n_buy + n_sell)
        turnover_total += turnover
        weights = target
        rebal_log.append({
            "date": str(dates[t]), "turnover": turnover, "cost": cost,
            "n_buy": n_buy, "n_sell": n_sell, "cash": bool(np.all(target == 0)),
        })

    nav_series = pd.Series(navs, index=dates)
    # 调仓期收益 = 相邻调仓日（收盘、扣成本后）之间的净值复合变化；
    # 首个周期 = 初始净值 1.0 → 第一次调仓日净值（含建仓成本）；
    # 末次调仓日之后到回测结束的收益不入「调仓期」（无调仓事件，非完整周期）。
    period_nav = [1.0] + [navs[p] for p in sorted(rebal_pos)]
    period_rets = [b / a - 1.0 for a, b in zip(period_nav[:-1], period_nav[1:])]
    metrics = compute_metrics(nav_series, period_rets, trades, turnover_total,
                              len(rebal_log), cash_days)
    return {"nav": nav_series, "trades": trades, "turnover_total": turnover_total,
            "rebal_log": rebal_log, "metrics": metrics}


def compute_metrics(nav: pd.Series, period_rets: List[float], trades: int,
                    turnover_total: float, n_rebal: int, cash_days: int) -> Dict:
    """统一指标口径（与对比臂一致）：年化收益、夏普、最大回撤、胜率等"""
    nav = nav.dropna()
    if len(nav) < 3:
        return {"nav_final": float("nan"), "total_return": float("nan"),
                "annual_return": float("nan"), "sharpe": float("nan"),
                "max_drawdown": float("nan"), "win_rate_daily": float("nan"),
                "win_rate_period": float("nan"), "trades": trades,
                "turnover_total": turnover_total, "n_rebal": n_rebal,
                "cash_days": cash_days}
    daily = nav.pct_change().dropna()
    n_days = len(daily)
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    annual = float((1.0 + total) ** (TRADING_DAYS / n_days) - 1.0)
    sd = float(daily.std(ddof=1))
    if sd > 0:
        sharpe = float(daily.mean() / sd * math.sqrt(TRADING_DAYS))
    elif float(daily.mean()) == 0.0:
        sharpe = 0.0   # 全空仓持现金：无收益无波动，夏普取 0（无风险口径）
    else:
        sharpe = float("nan")
    dd = float((nav / nav.cummax() - 1.0).min())
    win_d = float((daily > 0).mean())
    win_p = float(np.mean([r > 0 for r in period_rets])) if period_rets else float("nan")
    return {"nav_final": float(nav.iloc[-1]), "total_return": total,
            "annual_return": annual, "sharpe": sharpe, "max_drawdown": dd,
            "win_rate_daily": win_d, "win_rate_period": win_p, "trades": trades,
            "turnover_total": turnover_total, "n_rebal": n_rebal,
            "cash_days": cash_days}


# ---------------------------------------------------------------------------
# 对比臂
# ---------------------------------------------------------------------------
def equal_weight_arm(closes: pd.DataFrame) -> pd.Series:
    """候选池 43 只等权持有（纯 beta 对照：无择时无调仓，仅初始等权、零成本）"""
    norm = closes / closes.iloc[0]
    return norm.mean(axis=1)


def baseline_metrics() -> Dict:
    """588000 基线指标：引用 daily_pipeline 的 metrics.json（任务口径：以 json 为准）

    文件缺失时回退到任务给定的历史参考值并标注「参考」。
    """
    fallback = {"sharpe": 0.7057, "annual_return": 12.94, "max_drawdown": -15.87,
                "total_return": 51.99, "nav_final": 1.5199, "win_rate": 46.36,
                "trades": 110, "source": "任务历史参考值（metrics.json 缺失）"}
    try:
        with open(BASELINE_METRICS, encoding="utf-8") as f:
            d = json.load(f)
        return {"sharpe": float(d["sharpe"]), "annual_return": float(d["annual_return"]),
                "max_drawdown": float(d["max_drawdown"]),
                "total_return": float(d["total_return"]),
                "nav_final": 1.0 + float(d["total_return"]) / 100.0,
                "win_rate": float(d["win_rate"]),
                "trades": int(d.get("trade_count", 110)),
                "source": "引用 data/588000/backtest/metrics.json"}
    except (OSError, KeyError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# walk-forward + DSR（方法论复用 daily_pipeline/walk_forward.py 的 compute_dsr
# 思路，代码为独立新写）
# ---------------------------------------------------------------------------
def build_folds(dates: pd.DatetimeIndex) -> List[Tuple[str, str]]:
    """5 折验证段定义：(验证段起点, 验证段终点)，训练段 = 回测起点 → 验证段起点"""
    folds = []
    for lo, hi in FOLD_VALS:
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        if lo_ts >= dates[-1]:
            continue
        folds.append((lo, hi if hi_ts <= dates[-1] else str(dates[-1].date())))
    return folds


def _clip_window(idx: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """把 [start, end] 字符串窗口对齐到实际交易日（端点可能为节假日）"""
    lo = idx.searchsorted(pd.Timestamp(start), side="left")
    hi = idx.searchsorted(pd.Timestamp(end), side="right")
    return idx[lo:hi]


def _backtest_slice(scores: pd.DataFrame, closes: pd.DataFrame, vol20: pd.DataFrame,
                    market: pd.Series, start: str, end: str, freq: str, top_n: int,
                    cash_th: float, cost_rate: float,
                    vol_weight: bool = False) -> Optional[Dict]:
    """在 [start, end] 窗口内跑轮动（端点自动对齐到实际交易日）"""
    win = _clip_window(scores.index, start, end)
    if len(win) < 3:
        return None
    s = scores.loc[win]
    c = closes.loc[win]
    v = vol20.loc[win]
    m = market.loc[win]
    return run_rotation(s, c, freq=freq, top_n=top_n, cash_th=cash_th,
                        cost_rate=cost_rate, vol_weight=vol_weight, vol20=v, market=m)


def walk_forward(scores: pd.DataFrame, closes: pd.DataFrame, vol20: pd.DataFrame,
                 market: pd.Series, cost_rate: float,
                 vol_weight: bool = False) -> Tuple[List[Dict], Optional[Dict], Dict]:
    """5 折锚定式 walk-forward（与全段同权重口径，保证门禁与头条一致）

    每折：训练段（起点→验证段起点）在小网格选最优参数（按训练段夏普），
    验证段用该参数冻结跑出 OOS 夏普。返回 (每折明细, DSR dict, 参数统计)。
    """
    dates = scores.index
    folds = build_folds(dates)
    start0 = str(dates[0].date())
    grid = [(n, f, c) for n in WF_TOP_N for f in WF_FREQS for c in WF_CASH]
    trials = len(grid)

    fold_rows: List[Dict] = []
    param_hist: List[Dict] = []
    oos_sharpes: List[float] = []
    oos_ns: List[int] = []

    for i, (val_lo, val_hi) in enumerate(folds, 1):
        # 训练段终点取验证段起点前一天，避免训练/验证窗口在边界重叠一天
        train_end = str((pd.Timestamp(val_lo) - pd.offsets.Day(1)).date())
        train_res: Dict[Tuple, Dict] = {}
        for (n, f, c) in grid:
            r = _backtest_slice(scores, closes, vol20, market, start0, train_end,
                                f, n, c, cost_rate, vol_weight=vol_weight)
            if r is not None and np.isfinite(r["metrics"]["sharpe"]):
                train_res[(n, f, c)] = r["metrics"]["sharpe"]
        if train_res:
            best = max(train_res, key=train_res.get)
        else:
            best = (3, "weekly", 0.0)   # 训练段无有效结果 → 默认参数
        n_b, f_b, c_b = best
        param_hist.append({"fold": i, "top_n": n_b, "freq": f_b, "cash_th": c_b,
                           "train_sharpe": float(train_res.get(best, float("nan")))})
        oos = _backtest_slice(scores, closes, vol20, market, val_lo, val_hi,
                              f_b, n_b, c_b, cost_rate, vol_weight=vol_weight)
        if oos is None:
            continue
        m = oos["metrics"]
        oos_sharpes.append(m["sharpe"])
        oos_ns.append(len(oos["nav"]) - 1)
        fold_rows.append({"fold": i, "val_start": val_lo, "val_end": val_hi,
                          "top_n": n_b, "freq": f_b, "cash_th": c_b,
                          "train_sharpe": float(param_hist[-1]["train_sharpe"]),
                          "oos_sharpe": m["sharpe"], "oos_annual": m["annual_return"],
                          "oos_dd": m["max_drawdown"], "oos_n": len(oos["nav"]) - 1,
                          "oos_trades": m["trades"]})

    dsr = compute_dsr(oos_sharpes, oos_ns, trials)
    return fold_rows, dsr, {"trials": trials, "param_hist": param_hist}


def compute_dsr(sharpe_list: List[float], n_list: List[int], trials: int) -> Optional[Dict]:
    """DSR（Deflated Sharpe Ratio，López de Prado 2018 简化版）

    SR0 = √V[SR̂]·[(1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))]
    每折 deflated 夏普 = (SR_i - SR0) / SE_i，p = 单侧正态 1 - Φ(deflated)。
    试错次数 N = 网格组合数。有效折 < 2 时无法估计 V[SR̂]，返回 None。

    SE 口径：输入 SR 为年化夏普（×√252），SE 取年化形式
    √((252 + 0.5·SR²)/n)（由每期式 √((1+0.5·SR_d²)/n)×√252 推出）。
    注意 daily_pipeline/walk_forward.py 的生产版用每期式直接套年化 SR
    （SE 偏小 ~√252 倍，deflated 与 p 偏乐观），本文件按年化口径修正。
    """
    if len(sharpe_list) < 2:
        return None
    sr = np.asarray(sharpe_list, dtype=float)
    n = np.asarray(n_list, dtype=float)
    n = np.where(n > 0, n, 1.0)
    mean_sr = float(sr.mean())
    var_sr = float(sr.var(ddof=1))
    sd_sr = math.sqrt(var_sr)
    if trials <= 1:
        sr0 = 0.0
    else:
        z1 = NORMAL.inv_cdf(1.0 - 1.0 / trials)
        z2 = NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
        sr0 = sd_sr * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    per_fold = []
    for i in range(len(sr)):
        se = math.sqrt((TRADING_DAYS + 0.5 * sr[i] ** 2) / n[i])
        d = (sr[i] - sr0) / se
        per_fold.append({"sr": float(sr[i]), "n": int(n[i]), "se": se,
                         "deflated": d, "p": 1.0 - NORMAL.cdf(d)})
    se_mean = math.sqrt((TRADING_DAYS + 0.5 * mean_sr ** 2) / float(n.mean()))
    d_mean = (mean_sr - sr0) / se_mean
    return {"trials": trials, "mean_sr": mean_sr, "var_sr": var_sr, "sd_sr": sd_sr,
            "sr0": sr0, "per_fold": per_fold, "se_mean": se_mean,
            "mean_deflated": d_mean, "mean_p": 1.0 - NORMAL.cdf(d_mean)}


# ---------------------------------------------------------------------------
# 参数高原
# ---------------------------------------------------------------------------
def plateau_analysis(scores: pd.DataFrame, closes: pd.DataFrame, vol20: pd.DataFrame,
                     market: pd.Series, freq: str, cost_rate: float,
                     vol_weight: bool = False) -> Dict:
    """全段 Top N ∈ {2..5} × 空仓阈值邻域结果稳定性（小幅扰动不剧变）；权重与全段同口径"""
    rows = []
    for n in PLATEAU_TOP_N:
        for c in PLATEAU_CASH:
            r = run_rotation(scores, closes, freq=freq, top_n=n, cash_th=c,
                             cost_rate=cost_rate, vol_weight=vol_weight,
                             vol20=vol20, market=market)
            rows.append({"top_n": n, "cash_th": c, "sharpe": r["metrics"]["sharpe"],
                         "annual": r["metrics"]["annual_return"],
                         "dd": r["metrics"]["max_drawdown"],
                         "trades": r["metrics"]["trades"]})
    # 相邻判定：同一 Top N 内相邻空仓阈值相对差异 < 15% 视为平坦
    pairs = []
    for n in PLATEAU_TOP_N:
        sub = [r for r in rows if r["top_n"] == n]
        for a, b in zip(sub[:-1], sub[1:]):
            denom = max(abs(a["sharpe"]), abs(b["sharpe"]), 1e-9)
            rel = abs(a["sharpe"] - b["sharpe"]) / denom
            pairs.append({"top_n": n, "lo": a["cash_th"], "hi": b["cash_th"],
                          "rel_diff": rel, "flat": rel < PLATEAU_TOL})
    flat_n = sum(1 for p in pairs if p["flat"])
    overall_flat = flat_n >= math.ceil(len(pairs) / 2) if pairs else False
    # 最优参数邻域检查：夏普最高的参数组合，其同 Top N 相邻对必须平坦，
    # 否则即使整体平坦比例达标，最优邻域仍是尖峰（选参敏感）
    best = max(rows, key=lambda r: r["sharpe"]) if rows else None
    center_ok = True
    center_note = ""
    if best is not None:
        center_pairs = [p for p in pairs if p["top_n"] == best["top_n"]
                        and (p["hi"] == best["cash_th"] or p["lo"] == best["cash_th"])]
        center_ok = all(p["flat"] for p in center_pairs) if center_pairs else True
        worst = max(center_pairs, key=lambda p: p["rel_diff"]) if center_pairs else None
        if worst is not None and not worst["flat"]:
            center_note = (f"（最优参数 Top N={best['top_n']}/阈值={best['cash_th']:.2f} "
                           f"与相邻阈值 {worst['lo']:.2f}→{worst['hi']:.2f} 相对差异 "
                           f"{worst['rel_diff']:.0%}，为尖峰）")
    if overall_flat and center_ok:
        verdict = "参数高原（稳健，对参数选择不敏感）"
    else:
        verdict = "参数尖峰（过拟合敏感，参数微调即显著变差）" + center_note
    return {"rows": rows, "pairs": pairs, "flat_n": flat_n,
            "total_pairs": len(pairs), "overall_flat": overall_flat,
            "center_ok": center_ok, "verdict": verdict}


# ---------------------------------------------------------------------------
# 报告 / 图表 / JSON
# ---------------------------------------------------------------------------
def _zh_font():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    zh_fonts = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in zh_fonts:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            return "WenQuanYi Micro Hei"
    return None


def plot_nav(weekly_best_nav: pd.Series, base_nav: pd.Series, ew_nav: pd.Series) -> Path:
    """净值曲线 PNG：周频最优参数 vs 588000 基线（同窗口重算）vs 43 只等权"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    font = _zh_font()
    if font:
        matplotlib.rcParams["font.family"] = font
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    ax.plot(weekly_best_nav.index, weekly_best_nav.values, label="S2 轮动（周频最优参数）",
            lw=1.8)
    ax.plot(base_nav.index, base_nav.values, label="588000 基线（同窗口重算）", lw=1.4,
            alpha=0.85)
    ax.plot(ew_nav.index, ew_nav.values, label="候选池 43 只等权持有", lw=1.4, alpha=0.85)
    ax.set_title("S2 单边相对强弱轮动 vs 588000 基线 vs 等权持有（同窗口，含成本）")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "rotation_nav.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.2%}"


def fmt_num(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def write_report(ctx: Dict) -> Path:
    """生成 rotation_report.md（方法 + 对比臂 + 成本敏感性 + WF/DSR + 高原 + 结论）"""
    L: List[str] = []
    add = L.append
    add("# S2 单边相对强弱轮动回测报告（套利方向 · 阶段 2）\n")
    add(f"- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"- 数据: 候选池 42 只 ETF + 基准 588000（43 只），"
        f"{ctx['data_start']} ~ {ctx['data_end']}，{ctx['n_days']} 个对齐交易日，"
        f"前复权日线（腾讯/东财混合源）")
    add(f"- 回测窗口: {ctx['bt_start']} ~ {ctx['bt_end']}（{ctx['bt_days']} 个交易日，"
        f"前 {WARMUP} 日因子预热不计入）")
    add(f"- 成本: 单边费率（买入/卖出各按成交额计一次 r），默认 {ctx['cost_rate']:.6f}；"
        f"调仓按收盘价、同日先卖后买；T+1 约束天然满足")
    add(f"- 持仓权重: {'波动率倒数加权（--vol-weight）' if ctx.get('vol_weight') else '等权（默认）'}")
    add(f"- 随机性: 无随机过程（seed={SEED} 固定备用），结果可复现\n")

    # ---- 方法 ----
    add("## 1. 方法\n")
    add("每期横截面 5 因子打分（rank 归一化 0~1，等权合成）：")
    add("| 因子 | 计算 | 方向 |")
    add("|:--|:--|:--|")
    add("| 动量 | 5 日收益、20 日收益 | 高者加分 |")
    add("| 趋势 | 收盘/MA20−1、MA20/MA60−1 | 高者加分 |")
    add("| 波动率 | 20 日收益 std | 低者加分（负向） |")
    add("")
    add("轮动规则：调仓日按合成分取 Top N 等权持有（可选波动率倒数加权，"
        "默认等权）；当日最强标的绝对趋势（全池 收盘/MA20−1 最大值）≤ 空仓阈值 → "
        "空仓持现金（防系统性下跌）；缺数据标的（上市晚/停牌）跳过打分。"
        "频率：周频（每周最后交易日收盘）与日频对比。\n")
    add(f"> Kronos 对照臂：本轮不做。Kronos 为分钟线模型，候选池 43 只无分钟数据"
        f"（仅 588000 有 7 个月 min15）→ 横截面 Kronos 排序臂数据不足，暂缓。\n")

    # ---- 对比臂 ----
    base = ctx["baseline"]
    ew = ctx["ew_metrics"]
    add("## 2. 对比臂指标（同区间、同指标口径）\n")
    add(f"> 588000 基线数字{base['source']}（daily_pipeline 全历史口径，非本窗口重算；"
        f"基线净值曲线见下图为同窗口重算值，仅作曲线对照）。"
        f"任务文本参考值 夏普 0.7057/年化 +12.94% 与此 json 略有出入，以 json 为准。\n")
    add("| 指标 | 588000 基线（引用） | 候选池43等权 | S2 周频最优 | S2 日频最优 |")
    add("|:--|--:|--:|--:|--:|")
    add(f"| 净值 | {fmt_num(base['nav_final'], 4)} | {fmt_num(ew['nav_final'], 4)} | "
        f"{fmt_num(ctx['weekly_best']['metrics']['nav_final'], 4)} | "
        f"{fmt_num(ctx['daily_best']['metrics']['nav_final'], 4)} |")
    add(f"| 年化收益 | {base['annual_return']:.2f}% | {fmt_pct(ew['annual_return'])} | "
        f"{fmt_pct(ctx['weekly_best']['metrics']['annual_return'])} | "
        f"{fmt_pct(ctx['daily_best']['metrics']['annual_return'])} |")
    add(f"| 夏普 | {fmt_num(base['sharpe'])} | {fmt_num(ew['sharpe'])} | "
        f"{fmt_num(ctx['weekly_best']['metrics']['sharpe'])} | "
        f"{fmt_num(ctx['daily_best']['metrics']['sharpe'])} |")
    add(f"| 最大回撤 | {base['max_drawdown']:.2f}% | {fmt_pct(ew['max_drawdown'])} | "
        f"{fmt_pct(ctx['weekly_best']['metrics']['max_drawdown'])} | "
        f"{fmt_pct(ctx['daily_best']['metrics']['max_drawdown'])} |")
    add(f"| 胜率(按日) | — | {fmt_pct(ew['win_rate_daily'])} | "
        f"{fmt_pct(ctx['weekly_best']['metrics']['win_rate_daily'])} | "
        f"{fmt_pct(ctx['daily_best']['metrics']['win_rate_daily'])} |")
    add(f"| 胜率(按调仓期) | — | — | "
        f"{fmt_pct(ctx['weekly_best']['metrics']['win_rate_period'])} | "
        f"{fmt_pct(ctx['daily_best']['metrics']['win_rate_period'])} |")
    add(f"| 单边换手(累计) | — | 0 | "
        f"{fmt_num(ctx['weekly_best']['metrics']['turnover_total'], 1)} | "
        f"{fmt_num(ctx['daily_best']['metrics']['turnover_total'], 1)} |")
    add(f"| 交易笔数 | {int(base.get('trades', 110))}（引用） | 0 | "
        f"{ctx['weekly_best']['metrics']['trades']} | "
        f"{ctx['daily_best']['metrics']['trades']} |")
    add("")
    add(f"- 周频最优参数: Top N={ctx['weekly_best']['top_n']}, "
        f"空仓阈值={ctx['weekly_best']['cash_th']}，空仓天数 {ctx['weekly_best']['metrics']['cash_days']}")
    add(f"- 日频最优参数: Top N={ctx['daily_best']['top_n']}, "
        f"空仓阈值={ctx['daily_best']['cash_th']}，空仓天数 {ctx['daily_best']['metrics']['cash_days']}")
    wb_days = ctx['weekly_best']['metrics']['cash_days']
    add("> 胜率口径：按日胜率 = 日收益 > 0 的天数占比；空仓日（收益 0）计为不胜，"
        f"故空仓天数越多的策略按日胜率越低（周频最优 {wb_days}/{ctx['bt_days']} 天为"
        "空仓）。按调仓期胜率 = 相邻调仓日之间净值涨幅 > 0 的周期占比。\n")
    add(f"- 全段网格（{len(ctx['grid_rows'])} 组合，频率 × Top N × 空仓阈值）明细：\n")
    add("| 频率 | Top N | 空仓阈值 | 年化 | 夏普 | 最大回撤 | 交易笔数 | 空仓天数 |")
    add("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for g in ctx["grid_rows"]:
        add(f"| {g['freq']} | {g['top_n']} | {g['cash_th']:.2f} | "
            f"{fmt_pct(g['annual'])} | {fmt_num(g['sharpe'])} | {fmt_pct(g['dd'])} | "
            f"{g['trades']} | {g['cash_days']} |")
    add("")

    # ---- 成本敏感性 ----
    add("## 3. 成本敏感性（周频最优参数，单边费率扫描）\n")
    add("| 单边费率 | 年化收益 | 夏普 | 最大回撤 | 累计单边换手 | 交易笔数 | "
        "对年化侵蚀 |")
    add("|--:|--:|--:|--:|--:|--:|--:|")
    for c in ctx["cost_rows"]:
        add(f"| {c['rate']:.5f} | {fmt_pct(c['annual'])} | {fmt_num(c['sharpe'])} | "
            f"{fmt_pct(c['dd'])} | {fmt_num(c['turnover'], 1)} | {c['trades']} | "
            f"{fmt_pct(c['erosion'])} |")
    add("")
    add(f"- 平均每次调仓单边换手: "
        f"{fmt_num(ctx['cost_rows'][-1]['turnover'] / max(ctx['cost_rows'][-1]['n_rebal'], 1), 3)}"
        f"（按最高费率档）")
    add(f"- 换手主要来自{'周频' if ctx['weekly_best']['freq'] == 'weekly' else '日频'}调仓，"
        f"频率越高换手越高，成本对高换手策略侵蚀显著。\n")

    # ---- walk-forward + DSR ----
    add("## 4. 横截面 walk-forward（5 折锚定式）+ DSR\n")
    add(f"- 每折：训练段（回测起点 → 验证段起点）在小网格"
        f"（{ctx['wf']['trials']} 组合: N∈{WF_TOP_N} × 频率∈{WF_FREQS} × "
        f"空仓阈值∈{WF_CASH}）按训练段夏普选参，验证段冻结参数跑 OOS。")
    add(f"- DSR 试错次数 N = {ctx['wf']['dsr']['trials'] if ctx['wf']['dsr'] else '—'}（网格组合数）")
    add("- 说明：DSR 只按 walk-forward 小网格计试错次数（保守下限）；开发期另在全段网格/"
        "高原上扫过参数，未计入会使 SR0 略偏低、门禁略宽松（门禁结论见 §6）。"
        "周频策略首个调仓日前持有现金（约数日），属轻微保守拖累，非泄漏。\n")
    add("- 注意：末折验证段仅约 1 个月（2026-07~08），样本短，OOS 夏普年化放大后"
        "极端值属正常现象，但会显著影响均值（如实报告，不剔除）。"
        "另 fold5 训练段选出高空仓阈值参数，验证段 2026-07~08 最强标的趋势始终未"
        "超阈值 → 全期空仓（0 交易、0 收益、OOS 夏普 0），属真实避险行为而非失效。"
        "（若数据刷新后 fold5 不再全空仓，此句随表格自动失效，以当期为据。）\n")
    add("| 折 | 验证段 | 选参(Top N/频率/阈值) | 训练段夏普 | OOS 夏普 | OOS 年化 | OOS 回撤 |")
    add("|:--|:--|:--|--:|--:|--:|--:|")
    for f in ctx["wf"]["folds"]:
        add(f"| {f['fold']} | {f['val_start']}~{f['val_end']} | "
            f"{f['top_n']}/{f['freq']}/{f['cash_th']:.2f} | "
            f"{fmt_num(f['train_sharpe'])} | {fmt_num(f['oos_sharpe'])} | "
            f"{fmt_pct(f['oos_annual'])} | {fmt_pct(f['oos_dd'])} |")
    add("")
    if ctx["wf"]["dsr"]:
        dsr = ctx["wf"]["dsr"]
        add(f"- OOS 夏普均值 = **{fmt_num(dsr['mean_sr'])}**，样本方差 V[SR̂] = "
            f"{dsr['var_sr']:.5f}（√V = {dsr['sd_sr']:.4f}）")
        add(f"- 期望最大夏普阈值 SR0 = **{dsr['sr0']:.4f}**（试错 {dsr['trials']} 次）")
        add(f"- 均值 deflated 夏普 = **{fmt_num(dsr['mean_deflated'])}**，"
            f"p = **{dsr['mean_p']:.4f}**"
            f"{'（扣除多重检验后仍显著）' if dsr['mean_p'] < 0.05 else '（未达显著，存在过拟合/运气成分）'}")
        add("- 各折 deflated 夏普: " + ", ".join(
            f"折{i+1}: {fmt_num(p['deflated'])}" for i, p in enumerate(dsr["per_fold"])))
        add("")
    else:
        add("- 有效折 < 2，DSR 不适用。\n")

    # ---- 参数高原 ----
    pl = ctx["plateau"]
    add("## 5. 参数高原（全段，周频，Top N ∈ {2..5} × 空仓阈值邻域）\n")
    add("空仓阈值 = 最强标的绝对趋势（收盘/MA20−1）上限，单位 %。\n")
    add("| Top N | " + " | ".join(f"阈值{v*100:.0f}bp" if v < 0.01 else f"阈值{v*100:.0f}%"
                                  for v in PLATEAU_CASH) + " |")
    add("|:--|" + "".join("--:|" for _ in PLATEAU_CASH))
    for n in PLATEAU_TOP_N:
        cells = []
        for c in PLATEAU_CASH:
            r = next(r for r in pl["rows"] if r["top_n"] == n and r["cash_th"] == c)
            cells.append(fmt_num(r["sharpe"]))
        add(f"| {n} | " + " | ".join(cells) + " |")
    add("")
    add(f"- 相邻参数对 {pl['flat_n']}/{pl['total_pairs']} 平坦（相对夏普差异 < "
        f"{PLATEAU_TOL:.0%}）→ **{pl['verdict']}**\n")

    # ---- 结论 ----
    add("## 6. 门禁结论\n")
    gate = ctx["gate"]
    add(f"- 门禁标准（与 ADR-0001 一致）：OOS 夏普显著优于基线 "
        f"{fmt_num(ctx['baseline']['sharpe'])} 且 DSR p 值合格（< 0.05）。")
    add(f"- 5 折 OOS 夏普均值 = **{fmt_num(gate['oos_mean'])}** vs 基线 "
        f"{fmt_num(ctx['baseline']['sharpe'])}；"
        f"DSR p = **{gate['dsr_p']:.4f}**"
        f"（{'合格' if gate['dsr_p'] < 0.05 else '不合格'}）。")
    add(f"- **判定：{'✅ 通过门禁' if gate['pass'] else '❌ 不过门禁'}** — {gate['reason']}\n")
    add(f"- 净结论（S2 轮动 vs 588000 基线）：**{ctx['verdict']}**\n")
    if ctx["next_step"]:
        add(f"### 阶段 3 建议\n{ctx['next_step']}")
    out = OUT_DIR / "rotation_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[OK] 报告已生成: {out}")
    return out


def _json_safe(obj):
    """递归把 NaN/inf 转成 None，保证输出为严格合法 JSON"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_metrics_json(ctx: Dict) -> Path:
    """结构化指标 rotation_metrics.json（可复现用）"""
    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data": {"start": ctx["data_start"], "end": ctx["data_end"], "n_days": ctx["n_days"]},
        "backtest_window": {"start": ctx["bt_start"], "end": ctx["bt_end"],
                            "days": ctx["bt_days"]},
        "cost_rate_default": ctx["cost_rate"],
        "baseline_588000": {k: ctx["baseline"][k] for k in
                            ("sharpe", "annual_return", "max_drawdown", "nav_final", "source")},
        "equal_weight_43": ctx["ew_metrics"],
        "grid": ctx["grid_rows"],
        "weekly_best": {"top_n": ctx["weekly_best"]["top_n"],
                        "cash_th": ctx["weekly_best"]["cash_th"],
                        **{k: v for k, v in ctx["weekly_best"]["metrics"].items()}},
        "daily_best": {"top_n": ctx["daily_best"]["top_n"],
                       "cash_th": ctx["daily_best"]["cash_th"],
                       **{k: v for k, v in ctx["daily_best"]["metrics"].items()}},
        "cost_sensitivity": ctx["cost_rows"],
        "walk_forward": {"folds": ctx["wf"]["folds"], "dsr": ctx["wf"]["dsr"],
                         "trials": ctx["wf"]["trials"]},
        "plateau": {"rows": ctx["plateau"]["rows"], "verdict": ctx["plateau"]["verdict"],
                    "flat_n": ctx["plateau"]["flat_n"],
                    "total_pairs": ctx["plateau"]["total_pairs"]},
        "gate": ctx["gate"], "verdict": ctx["verdict"], "next_step": ctx["next_step"],
    }
    out = OUT_DIR / "rotation_metrics.json"
    out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[OK] 指标已生成: {out}")
    return out


# ---------------------------------------------------------------------------
# 全流程编排
# ---------------------------------------------------------------------------
def run_full(cost_rate: float, vol_weight: bool = False) -> Dict:
    """backtest + walkforward + report 全流程，返回上下文 dict（供 report 复用）"""
    np.random.seed(SEED)
    closes, volume = load_panels()
    # volume 面板用于数据完整性校验:与 close 同形状（对齐交易日）
    assert volume.shape == closes.shape, "volume 与 close 面板形状不一致"
    scores, vol20, market = compute_factors(closes)
    # 预热期后为回测窗口
    bt_scores = scores.iloc[WARMUP:]
    bt_close = closes.loc[bt_scores.index]
    bt_vol20 = vol20.loc[bt_scores.index]
    bt_market = market.loc[bt_scores.index]
    bt_start, bt_end = str(bt_scores.index[0].date()), str(bt_scores.index[-1].date())
    bt_days = len(bt_scores)

    ctx: Dict = {"data_start": str(closes.index[0].date()),
                 "data_end": str(closes.index[-1].date()), "n_days": len(closes),
                 "bt_start": bt_start, "bt_end": bt_end, "bt_days": bt_days,
                 "cost_rate": cost_rate, "vol_weight": vol_weight,
                 "baseline": baseline_metrics()}

    # 等权臂（同窗口）
    ew_nav = equal_weight_arm(bt_close)
    ctx["ew_metrics"] = compute_metrics(ew_nav, [], 0, 0.0, 0, 0)

    # 全段网格
    grid_rows: List[Dict] = []
    weekly_best: Optional[Dict] = None
    daily_best: Optional[Dict] = None
    for freq in GRID_FREQS:
        for n in GRID_TOP_N:
            for c in GRID_CASH:
                r = run_rotation(bt_scores, bt_close, freq=freq, top_n=n, cash_th=c,
                                 cost_rate=cost_rate, vol_weight=vol_weight,
                                 vol20=bt_vol20, market=bt_market)
                m = r["metrics"]
                grid_rows.append({"freq": freq, "top_n": n, "cash_th": c,
                                  "annual": m["annual_return"], "sharpe": m["sharpe"],
                                  "dd": m["max_drawdown"], "trades": m["trades"],
                                  "cash_days": m["cash_days"]})
                entry = {"freq": freq, "top_n": n, "cash_th": c,
                         "metrics": m, "nav": r["nav"]}
                if freq == "weekly":
                    if weekly_best is None or m["sharpe"] > weekly_best["metrics"]["sharpe"]:
                        weekly_best = entry
                else:
                    if daily_best is None or m["sharpe"] > daily_best["metrics"]["sharpe"]:
                        daily_best = entry
    ctx["grid_rows"] = grid_rows
    ctx["weekly_best"] = weekly_best
    ctx["daily_best"] = daily_best

    # 净值曲线用：周频最优 + 588000 基线（同窗口重算）+ 43 只等权
    base_nav = bt_close[BASE_SYMBOL] / bt_close[BASE_SYMBOL].iloc[0]
    ctx["navs"] = {"weekly": weekly_best["nav"], "base": base_nav, "ew": ew_nav}


    # 成本敏感性（周频最优参数）
    cost_rows = []
    wb = weekly_best
    zero_annual = None
    for rate in COST_RATES:
        r = run_rotation(bt_scores, bt_close, freq=wb["freq"], top_n=wb["top_n"],
                         cash_th=wb["cash_th"], cost_rate=rate, vol_weight=vol_weight,
                         vol20=bt_vol20, market=bt_market)
        m = r["metrics"]
        if rate == 0.0:
            zero_annual = m["annual_return"]
        cost_rows.append({"rate": rate, "annual": m["annual_return"], "sharpe": m["sharpe"],
                          "dd": m["max_drawdown"], "turnover": m["turnover_total"],
                          "trades": m["trades"], "n_rebal": m["n_rebal"],
                          "erosion": (zero_annual - m["annual_return"]) if zero_annual is not None
                          else float("nan")})
    ctx["cost_rows"] = cost_rows

    # walk-forward + DSR（与全段同权重口径，保证门禁与头条一致）
    folds, dsr, wf_extra = walk_forward(bt_scores, bt_close, bt_vol20, bt_market, cost_rate,
                                        vol_weight=vol_weight)
    ctx["wf"] = {"folds": folds, "dsr": dsr, "trials": wf_extra["trials"]}

    # 参数高原（周频，最优 Top N 附近的全网格已在 plateau 内展开）
    plateau = plateau_analysis(bt_scores, bt_close, bt_vol20, bt_market, "weekly",
                               cost_rate, vol_weight=vol_weight)
    ctx["plateau"] = plateau

    # 门禁
    oos_mean = float(np.mean([f["oos_sharpe"] for f in folds])) if folds else float("nan")
    dsr_p = dsr["mean_p"] if dsr else float("nan")
    base_sr = ctx["baseline"]["sharpe"]
    gate_pass = bool(np.isfinite(oos_mean) and oos_mean > base_sr and dsr_p < 0.05)
    if gate_pass:
        reason = (f"5 折 OOS 夏普均值 {oos_mean:.3f} > 基线 {base_sr:.3f}，"
                  f"且 DSR p={dsr_p:.4f} < 0.05（扣除 {dsr['trials']} 次试错后仍显著）")
    else:
        reason = (f"OOS 夏普均值 {oos_mean:.3f} "
                  f"{'≤' if not np.isfinite(oos_mean) or oos_mean <= base_sr else '>'}"
                  f" 基线 {base_sr:.3f}"
                  + ("" if not np.isfinite(oos_mean) or oos_mean > base_sr
                     else "（未显著优于基线）")
                  + f"；DSR p={dsr_p:.4f} "
                  + ("< 0.05" if dsr_p < 0.05 else "≥ 0.05（未通过显著性）"))
    ctx["gate"] = {"pass": gate_pass, "oos_mean": oos_mean, "dsr_p": dsr_p, "reason": reason}

    # 净结论（值不值得继续）
    wb_m = weekly_best["metrics"]
    if gate_pass and wb_m["sharpe"] > base_sr:
        ctx["verdict"] = ("值得继续：周频轮动全段夏普与 OOS 表现均显著优于 588000 基线，"
                          "且扣除了多重检验试错。建议进入阶段 3。")
        ctx["next_step"] = (
            "1) 把 S2 打分器接入实时链路：每日收盘后计算 5 因子横截面分，生成次日持仓建议；\n"
            "2) 实盘约束校验：ETF 申赎/流动性/最小交易单位、T+1 与涨跌停、成分股停牌影响；\n"
            "3) 引入交易成本细化（印花税仅股票、ETF 免印花税但含管理费差异）与滑点模型；\n"
            "4) Kronos 分钟线对照：为候选池补充分钟数据后复验横截面排序一致性。")
    else:
        ctx["verdict"] = ("回测证伪，套利方向应止损：S2 轮动在 5 折 OOS 上未显著跑赢 588000 "
                          "基线（或 DSR 不合格），收益来源不稳健，不建议投入更多回测/实盘资源。")
        ctx["next_step"] = (
            "1) 停止对 ETF 候选池的相对强弱/动量类轮动的进一步调参（避免过拟合加深）；\n"
            "2) 若仍要保留套利研究线，仅保留一个方向：以 588000 自身信号策略（daily_pipeline）"
            "为基准，研究费用更低、容量更大的纯基准工具化持有；\n"
            "3) 把阶段 1+2 的结论固化进 ADR：S1 无稳定协整对、S2 轮动未过门禁 → "
            "套利方向整体止损，资源转向单标的信号增强。")
    return ctx


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ETF 候选池单边相对强弱轮动回测（S2）")
    parser.add_argument("--step", default="all",
                        choices=["all", "backtest", "walkforward", "report"],
                        help="执行步骤（默认 all）")
    parser.add_argument("--cost-rate", type=float, default=0.0003,
                        help="单边费率（买卖各计一次，默认 0.0003）")
    parser.add_argument("--vol-weight", action="store_true",
                        help="持仓波动率倒数加权（默认等权）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "all":
        ctx = run_full(args.cost_rate, vol_weight=args.vol_weight)
        write_report(ctx)
        write_metrics_json(ctx)
        plot_nav(ctx["navs"]["weekly"], ctx["navs"]["base"], ctx["navs"]["ew"])
        return

    if args.step in ("backtest", "walkforward", "report"):
        ctx = run_full(args.cost_rate, vol_weight=args.vol_weight)
        if args.step in ("backtest", "report"):
            write_report(ctx)
            write_metrics_json(ctx)
            plot_nav(ctx["navs"]["weekly"], ctx["navs"]["base"], ctx["navs"]["ew"])
        if args.step == "walkforward":
            # walkforward 步骤仅输出 WF/DSR/高原相关摘要
            wf = ctx["wf"]
            print(f"\n===== walk-forward + DSR（试错 {wf['trials']} 组合）=====")
            print(f"{'折':>3} {'验证段':<24} {'选参':<20} {'OOS夏普':>8} {'OOS年化':>9}")
            for f in wf["folds"]:
                print(f"{f['fold']:>3} {f['val_start']}~{f['val_end']:<13} "
                      f"{f['top_n']}/{f['freq']}/{f['cash_th']:.2f} "
                      f"{f['oos_sharpe']:>8.3f} {f['oos_annual']:>9.2%}")
            if wf["dsr"]:
                d = wf["dsr"]
                print(f"OOS 均值夏普 {d['mean_sr']:.3f} | SR0 {d['sr0']:.4f} | "
                      f"deflated {d['mean_deflated']:+.3f} | p={d['mean_p']:.4f}")
            print(f"参数高原: {ctx['plateau']['verdict']} "
                  f"（{ctx['plateau']['flat_n']}/{ctx['plateau']['total_pairs']} 平坦）")
            print(f"门禁: {'✅ 通过' if ctx['gate']['pass'] else '❌ 不过'} — {ctx['gate']['reason']}")


if __name__ == "__main__":
    main()
