#!/usr/bin/env python3
"""
portfolio_tools_compare.py — 组合优化工具实测对比

场景：6 只 ETF（588000 + 5 备选），万元级，日线择时
对比：
  1. 等权（基准）
  2. 波动率倒数加权（自实现，10 行）
  3. PyPortfolioOpt：MVO(最大夏普) + HRP
  4. Riskfolio-Lib：HRP + 最大夏普

判断标准：
  - 权重合理性（会不会极端集中）
  - API 复杂度（几行出结果）
  - 依赖重量
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RETS = pd.read_csv("/tmp/etf_rets_clean.csv", index_col=0, parse_dates=True)
# 兜底：剔除 Inf/NaN（PyPortfolioOpt 内部计算可能产生）
RETS = RETS.replace([np.inf, -np.inf], np.nan).dropna()
SYMS = list(RETS.columns)


def equal_weight():
    return pd.Series(1.0 / len(SYMS), index=SYMS)


def inv_vol_weight():
    """波动率倒数加权（自实现）"""
    vol = RETS.std()
    w = 1.0 / vol
    return w / w.sum()


def pypfopt_mvo():
    """PyPortfolioOpt 最大夏普"""
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    # returns_data=True 关键：直接吃收益序列，避免内部价格推断出 NaN
    mu = expected_returns.mean_historical_return(RETS, returns_data=True)
    S = risk_models.sample_cov(RETS, returns_data=True)
    ef = EfficientFrontier(mu, S)
    try:
        w = ef.max_sharpe()
        return pd.Series(w)
    except Exception as e:
        print(f"    [pypfopt MVO] {type(e).__name__}: {str(e)[:80]}")
        return pd.Series({s: np.nan for s in SYMS}, dtype=float)


def pypfopt_hrp():
    """PyPortfolioOpt HRP（用 scipy 兼容的方式）"""
    from pypfopt import HRPOpt, risk_models
    rets = RETS.copy()
    S = risk_models.sample_cov(rets, returns_data=True)
    hrp = HRPOpt(rets, S)
    try:
        w = hrp.optimize()
        return pd.Series(w)
    except Exception as e:
        print(f"    [pypfopt HRP] {type(e).__name__}: {str(e)[:80]}")
        return pd.Series({s: np.nan for s in SYMS}, dtype=float)


def riskfolio_hrp():
    """Riskfolio-Lib HRP（Portfolio 类 + optimization）"""
    import riskfolio as rp
    Y = RETS.copy()
    try:
        port = rp.Portfolio(returns=Y)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.optimization(model="HRP", rm="MV", obj="Sharpe", hist=True)
        return pd.Series(w["weights"].values, index=w.index)
    except Exception as e:
        print(f"    [riskfolio HRP] {type(e).__name__}: {str(e)[:80]}")
        return pd.Series({s: np.nan for s in SYMS}, dtype=float)


def riskfolio_mvo():
    """Riskfolio-Lib 最大夏普（经典 MVO）"""
    import riskfolio as rp
    Y = RETS.copy()
    try:
        port = rp.Portfolio(returns=Y)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.optimization(model="Classic", rm="MV", obj="Sharpe", hist=True)
        return pd.Series(w["weights"].values, index=w.index)
    except Exception as e:
        print(f"    [riskfolio MVO] {type(e).__name__}: {str(e)[:80]}")
        return pd.Series({s: np.nan for s in SYMS}, dtype=float)


def portfolio_stats(weights: pd.Series, label: str):
    """组合绩效：年化收益/波动/夏普/最大回撤"""
    rets = RETS[weights.index] if len(weights.index) == len(SYMS) else RETS
    # 对齐
    w = weights.reindex(rets.columns).fillna(0)
    port_ret = (rets * w).sum(axis=1)
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + port_ret).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {"label": label, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_dd": dd, "n_nonzero": int((w > 0.01).sum())}


def main():
    print(f"═══ 组合优化工具对比（{len(SYMS)} 只 ETF, 138 天）═══\n")

    methods = [
        ("等权(基准)", equal_weight),
        ("波动率倒数(自实现)", inv_vol_weight),
        ("PyPortfolioOpt MVO", pypfopt_mvo),
        ("PyPortfolioOpt HRP", pypfopt_hrp),
        ("Riskfolio HRP", riskfolio_hrp),
        ("Riskfolio MVO(上限30%)", riskfolio_mvo),
    ]

    results = []
    for label, fn in methods:
        try:
            w = fn()
            stats = portfolio_stats(w, label)
            results.append((label, w, stats))
            print(f"\n── {label} ──")
            for s in SYMS:
                val = w.get(s, 0)
                if val > 0.01:
                    print(f"   {s}: {val*100:5.1f}%")
            print(f"   年化 {stats['ann_ret']*100:+.1f}% | 波动 {stats['ann_vol']*100:.1f}% | 夏普 {stats['sharpe']:.3f} | 回撤 {stats['max_dd']*100:.1f}% | 持仓 {stats['n_nonzero']} 只")
        except Exception as e:
            print(f"\n── {label} ── FAIL: {str(e)[:100]}")

    # 汇总表
    print(f"\n═══ 汇总 ═══")
    print(f"  {'方法':<28s}{'年化':>8s}{'波动':>8s}{'夏普':>8s}{'回撤':>9s}{'持仓':>6s}")
    print(f"  {'-'*68}")
    for label, w, s in results:
        print(f"  {label:<28s}{s['ann_ret']*100:>+7.1f}%{s['ann_vol']*100:>7.1f}%{s['sharpe']:>8.3f}{s['max_dd']*100:>8.1f}%{s['n_nonzero']:>6d}")


if __name__ == "__main__":
    main()
