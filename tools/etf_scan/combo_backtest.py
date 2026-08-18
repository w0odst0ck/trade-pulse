#!/usr/bin/env python3
"""combo_backtest.py — M3 v2 组合回测（B 项，门禁出结果后跑）

输入: data/etf_scan/cs/{s}/bt/equity_curve.csv（各标策略净值，已含成本万5.5）
流程:
  1. 有效标的（gate.json 判定 ✅ 过门禁）→ 策略日收益序列
  2. 相关性去重: 两两日收益相关 > 0.8 → 每簇保留策略夏普最高者
  3. 权重: 等权 / 波动率倒数加权（策略日收益 vol 倒数）
  4. 组合指标: 年化/夏普/回撤 vs 单标 588000
  5. 报告: docs/M3v2组合回测_2026-08-19.md

用法: python3 tools/etf_scan/combo_backtest.py
约束: 只新增本文件；只读 cs/ 产物；不碰生产代码。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CS_DIR = PROJECT_ROOT / "data" / "etf_scan" / "cs"
OUT = PROJECT_ROOT / "docs" / "M3v2组合回测_2026-08-19.md"
CORR_CAP = 0.8
TRADING_DAYS = 252


def passed_symbols() -> list:
    """gate.json 判定 ✅ 的标的（OOS 正折 ≥3/5 且训练段非全负，ADR-0001 项目口径）"""
    out = []
    for cs in sorted(CS_DIR.iterdir()):
        gf = cs / "gate.json"
        if not gf.exists():
            continue
        g = json.loads(gf.read_text())
        if "error" not in g and g.get("oos_pos_folds", 0) >= 3 and \
                not all(f["train_sharpe"] < 0 for f in g.get("folds", [])):
            q = json.loads((cs / "quick.json").read_text())
            out.append({"symbol": cs.name, "name": q.get("name", cs.name),
                        "group": q.get("group", ""),
                        "strategy_sharpe": q.get("strategy_sharpe")})
    return out


def strategy_returns(symbol: str) -> pd.Series:
    df = pd.read_csv(CS_DIR / symbol / "bt" / "equity_curve.csv")
    df = df.drop_duplicates("date", keep="last").set_index("date")
    return df["equity"].pct_change().dropna()


def dedupe(syms: list) -> list:
    """相关性 > CORR_CAP 的簇：保留策略夏普最高者"""
    if len(syms) <= 1:
        return syms
    rets = {s["symbol"]: strategy_returns(s["symbol"]) for s in syms}
    joined = pd.DataFrame(rets).dropna()
    corr = joined.corr()
    keep, drop = [], set()
    order = sorted(syms, key=lambda s: -s["strategy_sharpe"])
    for s in order:
        if s["symbol"] in drop:
            continue
        keep.append(s)
        for o in order:
            if o["symbol"] != s["symbol"] and o["symbol"] not in drop and \
                    corr.loc[s["symbol"], o["symbol"]] > CORR_CAP:
                drop.add(o["symbol"])
    return keep


def combo_metrics(rets: pd.DataFrame, weights: np.ndarray) -> dict:
    r = rets @ weights
    ann = r.mean() * TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann / vol if vol > 0 else 0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"annual": round(ann * 100, 2), "sharpe": round(float(sharpe), 4),
            "vol": round(vol * 100, 2), "dd": round(dd * 100, 2)}


def main():
    syms = passed_symbols()
    if len(syms) < 2:
        print(f"[combo] 过门禁标的 {len(syms)} < 2，无需组合回测（多标不成立）")
        (OUT.parent / "M3v2组合回测_2026-08-19.md").write_text(
            f"# M3 v2 组合回测（2026-08-19）\n\n过门禁标的 {len(syms)} 只，不足 2 只——多标组合不成立，维持单标 588000。\n",
            encoding="utf-8")
        return

    print(f"[combo] 过门禁 {len(syms)} 只: {[s['symbol'] for s in syms]}")
    keep = dedupe(syms)
    dropped = [s["symbol"] for s in syms if s not in keep]
    print(f"[combo] 相关性去重后 {len(keep)} 只: {[s['symbol'] for s in keep]}" +
          (f"（剔除 {dropped}）" if dropped else ""))

    # 高置信子集：参数高原稳健（verdict 含「稳健」）
    hi = []
    for s in keep:
        gf = CS_DIR / s["symbol"] / "gate.json"
        if gf.exists():
            g = json.loads(gf.read_text())
            if g.get("plateau") and "稳健" in str(g["plateau"].get("verdict", "")):
                hi.append(s)
    print(f"[combo] 其中参数稳健 {len(hi)} 只: {[s['symbol'] for s in hi]}")

    rets = pd.DataFrame({s["symbol"]: strategy_returns(s["symbol"]) for s in keep}).dropna()
    vol = rets.std()
    eq_w = np.full(len(keep), 1 / len(keep))
    iv_w = (1 / vol) / (1 / vol).sum()

    eq_m = combo_metrics(rets, eq_w)
    iv_m = combo_metrics(rets, iv_w.to_numpy())

    lines = ["# M3 v2 组合回测（2026-08-19）", "",
             f"- 有效标的: {[s['symbol'] for s in keep]}（相关性去重后，剔除 {dropped or '无'}）",
             f"- 组合期: {rets.index[0]} ~ {rets.index[-1]}（{len(rets)} 交易日，策略净值已含成本万5.5）", "",
             "| 组合 | 年化 | 夏普 | 波动 | 回撤 |", "|---|---|---|---|---|",
             f"| 等权（全部 {len(keep)} 只） | {eq_m['annual']}% | {eq_m['sharpe']} | {eq_m['vol']}% | {eq_m['dd']}% |",
             f"| 波动率倒数加权（全部 {len(keep)} 只） | {iv_m['annual']}% | {iv_m['sharpe']} | {iv_m['vol']}% | {iv_m['dd']}% |"]
    if hi:
        rets_hi = pd.DataFrame({s["symbol"]: strategy_returns(s["symbol"]) for s in hi}).dropna()
        eq_hi = combo_metrics(rets_hi, np.full(len(hi), 1 / len(hi)))
        lines += ["", f"| 等权（仅参数稳健 {len(hi)} 只） | {eq_hi['annual']}% | {eq_hi['sharpe']} | {eq_hi['vol']}% | {eq_hi['dd']}% |"]

    lines += ["", "## 对比", "",
              "| 参照 | 年化 | 夏普 | 回撤 |", "|---|---|---|---|"]
    if any(s["symbol"] == "588000" for s in keep):
        b = [s for s in keep if s["symbol"] == "588000"][0]
        rb = rets["588000"]
        lines += [f"| 单标 588000（组合内） | {combo_metrics(pd.DataFrame({'x': rb}), np.array([1.0]))['annual']}% | "
                  f"{combo_metrics(pd.DataFrame({'x': rb}), np.array([1.0]))['sharpe']} | "
                  f"{combo_metrics(pd.DataFrame({'x': rb}), np.array([1.0]))['dd']}% |"]
    else:
        qb = json.loads((CS_DIR / "588000" / "quick.json").read_text())
        lines.append(f"| 单标 588000（快筛口径） | — | {qb.get('strategy_sharpe')} | — |")
    lines += ["", "## 结论（机器初判）", "",
              f"- 组合夏普最高 {max(eq_m['sharpe'], iv_m['sharpe'])}（"
              f"{'等权' if eq_m['sharpe'] >= iv_m['sharpe'] else '波动率倒数加权'}）",
              f"- 多标增益判定: {'✅ 组合 > 单标，多标可行' if max(eq_m['sharpe'], iv_m['sharpe']) > 0.69 else '⚠️ 组合未跑赢单标基线 0.69，多标无增益'}",
              "", f"- 详细净值/相关性矩阵: data/etf_scan/cs/ 下各标产物"]
    txt = "\n".join(lines)
    OUT.write_text(txt, encoding="utf-8")
    print(f"\n[combo] 报告 → {OUT}")
    print(txt)


if __name__ == "__main__":
    main()
