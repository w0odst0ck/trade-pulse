#!/usr/bin/env python3
"""cross_section_scan.py — M3 v2 两阶段横截面门禁扫描（08-18 深夜版）

阶段 1 快筛（quick）: 43 标的 compute_features + backtest（统一 --end 截止日）
  判定: 策略夏普 > 持有夏普 且 策略夏普 > 0 → 候选
阶段 2 门禁（gate）: 候选标的 walk_forward（105 trials）
  判定: OOS 正夏普折数 / DSR p / 参数高原平滑对
阶段 3 汇总（report）: cross_section_report.md（风格分组 + 三态结论）

用法:
  python3 tools/etf_scan/cross_section_scan.py --step all [--end 2026-08-11] [--parallel 3]
  python3 tools/etf_scan/cross_section_scan.py --step quick    # 只快筛
  python3 tools/etf_scan/cross_section_scan.py --step gate     # 只门禁（候选）
  python3 tools/etf_scan/cross_section_scan.py --step report   # 只汇总

约束: 只新增本文件；不修改任何生产代码；结果全部落 data/etf_scan/cs/（不污染 data/{s}/）。
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CS_DIR = PROJECT_ROOT / "data" / "etf_scan" / "cs"
COST = 0.00055
TRIALS = 105
PY = sys.executable
DP = PROJECT_ROOT / "tools" / "daily_pipeline"

# 候选池单一真源：scan_pool.py 的 ETF_POOL（code, name, group）
sys.path.insert(0, str(SCRIPT_DIR))
from scan_pool import ETF_POOL, BASE_SYMBOL  # noqa: E402

POOL = [(c, n, g) for c, n, g in ETF_POOL if c != BASE_SYMBOL]  # 42 池（588000 作基线单独跑）
ALL = [(BASE_SYMBOL, "科创50ETF(基线)", "宽基")] + POOL


def run(cmd: list, timeout: int = 1200) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def load_daily(symbol: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / symbol / "daily.csv")


def hold_sharpe(symbol: str, end: str) -> float:
    """持有不动年化夏普（无成本，截止 end 统一口径）"""
    df = load_daily(symbol)
    df = df[df["date"] <= end].copy()
    r = df["close"].pct_change().dropna()
    if len(r) < 60 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# 阶段 1：快筛
# ---------------------------------------------------------------------------
def quick_one(symbol: str, name: str, group: str, end: str) -> dict:
    out = CS_DIR / symbol
    out.mkdir(parents=True, exist_ok=True)
    qf = out / "quick.json"
    if qf.exists():  # 断点续跑
        return json.loads(qf.read_text())

    # 1a. 特征（无特征缓存时自动算；已有缓存直接复用）
    feat = DP / "compute_features.py"
    r = run([PY, str(feat), "--symbol", symbol])
    if r.returncode != 0:
        res = {"symbol": symbol, "name": name, "group": group, "error": "features",
               "stderr": r.stderr[-500:]}
        qf.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return res

    # 1b. 全段回测（输出到 cs 目录，不污染 data/{s}/backtest/）
    bt = DP / "backtest.py"
    r = run([PY, str(bt), "--symbol", symbol, "--cost", str(COST),
             "--end", end, "--output", str(out / "bt")], timeout=1800)
    if r.returncode != 0 or not (out / "bt" / "metrics.json").exists():
        res = {"symbol": symbol, "name": name, "group": group, "error": "backtest",
               "stderr": r.stderr[-500:]}
        qf.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return res

    m = json.loads((out / "bt" / "metrics.json").read_text())
    hs = hold_sharpe(symbol, end)
    strategy_sharpe = m.get("sharpe", 0)
    res = {
        "symbol": symbol, "name": name, "group": group,
        "strategy_sharpe": round(strategy_sharpe, 4),
        "hold_sharpe": round(hs, 4) if hs == hs else None,
        "annual_return": m.get("annual_return"),
        "max_drawdown": m.get("max_drawdown"),
        "win_rate": m.get("win_rate"),
        "trade_count": m.get("trade_count"),
        "edge": round(strategy_sharpe - hs, 4) if hs == hs else None,
        "candidate": bool(strategy_sharpe > 0 and (hs != hs or strategy_sharpe > hs)),
    }
    qf.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def quick(end: str, parallel: int) -> None:
    print(f"[quick] {len(ALL)} 标的快筛，截止 {end}，并行 {parallel}")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(quick_one, c, n, g, end): c for c, n, g in ALL}
        for fut in as_completed(futs):
            try:
                res = fut.result()
                tag = "✅候选" if res.get("candidate") else ("❌" if "error" not in res else f"⚠{res['error']}")
                print(f"  {res['symbol']} {res.get('name',''):12s} {tag}"
                      + (f"  策略 {res.get('strategy_sharpe')} vs 持有 {res.get('hold_sharpe')}"
                         if "error" not in res else f"  {res.get('stderr','')[-80:]}"))
            except Exception as e:  # noqa: BLE001
                print(f"  worker 异常: {e}")


def candidates() -> list:
    out = []
    for c, n, g in ALL:
        qf = CS_DIR / c / "quick.json"
        if qf.exists():
            q = json.loads(qf.read_text())
            if q.get("candidate"):
                out.append((c, n, g, q))
    return out


# ---------------------------------------------------------------------------
# 阶段 2：门禁（walk_forward）
# ---------------------------------------------------------------------------
def parse_wf_report(md: str) -> dict:
    """从 walk_forward 报告 md 解析：折表（训练/验证夏普）、DSR p、高原"""
    d = {"folds": [], "dsr_p": None, "plateau": None}
    for line in md.splitlines():
        if line.startswith("| 折 "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 9 and cells[0].startswith("折 "):
                d["folds"].append({
                    "fold": int(cells[0].split()[1]),
                    "train_sharpe": float(cells[6]),
                    "val_sharpe": float(cells[7]),
                })
        m = re.search(r"p = ([\d.]+)", line)
        if m:
            d["dsr_p"] = float(m.group(1))
        m = re.search(r"平滑相邻对 (\d+)/(\d+) → (\S+)", line)
        if m:
            d["plateau"] = {"flat": int(m.group(1)), "total": int(m.group(2)),
                            "verdict": m.group(3)}
    return d


def gate_one(symbol: str, name: str, group: str) -> dict:
    out = CS_DIR / symbol
    out.mkdir(parents=True, exist_ok=True)
    gf = out / "gate.json"
    if gf.exists():
        return json.loads(gf.read_text())

    wf = DP / "walk_forward.py"
    r = run([PY, str(wf), "--symbol", symbol, "--cost", str(COST),
             "--trials", str(TRIALS), "--report", str(out / "wf_report.md")],
            timeout=3600)
    if r.returncode != 0:
        res = {"symbol": symbol, "name": name, "group": group, "error": "walkforward",
               "stderr": r.stderr[-500:]}
        gf.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return res

    rep = (out / "wf_report.md")
    info = parse_wf_report(rep.read_text() if rep.exists() else r.stdout)
    vals = [f["val_sharpe"] for f in info["folds"]]
    info["oos_pos_folds"] = sum(1 for v in vals if v > 0)
    info["oos_neg_folds"] = sum(1 for v in vals if v <= 0)
    info["oos_mean_sharpe"] = round(float(np.mean(vals)), 4) if vals else None
    res = {"symbol": symbol, "name": name, "group": group, **info}
    gf.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def gate() -> None:
    cands = candidates()
    print(f"[gate] {len(cands)} 个候选过门禁验证")
    for c, n, g, _ in cands:
        res = gate_one(c, n, g)
        if "error" in res:
            print(f"  {c} {n:12s} ⚠{res['error']}  {res.get('stderr','')[-100:]}")
        else:
            print(f"  {c} {n:12s} OOS正折 {res['oos_pos_folds']}/5"
                  f"  均值 {res['oos_mean_sharpe']}  DSR p={res['dsr_p']}"
                  f"  高原 {res['plateau']}")


# ---------------------------------------------------------------------------
# 阶段 3：汇总报告
# ---------------------------------------------------------------------------
def report() -> str:
    rows = []
    for c, n, g in ALL:
        qf, gf = CS_DIR / c / "quick.json", CS_DIR / c / "gate.json"
        if not qf.exists():
            rows.append({"symbol": c, "name": n, "group": g, "status": "未跑"})
            continue
        q = json.loads(qf.read_text())
        if "error" in q:
            rows.append({"symbol": c, "name": n, "group": g, "status": f"❌ {q['error']}"})
            continue
        r = {"symbol": c, "name": n, "group": g, "strategy_sharpe": q["strategy_sharpe"],
             "hold_sharpe": q["hold_sharpe"], "annual": q["annual_return"],
             "dd": q["max_drawdown"], "trades": q["trade_count"],
             "status": "快筛候选" if q["candidate"] else "快筛淘汰"}
        if gf.exists():
            gg = json.loads(gf.read_text())
            if "error" not in gg:
                r["oos"] = f"{gg['oos_pos_folds']}/5"
                r["dsr_p"] = gg["dsr_p"]
                # 项目口径（ADR-0001）：OOS 正折≥3/5 且训练段非全负；DSR p 如实列示（基线 588000 亦 p=1.0，不作硬否决）
                train_neg_all = all(f["train_sharpe"] < 0 for f in gg["folds"])
                r["verdict"] = "✅门禁过" if (gg["oos_pos_folds"] >= 3 and not train_neg_all) else "❌门禁不过"
        rows.append(r)

    df = pd.DataFrame(rows)
    df = df.sort_values(["group", "strategy_sharpe"], ascending=[True, False]) \
        if "strategy_sharpe" in df.columns else df
    cols = [c for c in ["symbol", "name", "group", "strategy_sharpe", "hold_sharpe",
                        "annual", "dd", "trades", "oos", "dsr_p", "verdict", "status"] if c in df.columns]
    md = ["# M3 v2 横截面门禁扫描报告（2026-08-19）", "",
          f"- 生成: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
          f"- 池: {len(POOL)} ETF + 基线 588000（统一截止，成本万5.5）",
          f"- 门禁: 快筛(策略夏普>持有) → walk_forward 105 trials（OOS正折≥3/5 且训练段非全负，ADR-0001 口径；DSR p 如实列示）", "",
          df[cols].to_markdown(index=False), ""]
    passed = df[df["verdict"] == "✅门禁过"] if "verdict" in df.columns else df.iloc[0:0]
    md.append(f"## 结论（机器初判，人工复核）")
    md.append(f"- 过门禁: {len(passed)} 只（{', '.join(passed['symbol'])}）" if len(passed) else "- 过门禁: 0 只")
    md.append(f"- 口径说明: DSR p 如实列示；项目门禁口径 = OOS 正折≥3/5 且训练段非全负（ADR-0001，基线 588000 亦 p=1.0）")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["quick", "gate", "report", "all"], default="all")
    ap.add_argument("--end", default="2026-08-11", help="统一数据截止日（默认 08-11：42 标最新公共日）")
    ap.add_argument("--parallel", type=int, default=3)
    args = ap.parse_args()

    CS_DIR.mkdir(parents=True, exist_ok=True)
    if args.step in ("quick", "all"):
        quick(args.end, args.parallel)
    if args.step in ("gate", "all"):
        gate()
    if args.step in ("report", "all"):
        md = report()
        out = PROJECT_ROOT / "docs" / "M3v2横截面扫描报告_2026-08-19.md"
        out.write_text(md, encoding="utf-8")
        print(f"\n[report] → {out}\n")
        print(md[:3000])


if __name__ == "__main__":
    main()
