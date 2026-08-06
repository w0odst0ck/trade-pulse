#!/usr/bin/env python3
"""scan_pool.py — 多标的套利候选池 · 阶段 1：批量拉取 + 层次聚类 + 协整预筛

纯数据预筛脚本（只分析，不产出策略结论；阶段 2 才做回测）：
  1. fetch   批量拉取候选 ETF 日线（腾讯主源，经 fetch_data 复用生产数据链路）
  2. cluster 全标的对齐交易日 → 日收益相关性 → ward 层次聚类 → 树状图 + 簇分组
  3. coint   每只候选 vs 588000：Engle-Granger 两步 + ADF + 半衰期 + z-score
             + 全段 3 等分滚动稳定性（3 段均 p<0.05 才算「稳定协整对」）
  4. report  生成 data/etf_scan/scan_report.md

用法（用 kronos venv 的 python，已装 pandas/scipy/statsmodels/matplotlib）：
  tools/kronos/.venv/bin/python tools/etf_scan/scan_pool.py --step all
  tools/kronos/.venv/bin/python tools/etf_scan/scan_pool.py --step fetch
  tools/kronos/.venv/bin/python tools/etf_scan/scan_pool.py --step cluster --n-clusters 8
  tools/kronos/.venv/bin/python tools/etf_scan/scan_pool.py --step coint
  tools/kronos/.venv/bin/python tools/etf_scan/scan_pool.py --step report
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "etf_scan"

# ---------------------------------------------------------------------------
# 候选池（任务列表逐组核对共 42 只；标题「40 只」为笔误，以列表为准）
# 每项: (symbol, 中文名, 风格分组)
# ---------------------------------------------------------------------------
ETF_POOL: List[Tuple[str, str, str]] = [
    # 宽基
    ("510300", "沪深300", "宽基"),
    ("510500", "中证500", "宽基"),
    ("512100", "中证1000", "宽基"),
    ("159915", "创业板", "宽基"),
    ("588090", "科创100", "宽基"),
    ("159781", "双创50", "宽基"),
    # 科创/成长
    ("588080", "科创50增强", "科创/成长"),
    ("588200", "科创芯片", "科创/成长"),
    ("159819", "人工智能", "科创/成长"),
    ("562500", "机器人", "科创/成长"),
    # 半导体
    ("512480", "半导体", "半导体"),
    ("159995", "芯片", "半导体"),
    ("512760", "芯片龙头", "半导体"),
    # 新能源
    ("516160", "新能源", "新能源"),
    ("515790", "光伏", "新能源"),
    ("515030", "新能源车", "新能源"),
    ("159875", "新能源车龙头", "新能源"),
    # 红利/价值
    ("510880", "红利", "红利/价值"),
    ("515080", "中证红利", "红利/价值"),
    ("512890", "红利低波", "红利/价值"),
    ("512530", "红利低波100", "红利/价值"),
    # 银行/金融
    ("512800", "银行", "银行/金融"),
    ("512880", "证券", "银行/金融"),
    ("512000", "券商", "银行/金融"),
    ("512070", "非银金融", "银行/金融"),
    ("510230", "金融", "银行/金融"),
    # 军工
    ("512660", "军工", "军工"),
    ("512710", "军工龙头", "军工"),
    # 医药
    ("512010", "医药", "医药"),
    ("159929", "医药龙头", "医药"),
    ("512170", "医疗", "医药"),
    # 消费
    ("510150", "消费", "消费"),
    ("512690", "白酒", "消费"),
    ("159928", "消费龙头", "消费"),
    # 资源
    ("512400", "有色", "资源"),
    ("515220", "煤炭", "资源"),
    # 地产/基建
    ("512200", "房地产", "地产/基建"),
    ("516950", "基建", "地产/基建"),
    # 通信/传媒
    ("515880", "通信", "通信/传媒"),
    ("512980", "传媒", "通信/传媒"),
    # 跨境
    ("513180", "恒生科技", "跨境"),
    ("513050", "中概互联", "跨境"),
]

BASE_SYMBOL = "588000"          # 配对基准：科创50 ETF
START_DATE = "2023-01-01"       # 统一回看起点
ADF_PVAL = 0.05                 # 协整显著性阈值
N_SEGMENTS = 3                  # 滚动稳定性分段数
MIN_ROWS = 500                  # 拉取行数下限：低于视为残缺（baostock 服务端只回近 ~7 个月数据）

# 名称索引
POOL_NAME = {s: n for s, n, _ in ETF_POOL}
POOL_GROUP = {s: g for s, _, g in ETF_POOL}


# ---------------------------------------------------------------------------
# 第 1 步：批量拉取
# ---------------------------------------------------------------------------
def _init_fetch_data():
    """以 sys.path 方式引入生产数据链路的 fetch_data（不改动生产代码）"""
    sys.path.insert(0, str(PROJECT_ROOT / "tools" / "daily_pipeline"))
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from fetch_data import fetch_data  # noqa: F401
    return fetch_data


def fetch_all(force: bool = True, only_missing: bool = False) -> Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]]:
    """拉取候选池 + 基准 588000，存 data/etf_scan/{symbol}.csv

    only_missing=True 时只补拉 etf_scan 下缺失（上次失败）的标的，用于
    腾讯限流后的补拉；否则全量拉取。
    返回 {symbol: (df 或 None, 失败原因或 None)}。单只失败不中断整体。
    """
    fetch_data = _init_fetch_data()
    results: Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]] = {}
    symbols = [s for s, _, _ in ETF_POOL] + [BASE_SYMBOL]
    if only_missing:
        symbols = [s for s in symbols if not (OUT_DIR / f"{s}.csv").exists()]
    if not symbols:
        print("  [INFO] 无缺失标的（etf_scan 数据已齐），跳过拉取")
        return results
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    total = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        name = POOL_NAME.get(symbol, "科创50ETF(基准)")
        print(f"\n[{i}/{total}] 拉取 {symbol} {name} ...", flush=True)
        t0 = time.time()
        try:
            df = fetch_data(symbol, START_DATE, force=force, provider_name="tencent")
            if df is None or len(df) == 0:
                raise RuntimeError("fetch_data 返回空数据")
            if len(df) < MIN_ROWS:
                # baostock fallback 在部分环境只返回近 ~7 个月数据（残缺），
                # 不落盘，按失败处理以便 --retry-failed 自动补拉
                raise RuntimeError(
                    f"数据残缺: 仅 {len(df)} 条 < {MIN_ROWS}（baostock 近期数据限制或源异常），不落盘"
                )
            df.to_csv(OUT_DIR / f"{symbol}.csv", index=False)
            results[symbol] = (df, None)
            ok += 1
            print(f"  ✓ {symbol}: {len(df)} 条, 用时 {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            results[symbol] = (None, str(e))
            fail += 1
            print(f"  ✗ {symbol}: 失败 - {e}", flush=True)
        # 请求间隔：腾讯对连续高频请求会限流（实测 ~800 请求后开始拒绝），
        # 用较长间隔降低触发概率；补拉轮次用更保守的间隔
        time.sleep(1.5 if force else 0.5)

    print(f"\n===== 拉取汇总: 成功 {ok} / 失败 {fail} =====")
    for symbol, (df, err) in results.items():
        if err is not None:
            print(f"  FAIL {symbol} {POOL_NAME.get(symbol, '')}: {err}")
    return results


def build_disk_summary() -> Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]]:
    """按 etf_scan 磁盘状态重建拉取汇总（存在且行数达标视为成功）"""
    summary: Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]] = {}
    for symbol in [s for s, _, _ in ETF_POOL] + [BASE_SYMBOL]:
        path = OUT_DIR / f"{symbol}.csv"
        if not path.exists():
            summary[symbol] = (None, "缺少数据文件（未拉取）")
        else:
            try:
                n = sum(1 for _ in open(path, encoding="utf-8")) - 1
                if n < MIN_ROWS:
                    summary[symbol] = (None, f"数据残缺: 仅 {n} 条 < {MIN_ROWS}")
                else:
                    summary[symbol] = (None, None)
            except OSError as e:
                summary[symbol] = (None, f"读取失败: {e}")
    return summary


# ---------------------------------------------------------------------------
# 数据读取与对齐
# ---------------------------------------------------------------------------
def load_closes(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """读取 etf_scan 下各标的 close，返回 DataFrame(行=date, 列=symbol)"""
    symbols = symbols or ([s for s, _, _ in ETF_POOL] + [BASE_SYMBOL])
    closes = {}
    for symbol in symbols:
        path = OUT_DIR / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}，请先执行 --step fetch")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.dropna(subset=["close"]).sort_values("date")
        closes[symbol] = df.set_index("date")["close"]
    return pd.DataFrame(closes).dropna(how="any")


# ---------------------------------------------------------------------------
# 第 2 步：层次聚类
# ---------------------------------------------------------------------------
def cluster_analysis(n_clusters: int = 8) -> Dict:
    """全标的（候选 + 基准）对齐 → 日收益相关 → ward 层次聚类 → 树状图 + 簇分组"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import squareform

    # 注册中文字体（文泉驿微米黑），中文标签不乱码
    zh_fonts = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in zh_fonts:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    matplotlib.rcParams["font.family"] = "WenQuanYi Micro Hei"
    matplotlib.rcParams["axes.unicode_minus"] = False

    closes = load_closes()
    all_symbols = list(closes.columns)
    rets = closes.pct_change().dropna()
    corr = rets.corr().fillna(0.0)  # 防单标的 NaN 收益列使整矩阵 NaN 而崩溃

    # 相关矩阵 → 欧氏距离（ward 适用）：d = sqrt(2*(1-corr))
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="ward")

    labels = [f"{s} {POOL_NAME.get(s, '基准')}" for s in all_symbols]

    # 树状图（竖排，标签可读）
    fig, ax = plt.subplots(figsize=(15, 12), dpi=150)
    dendrogram(
        Z, labels=labels, orientation="left", ax=ax,
        leaf_font_size=8, color_threshold=None,
    )
    ax.set_title(f"ETF 日收益相关性层次聚类（ward，{len(all_symbols)} 标的，"
                 f"{START_DATE} 起 {len(rets)} 个交易日）")
    ax.set_xlabel("距离 (sqrt(2*(1-corr)))")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 树状图已保存: {OUT_DIR / 'cluster.png'}")

    # 簇分组（maxclust 切分）
    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")
    groups: Dict[int, List[str]] = {}
    for symbol, cid in zip(all_symbols, cluster_ids):
        groups.setdefault(int(cid), []).append(symbol)

    order = sorted(groups.keys())
    print(f"\n===== 簇分组（{len(order)} 簇, n_clusters={n_clusters}）=====")
    for cid in order:
        members = groups[cid]
        names = " | ".join(f"{s} {POOL_NAME.get(s, '基准')}" for s in members)
        print(f"簇 {cid} (n={len(members)}): {names}")

    return {"Z": Z, "groups": groups, "corr": corr, "rets": rets, "labels": labels}


# ---------------------------------------------------------------------------
# 第 3 步：pairwise 协整预筛（vs 588000）
# ---------------------------------------------------------------------------
def _eg_resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Engle-Granger 第一步：log(y) ~ log(x) OLS，返回残差"""
    import statsmodels.api as sm
    X = sm.add_constant(np.log(x))
    model = sm.OLS(np.log(y), X).fit()
    return model.resid


def _adf_pvalue(resid: np.ndarray) -> float:
    """ADF 检验 p 值（statsmodels，AIC 定阶，带常数项）"""
    import statsmodels.api as sm
    try:
        result = sm.tsa.adfuller(resid, autolag="AIC", regression="c")
        return float(result[1])
    except Exception:
        return np.nan


def _half_life_days(resid: np.ndarray) -> Optional[float]:
    """价差半衰期：残差 AR(1) 系数 rho → -ln2/ln(rho)（单位：交易日）

    rho<=0（不均值回归）或 rho>=1（发散/单位根）时无意义，返回 None。
    """
    r0 = resid[:-1]
    r1 = resid[1:]
    denom = float(np.sum(r0 * r0))
    if denom <= 0 or len(r0) < 2:
        return None
    rho = float(np.sum(r0 * r1)) / denom
    if rho <= 0 or rho >= 1:
        return None
    return -math.log(2.0) / math.log(rho)


def cointegration_screen() -> pd.DataFrame:
    """每只候选 vs 588000 的 EG 两步协整检验 + 全段 3 等分滚动稳定性"""
    closes = load_closes()
    base = closes[BASE_SYMBOL].values
    base_dates = closes.index

    rows = []
    for symbol, name, _ in ETF_POOL:
        y = closes[symbol].values
        n = len(y)
        if n < 30:
            rows.append({"symbol": symbol, "name": name, "p_full": np.nan,
                         "p1": np.nan, "p2": np.nan, "p3": np.nan,
                         "half_life_days": None, "z_score": None, "verdict": "数据不足"})
            continue

        # 全段 EG
        resid_full = _eg_resid(y, base)
        p_full = _adf_pvalue(resid_full)
        hl = _half_life_days(resid_full)
        z = float(resid_full[-1] / resid_full.std()) if resid_full.std() > 0 else None

        # 全段 3 等分（每段约 n/3 个交易日，末段吸收余数）
        seg_ps = []
        for k in range(N_SEGMENTS):
            lo = (n * k) // N_SEGMENTS
            hi = (n * (k + 1)) // N_SEGMENTS
            if hi - lo < 20:
                seg_ps.append(np.nan)
                continue
            resid_seg = _eg_resid(y[lo:hi], base[lo:hi])
            seg_ps.append(_adf_pvalue(resid_seg))

        p1, p2, p3 = seg_ps
        if any(math.isnan(p) for p in seg_ps):
            verdict = "不稳定"  # 存在 NaN 段（样本不足）按不稳定处理
        elif all(p < ADF_PVAL for p in seg_ps):
            verdict = "稳定"
        elif p_full >= ADF_PVAL and all(p >= ADF_PVAL for p in seg_ps):
            verdict = "无协整"
        else:
            verdict = "不稳定"

        rows.append({"symbol": symbol, "name": name, "p_full": p_full,
                     "p1": p1, "p2": p2, "p3": p3,
                     "half_life_days": hl, "z_score": z, "verdict": verdict})

    df = pd.DataFrame(rows)
    order = {"稳定": 0, "不稳定": 1, "无协整": 2, "数据不足": 3}
    df["_ord"] = df["verdict"].map(order)
    df["_sort"] = df["p_full"].fillna(1.0)
    df = df.sort_values(["_ord", "_sort"], ascending=[True, True]).drop(
        columns=["_ord", "_sort"]).reset_index(drop=True)
    return df


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def fmt_hl(hl: Optional[float]) -> str:
    if hl is None or math.isnan(hl):
        return "—"
    return f"{hl:.1f}"


def fmt_z(z: Optional[float]) -> str:
    if z is None or math.isnan(z):
        return "—"
    return f"{z:+.2f}"


# ---------------------------------------------------------------------------
# 第 4 步：报告
# ---------------------------------------------------------------------------
def write_report(fetch_summary: Dict, cluster_res: Optional[Dict],
                 coint_df: Optional[pd.DataFrame]) -> Path:
    lines: List[str] = []
    add = lines.append
    add("# ETF 套利候选池 · 阶段 1 数据预筛报告\n")
    add(f"- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"- 数据源: 腾讯日线（fetch_data 生产链路, provider=tencent），区间 {START_DATE} 起")
    add(f"- 候选标的: {len(ETF_POOL)} 只（任务列表 42 只，标题 40 为笔误，以列表为准），"
        f"基准 588000（科创50）")
    add(f"- 方法: 日收益相关性 ward 层次聚类；Engle-Granger 两步法"
        f"（log 价 OLS 残差 + ADF, p<{ADF_PVAL}）+ 全段 3 等分滚动稳定性（3 段均 p<{ADF_PVAL} 为稳定）\n")

    # --- 拉取汇总 ---
    add("## 1. 数据拉取汇总\n")
    ok = sum(1 for _, err in fetch_summary.values() if err is None)
    fail = len(fetch_summary) - ok
    add(f"- 成功 {ok} / 失败 {fail}（候选 {len(ETF_POOL)} + 基准 588000）")
    if fail:
        add("\n| 标的 | 名称 | 失败原因 |")
        add("|:--|:--|:--|")
        for symbol, (_, err) in fetch_summary.items():
            if err is not None:
                add(f"| {symbol} | {POOL_NAME.get(symbol, '基准')} | {err} |")
    add("")
    # 拉取实况说明：基于磁盘实际数据动态生成
    try:
        ref = pd.read_csv(OUT_DIR / f"{BASE_SYMBOL}.csv", parse_dates=["date"])
        ref_n = len(ref)
        ref_start = ref["date"].min().strftime("%Y-%m-%d")
        ref_end = ref["date"].max().strftime("%Y-%m-%d")
        add(f"> 数据实况：每只 {ref_n} 条（{ref_start} ~ {ref_end}）。"
            "腾讯源对连续高频请求会限流（返回非 JSON 即触发，此时自动走 fallback 链 "
            "akshare/eastmoney 重试补齐）；脚本对 <500 条的结果判定为残缺不落盘"
            "（baostock 服务端近期只回约 7 个月数据）。各标的最终数据源可能为"
            "腾讯/akshare/eastmoney 混合，均为前复权日线、归一化后同构，"
            "对基于收益率的聚类与协整分析无实质影响。\n")
    except Exception:
        add("> 数据实况：etf_scan 目录数据缺失或不可读，请先执行 `--step fetch`。\n")

    # --- 聚类 ---
    add("## 2. 相关性层次聚类\n")
    add("树状图: `data/etf_scan/cluster.png`（候选 + 基准共 "
        f"{len(cluster_res['groups'])} 簇… 见下表）\n" if cluster_res else "")
    if cluster_res:
        groups = cluster_res["groups"]
        add(f"簇数: {len(groups)}（--n-clusters 切分）\n")
        add("| 簇 | 标的数 | 标的 | 主要风格 |")
        add("|:--|:--|:--|:--|")
        for cid in sorted(groups.keys()):
            members = groups[cid]
            names = ", ".join(f"{s} {POOL_NAME.get(s, '基准')}" for s in members)
            # 主要风格 = 成员中占比最高的任务分组
            grp_counts: Dict[str, int] = {}
            for s in members:
                g = POOL_GROUP.get(s, "基准")
                grp_counts[g] = grp_counts.get(g, 0) + 1
            main_style = max(grp_counts.items(), key=lambda kv: kv[1])[0]
            add(f"| {cid} | {len(members)} | {names} | {main_style} |")
        add("")
        # 基准 588000 所在簇
        for cid, members in groups.items():
            if BASE_SYMBOL in members:
                add(f"- 基准 588000 位于**簇 {cid}**，同簇标的: "
                    f"{', '.join(s for s in members if s != BASE_SYMBOL)}\n")
                break

    # --- 协整表 ---
    add("## 3. 与 588000 的 pairwise 协整预筛\n")
    add("判定：**稳定** = 全段 3 等分各段 ADF 均 p<0.05；**不稳定** = 全段有协整但分段有破裂"
        "（或某段有协整而全段无）；**无协整** = 全段与各段均 p≥0.05。"
        "半衰期 = 残差 AR(1) 系数换算（交易日）。z-score = 全段残差末端相对全段分布。\n")
    if coint_df is not None:
        add("| 标的 | 名称 | 全段 p | 段1 p | 段2 p | 段3 p | 半衰期(天) | z-score | 结论 |")
        add("|:--|:--|--:|--:|--:|--:|--:|--:|:--|")
        for _, r in coint_df.iterrows():
            add(f"| {r['symbol']} | {r['name']} | {fmt_p(r['p_full'])} | "
                f"{fmt_p(r['p1'])} | {fmt_p(r['p2'])} | {fmt_p(r['p3'])} | "
                f"{fmt_hl(r['half_life_days'])} | {fmt_z(r['z_score'])} | {r['verdict']} |")
        add("")

        # --- 预筛结论 ---
        add("## 4. 预筛结论\n")
        stable = coint_df[coint_df["verdict"] == "稳定"]
        unstable = coint_df[coint_df["verdict"] == "不稳定"]
        none_ = coint_df[coint_df["verdict"] == "无协整"]
        add(f"- 稳定协整对: **{len(stable)}** 只；不稳定: {len(unstable)} 只；无协整: {len(none_)} 只\n")
        if len(stable) > 0:
            add("**值得进入阶段 2 配对实验的标的**（与 588000 全段及 3 等分段均协整稳定）:\n")
            for _, r in stable.iterrows():
                add(f"- `{r['symbol']} {r['name']}` — 全段 p={fmt_p(r['p_full'])}, "
                    f"半衰期 {fmt_hl(r['half_life_days'])} 天, "
                    f"当前 z-score {fmt_z(r['z_score'])}（段 p: {fmt_p(r['p1'])}/"
                    f"{fmt_p(r['p2'])}/{fmt_p(r['p3'])}）")
            add("")
            add("阶段 2 建议：对上述标的与 588000 的价差做配对回测（阈值开平仓、"
                "半衰期校准、手续费与滑点敏感性），并交叉验证聚类中与 588000 同簇的标的。")
        else:
            add("**S1 配对方案不成立**：候选池中无任何标的与 588000 形成全段 3 等分均稳定的"
                "协整对（若仅看全段 p，部分标的有协整但分段破裂，配对的持续有效性无法保证）。")
            add("建议只走 S2 轮动方向（多标的动量/趋势轮动），不再投入配对回测资源。\n")
        add("> 说明：本报告为阶段 1 纯数据预筛，未做任何回测；策略结论留待阶段 2。")

    out = OUT_DIR / "scan_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] 报告已生成: {out}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ETF 套利候选池阶段 1 预筛")
    parser.add_argument("--step", default="all",
                        choices=["all", "fetch", "cluster", "coint", "report"],
                        help="执行步骤（默认 all）")
    parser.add_argument("--n-clusters", type=int, default=8,
                        help="层次聚类簇数（默认 8）")
    parser.add_argument("--no-force", action="store_true",
                        help="拉取时不做全量重拉（默认 force=True 全量）")
    parser.add_argument("--retry-failed", action="store_true",
                        help="只补拉 etf_scan 下缺失的标的（腾讯限流失败后重试）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fetch_summary: Dict = {}
    cluster_res = None
    coint_df = None

    if args.step in ("all", "fetch"):
        # --step all 只补缺失标的（幂等）；--step fetch 单独跑默认全量
        fetch_summary = fetch_all(force=not args.no_force,
                                  only_missing=args.retry_failed or args.step == "all")
    else:
        # 非 fetch 步骤：按磁盘状态重建汇总
        fetch_summary = build_disk_summary()

    # 合并磁盘状态：本轮未触及的标的按磁盘汇总补齐（幂等重跑时报告仍显示全量状态）
    disk_summary = build_disk_summary()
    for sym, (_, err) in disk_summary.items():
        if sym not in fetch_summary:
            fetch_summary[sym] = (None, err)

    if args.step in ("all", "cluster"):
        cluster_res = cluster_analysis(n_clusters=args.n_clusters)
    if args.step in ("all", "coint"):
        coint_df = cointegration_screen()
        print("\n===== 协整预筛表（vs 588000, 按结论+全段 p 排序）=====")
        print(coint_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.step in ("all", "report"):
        # report 单独跑时自动重建聚类/协整结果（秒级，幂等）
        if cluster_res is None:
            cluster_res = cluster_analysis(n_clusters=args.n_clusters)
        if coint_df is None:
            coint_df = cointegration_screen()
        write_report(fetch_summary, cluster_res, coint_df)


if __name__ == "__main__":
    main()
