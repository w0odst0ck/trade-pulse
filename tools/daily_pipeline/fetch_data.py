#!/usr/bin/env python3
"""
fetch_data.py — 588000 日线数据拉取（Provider 版本）

通过 DataProvider 抽象接口获取数据，支持自动降级。

用法：
  python fetch_data.py                    # 增量更新
  python fetch_data.py --force            # 强制全量拉取
  python fetch_data.py --start 2024-01-01 # 指定起始日期
  python fetch_data.py --provider akshare # 指定数据源
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
PROVIDER_DIR = PROJECT_ROOT / "tools" / "data_provider"

# 导入 Provider
import sys
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from data_provider import AkShareProvider, EastMoneyProvider, BaoStockProvider


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_data_path(symbol: str) -> Path:
    """数据缓存路径：data/{symbol}/daily.csv"""
    config = load_config()
    data_dir = PROJECT_ROOT / config["data_dir"]
    return data_dir / symbol / "daily.csv"


def load_local(path: Path) -> pd.DataFrame:
    """读取本地缓存数据"""
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame()


def get_provider(name: str = "akshare"):
    """获取 Provider 实例"""
    providers = {
        "akshare": AkShareProvider(),
        "eastmoney": EastMoneyProvider(),
        "baostock": BaoStockProvider(),
    }
    return providers.get(name, providers["akshare"])


def quality_gate(df_new: pd.DataFrame, local_df: pd.DataFrame, symbol: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """数据质量闸门：合并入库前的最后一道校验

    目的：防止脏数据（如前复权跳空 -65%）污染本地缓存和特征计算。

    检查项：
    1. 单日涨跌异常：|change_pct| > 20% → 剔除该行（ETF 正常单日波动 < 10%）
    2. 日期连续性：增量更新时，若新数据头部与本地尾部缺口 > 5 个自然日 → 报警

    Returns
    -------
    (df_new, warnings)
      df_new  — 清洗后的数据（若全部被剔除则为 None）
      warnings — 报警信息列表
    """
    warnings: List[str] = []
    if df_new is None or len(df_new) == 0:
        return df_new, warnings

    df = df_new.copy()

    # 1. 异常涨跌剔除（前复权跳空防护）
    if "change_pct" in df.columns:
        n_before = len(df)
        bad_mask = df["change_pct"].abs() > 20
        if bad_mask.any():
            bad_dates = df.loc[bad_mask, "date"].astype(str).tolist()
            warnings.append(
                f"{symbol} 剔除 {int(bad_mask.sum())} 行异常涨跌(|Δ|>20%): {bad_dates[:5]}{'...' if len(bad_dates) > 5 else ''}"
            )
            df = df[~bad_mask]
            if len(df) == 0:
                return None, warnings

    # 2. 日期连续性（仅增量更新时检查）
    if len(local_df) > 0:
        local_last = local_df["date"].max()
        new_first = df["date"].min()
        gap_days = (new_first - local_last).days
        if gap_days > 5:
            warnings.append(
                f"{symbol} 数据缺口 {gap_days} 天 (本地尾 {local_last.date()} → 新头 {new_first.date()})"
            )

    return df, warnings


def fetch_data(
    symbol: str,
    start_date: str,
    force: bool = False,
    provider_name: str = "akshare",
) -> pd.DataFrame:
    """拉取数据，自动增量更新"""
    # 东财系（AkShare/EastMoney）走代理必失败（mihomo 规则拒绝），强制直连
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("all_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)

    config = load_config()
    data_path = get_data_path(symbol)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    local_df = load_local(data_path)

    if force or len(local_df) == 0:
        start = start_date
        print(f"  [INFO] 全量拉取 {symbol} 从 {start_date}")
    else:
        last_date = local_df["date"].max().strftime("%Y-%m-%d")
        start = last_date
        print(f"  [INFO] 增量更新 {symbol} 从 {start}（已有 {len(local_df)} 条）")

    # 主 Provider
    provider = get_provider(provider_name)
    df_new = None
    retries = config.get("retry_count", 2)

    for attempt in range(retries + 1):
        try:
            df_new = provider.fetch_daily(symbol, start)
            if df_new is not None and len(df_new) > 0:
                df_new = provider.normalize(df_new, symbol)
                break
        except Exception as e:
            print(f"  [WARN] {provider.name} 失败: {e}")
            df_new = None
        if attempt < retries:
            time.sleep(config.get("retry_delay_sec", 3))

    # 备用 Provider：东财系 → baostock（独立源灾备）
    fallbacks = ["eastmoney", "baostock"] if provider_name == "akshare" else ["akshare", "baostock"]
    for fb_name in fallbacks:
        if df_new is not None and len(df_new) > 0:
            break
        print(f"  [FALLBACK] 切 {fb_name}...")
        fb = get_provider(fb_name)
        for attempt in range(retries + 1):
            try:
                df_new = fb.fetch_daily(symbol, start)
                if df_new is not None and len(df_new) > 0:
                    df_new = fb.normalize(df_new, symbol)
                    break
            except Exception as e:
                print(f"  [WARN] {fb.name} 失败: {e}")
                df_new = None
            if attempt < retries:
                time.sleep(config.get("retry_delay_sec", 3))

    # 全部失败 → 用本地缓存
    if df_new is None or len(df_new) == 0:
        if len(local_df) > 0:
            print(f"  ⚠️ 数据源均不可用，使用本地缓存（{len(local_df)} 条）")
            local_df.attrs["stale"] = True
            return local_df
        raise RuntimeError(f"无法获取 {symbol} 数据，且无本地缓存")

    # 数据质量闸门（合并前最后一道校验）
    df_new, q_warnings = quality_gate(df_new, local_df, symbol)
    for w in q_warnings:
        print(f"  ⚠️ [QUALITY] {w}")
    if df_new is None or len(df_new) == 0:
        if len(local_df) > 0:
            print(f"  ⚠️ 数据全部被质量闸门拦截，使用本地缓存（{len(local_df)} 条）")
            local_df.attrs["stale"] = True
            return local_df
        raise RuntimeError(f"{symbol} 数据全部被质量闸门拦截，且无本地缓存")

    # 合并增量
    if len(local_df) > 0 and not force:
        combined = pd.concat([local_df, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    else:
        combined = df_new

    combined.to_csv(data_path, index=False)
    print(f"  [OK] {symbol}: {len(combined)} 条 ({combined['date'].min().date()} ~ {combined['date'].max().date()})")
    return combined


def main():
    parser = argparse.ArgumentParser(description="拉取 588000 + 000688 日线数据")
    parser.add_argument("--force", action="store_true", help="强制全量重拉")
    parser.add_argument("--start", default="2023-01-01", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--provider", default="akshare", choices=["akshare", "eastmoney", "baostock"],
                        help="数据源")
    args = parser.parse_args()

    config = load_config()
    print("\n📥 数据拉取")
    df_sym = fetch_data(config["symbol"], args.start, args.force, args.provider)
    df_bench = fetch_data(config["benchmark"], args.start, args.force, args.provider)
    return df_sym, df_bench


if __name__ == "__main__":
    main()
