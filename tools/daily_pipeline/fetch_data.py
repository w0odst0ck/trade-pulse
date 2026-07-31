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
from typing import Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
PROVIDER_DIR = PROJECT_ROOT / "tools" / "data_provider"

# 导入 Provider
import sys
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from data_provider import AkShareProvider, EastMoneyProvider


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
    }
    return providers.get(name, providers["akshare"])


def fetch_data(
    symbol: str,
    start_date: str,
    force: bool = False,
    provider_name: str = "akshare",
) -> pd.DataFrame:
    """拉取数据，自动增量更新"""
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

    # 备用 Provider
    fallbacks = ["eastmoney"] if provider_name == "akshare" else ["akshare"]
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
    parser.add_argument("--provider", default="akshare", choices=["akshare", "eastmoney"],
                        help="数据源")
    args = parser.parse_args()

    config = load_config()
    print("\n📥 数据拉取")
    df_sym = fetch_data(config["symbol"], args.start, args.force, args.provider)
    df_bench = fetch_data(config["benchmark"], args.start, args.force, args.provider)
    return df_sym, df_bench


if __name__ == "__main__":
    main()
