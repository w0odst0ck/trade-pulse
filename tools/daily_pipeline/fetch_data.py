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
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from data_quality import MAX_GAP_DAYS, quality_gate

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
PROVIDER_DIR = PROJECT_ROOT / "tools" / "data_provider"

# 导入 Provider
import sys
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from data_provider import AkShareProvider, EastMoneyProvider, BaoStockProvider, TencentProvider

# Provider 固定 fallback 顺序（取除主源外的其余源按序降级）
PROVIDER_ORDER = ["akshare", "tencent", "eastmoney", "baostock"]


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
        # symbol 列强制 str：避免 '000688' 被 pandas 推断为 int 688（前导零丢失）
        df = pd.read_csv(path, parse_dates=["date"], dtype={"symbol": str})
        return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame()


def validate_incremental(df: pd.DataFrame, require_date: str = None) -> dict:
    """校验增量结果。返回 {ok: bool, last_date: str, issues: list[str]}

    校验项：
      - df 非空
      - 最新日期 >= require_date（若传入，格式 YYYY-MM-DD）
      - 最新一行 close 非空且 > 0
      - 最新一行 volume 非空且 > 0，且与最近 20 日（不含当日）均量之比在
        [0.1, 10] 区间内（防半日/异常量数据；均量为去首尾 10% 极端值后的
        稳健均值，避免个别巨量日干扰；历史不足 20 日用可用行计算，
        无历史则跳过区间检查）
      - date 无重复、连续无缺失：全量查重复；相邻日期自然日间隔超
        MAX_GAP_DAYS 记不连续（判据与 data_quality.py 风格一致，仅检查
        最近 21 行增量核心窗口，避免 A 股长假间隔误报；存量历史由
        health_check 的 scan_quality 覆盖）
    """
    issues: List[str] = []
    if df is None or len(df) == 0:
        issues.append("数据为空（df 无任何行）")
        return {"ok": False, "last_date": "", "issues": issues}

    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    last_ts = dates.iloc[-1]
    last_date = last_ts.strftime("%Y-%m-%d")

    # 1. require_date 门槛：最新日期必须 >= 要求日期
    if require_date:
        try:
            require_ts = pd.to_datetime(require_date)
        except (ValueError, TypeError):
            issues.append(f"require_date 格式无效: {require_date}（应为 YYYY-MM-DD）")
        else:
            if last_ts < require_ts:
                issues.append(f"最新日期 {last_date} 早于要求日期 {require_date}")

    # 2. 最新一行 close 非空且 > 0
    if "close" not in df.columns:
        issues.append("缺少 close 列")
    else:
        last_close = df["close"].iloc[-1]
        if pd.isna(last_close) or float(last_close) <= 0:
            issues.append(f"最新一行 close 非法: {last_close}（应为 > 0）")

    # 3. 最新一行 volume 非空且 > 0，且与最近 20 日均量之比在 [0.1, 10]
    if "volume" not in df.columns:
        issues.append("缺少 volume 列")
    else:
        last_vol = df["volume"].iloc[-1]
        if pd.isna(last_vol) or float(last_vol) <= 0:
            issues.append(f"最新一行 volume 非法: {last_vol}（应为 > 0）")
        else:
            prior = pd.to_numeric(df["volume"].iloc[:-1], errors="coerce").tail(20).dropna()
            if len(prior) > 0:
                # 去首尾各 10% 极端值后求均量：个别巨量/异常日会拉高均值，
                # 导致正常缩量日被误判（如 2026-08 初 588000 缩量 vs 7 月末巨量）
                p = prior.sort_values()
                k = max(1, len(p) // 10)
                trimmed = p.iloc[k:-k] if len(p) > 2 * k else p
                avg = float(trimmed.mean()) if len(trimmed) > 0 else float(p.mean())
                if avg > 0:
                    ratio = float(last_vol) / avg
                    if not 0.1 <= ratio <= 10.0:
                        issues.append(
                            f"最新 volume {last_vol} 与 20 日均量 {avg:.1f} 比值 "
                            f"{ratio:.2f} 超出 [0.1, 10]"
                        )

    # 4. date 无重复、连续无缺失
    dups = dates[dates.duplicated(keep=False)]
    if len(dups) > 0:
        dup_days = sorted({d.strftime("%Y-%m-%d") for d in dups})
        issues.append(
            f"date 存在 {len(dup_days)} 个重复日: "
            f"{dup_days[:5]}{'...' if len(dup_days) > 5 else ''}"
        )

    recent = dates.tail(21)  # 增量核心窗口：当日 + 20 日均量所需历史
    prev = None
    for d in recent:
        if prev is not None:
            gap = (d - prev).days
            if gap > MAX_GAP_DAYS:
                issues.append(
                    f"日期不连续: {prev.strftime('%Y-%m-%d')} 与 {d.strftime('%Y-%m-%d')} "
                    f"间隔 {gap} 天 > {MAX_GAP_DAYS}"
                )
        prev = d

    return {"ok": len(issues) == 0, "last_date": last_date, "issues": issues}


def get_provider(name: str = "akshare"):
    """获取 Provider 实例"""
    providers = {
        "akshare": AkShareProvider(),
        "tencent": TencentProvider(),
        "eastmoney": EastMoneyProvider(),
        "baostock": BaoStockProvider(),
    }
    return providers.get(name, providers["akshare"])


def resolve_tencent_symbol(symbol: str) -> str:
    """按 config['markets'] 将 symbol 映射为腾讯带前缀形式（如 588000 -> sh588000）

    纯代码前缀规则无法区分「000xxx 上证指数」与「000xxx 深市股票」
    （如 000688 既可能是 sh000688 科创50指数，也可能是 sz000688 深市个股），
    必须显式指定市场。无映射时原样返回，由 TencentProvider._market_prefix 兜底。
    """
    market = (load_config().get("markets") or {}).get(symbol)
    return f"{market}{symbol}" if market else symbol


def fetch_data(
    symbol: str,
    start_date: str,
    force: bool = False,
    provider_name: str = None,
) -> pd.DataFrame:
    """拉取数据，自动增量更新

    provider_name 为 None 时使用 config.json 的 provider 字段（默认主源）。
    """
    if provider_name is None:
        provider_name = load_config().get("provider", "akshare")
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
    # 腾讯源需带市场前缀的 symbol（config['markets'] 映射），其余源用原始 symbol
    call_symbol = resolve_tencent_symbol(symbol) if provider_name == "tencent" else symbol

    for attempt in range(retries + 1):
        try:
            df_new = provider.fetch_daily(call_symbol, start)
            if df_new is not None and len(df_new) > 0:
                df_new = provider.normalize(df_new, symbol)
                break
        except Exception as e:
            print(f"  [WARN] {provider.name} 失败: {e}")
            df_new = None
        if attempt < retries:
            time.sleep(config.get("retry_delay_sec", 3))

    # 备用 Provider：按固定顺序取除主源外的其余源降级
    fallbacks = [p for p in PROVIDER_ORDER if p != provider_name]
    for fb_name in fallbacks:
        if df_new is not None and len(df_new) > 0:
            break
        print(f"  [FALLBACK] 切 {fb_name}...")
        fb = get_provider(fb_name)
        # 主源/备用统一在调用点处理：tencent 源用带市场前缀的 symbol
        fb_symbol = resolve_tencent_symbol(symbol) if fb_name == "tencent" else symbol
        for attempt in range(retries + 1):
            try:
                df_new = fb.fetch_daily(fb_symbol, start)
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
            print(f"  [FETCH_FAIL] {symbol} 数据源均不可用（主源 {provider_name} + 备用均失败），使用本地缓存")
            local_df.attrs["stale"] = True
            return local_df
        print(f"  [FETCH_FAIL] {symbol} 无法获取数据，且无本地缓存")
        raise RuntimeError(f"无法获取 {symbol} 数据，且无本地缓存")

    # 数据质量闸门（合并前最后一道校验）
    df_new, q_warnings = quality_gate(df_new, local_df, symbol)
    for w in q_warnings:
        print(f"  ⚠️ [QUALITY] {w}")
    if df_new is None or len(df_new) == 0:
        if len(local_df) > 0:
            print(f"  ⚠️ 数据全部被质量闸门拦截，使用本地缓存（{len(local_df)} 条）")
            print(f"  [FETCH_FAIL] {symbol} 数据全部被质量闸门拦截，使用本地缓存")
            local_df.attrs["stale"] = True
            return local_df
        print(f"  [FETCH_FAIL] {symbol} 数据全部被质量闸门拦截，且无本地缓存")
        raise RuntimeError(f"{symbol} 数据全部被质量闸门拦截，且无本地缓存")

    # 合并增量
    if len(local_df) > 0 and not force:
        combined = pd.concat([local_df, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    else:
        combined = df_new

    # 盘中防护：收盘前剔除当日未完成 bar（避免半日数据污染信号/特征）
    # 15:00 收盘后允许保留当日完整日线；14:25 信号任务等盘中跑则剔除，
    # 收盘后的 15:30 增量任务与次日任务自然补回。
    if datetime.now().strftime("%H:%M") < "15:00":
        today_str = datetime.now().strftime("%Y-%m-%d")
        n_before = len(combined)
        combined = combined[combined["date"] != today_str].reset_index(drop=True)
        if len(combined) < n_before:
            print(f"  [INFO] 剔除当日盘中未收盘数据（{today_str}）")

    combined.to_csv(data_path, index=False)
    print(f"  [OK] {symbol}: {len(combined)} 条 ({combined['date'].min().date()} ~ {combined['date'].max().date()})")
    return combined


def main():
    parser = argparse.ArgumentParser(description="拉取 588000 + 000688 日线数据")
    parser.add_argument("--force", action="store_true", help="强制全量重拉")
    parser.add_argument("--start", default="2023-01-01", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--provider", default=None,
                        choices=["akshare", "tencent", "eastmoney", "baostock"],
                        help="数据源（默认取 config.json 的 provider 字段）")
    parser.add_argument("--require-date", default=None,
                        help="增量完成后校验最新日期 >= 该日期 (YYYY-MM-DD)，不满足则退出码 1")
    args = parser.parse_args()

    config = load_config()
    print("\n📥 数据拉取")
    df_sym = fetch_data(config["symbol"], args.start, args.force, args.provider)
    df_bench = fetch_data(config["benchmark"], args.start, args.force, args.provider)

    if args.require_date:
        for name, df in ((config["symbol"], df_sym), (config["benchmark"], df_bench)):
            result = validate_incremental(df, require_date=args.require_date)
            if not result["ok"]:
                print(f"  [FAIL] {name} 数据完整性校验失败（require-date={args.require_date}）")
                for issue in result["issues"]:
                    print(f"    - {issue}")
                sys.exit(1)
            print(f"  [OK] {name} 数据完整，最新日期 {result['last_date']}")

    return df_sym, df_bench


if __name__ == "__main__":
    main()
