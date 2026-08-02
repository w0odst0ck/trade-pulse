#!/usr/bin/env python3
"""data_quality.py — 数据质量判据（共享模块）

fetch_data.py（拉取时拦截）与 health_check.py（存量扫描）共用同一套判据，
避免两处逻辑漂移。

判据：
  1. 跳空检测：单日涨跌 |change_pct| > 20% → 视为异常（ETF 正常单日波动 < 10%，
     前复权除权跳空可到 -65%）
  2. NaN 检测：close/open/high/low/volume 关键列存在 NaN
  3. 连续性检测：增量更新时新数据头部与本地尾部缺口 > 5 个自然日 → 报警
"""

from typing import List, Optional, Tuple

import pandas as pd

JUMP_THRESHOLD_PCT = 20.0
MAX_GAP_DAYS = 5


def find_jumps(df: pd.DataFrame, threshold_pct: float = JUMP_THRESHOLD_PCT) -> pd.DataFrame:
    """返回单日涨跌 |change_pct| > threshold_pct 的行（空 DataFrame 表示无异常）"""
    if df is None or len(df) == 0 or "change_pct" not in df.columns:
        return pd.DataFrame()
    return df[df["change_pct"].abs() > threshold_pct]


def find_nan_rows(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """返回关键列含 NaN 的行（空 DataFrame 表示无异常）"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    cols = columns or [c for c in ["close", "open", "high", "low", "volume"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[df[cols].isna().any(axis=1)]


def check_gap_days(local_df: pd.DataFrame, new_df: pd.DataFrame) -> int:
    """增量更新时，新数据头部与本地尾部的时间缺口（自然日）。无缺口返回 0"""
    if local_df is None or new_df is None or len(local_df) == 0 or len(new_df) == 0:
        return 0
    local_last = pd.to_datetime(local_df["date"]).max()
    new_first = pd.to_datetime(new_df["date"]).min()
    return (new_first - local_last).days


def quality_gate(
    df_new: pd.DataFrame,
    local_df: Optional[pd.DataFrame] = None,
    symbol: str = "",
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """合并入库前的最后一道校验（fetch_data.py 使用）

    Returns
    -------
    (df_new, warnings)
      df_new    — 清洗后的数据（若全部被剔除则为 None）
      warnings  — 报警信息列表
    """
    warnings: List[str] = []
    if df_new is None or len(df_new) == 0:
        return df_new, warnings

    df = df_new.copy()

    # 1. 跳空剔除
    jumps = find_jumps(df)
    if len(jumps) > 0:
        bad_dates = jumps["date"].astype(str).tolist()
        warnings.append(
            f"{symbol} 剔除 {len(jumps)} 行异常涨跌(|Δ|>{JUMP_THRESHOLD_PCT:.0f}%): "
            f"{bad_dates[:5]}{'...' if len(bad_dates) > 5 else ''}"
        )
        df = df[~df.index.isin(jumps.index)]
        if len(df) == 0:
            return None, warnings

    # 2. 连续性检测（仅增量更新时检查）
    if local_df is not None and len(local_df) > 0:
        gap = check_gap_days(local_df, df)
        if gap > MAX_GAP_DAYS:
            local_last = pd.to_datetime(local_df["date"]).max().date()
            new_first = pd.to_datetime(df["date"]).min().date()
            warnings.append(
                f"{symbol} 数据缺口 {gap} 天 (本地尾 {local_last} → 新头 {new_first})"
            )

    return df, warnings


def scan_quality(df: pd.DataFrame, symbol: str = "") -> List[str]:
    """存量数据质量扫描（health_check.py 使用）

    返回异常描述列表；空列表表示数据干净。
    """
    problems: List[str] = []
    if df is None or len(df) == 0:
        problems.append(f"{symbol} daily.csv 为空")
        return problems

    # 跳空残留
    jumps = find_jumps(df)
    if len(jumps) > 0:
        dates = jumps["date"].astype(str).tolist()[:5]
        problems.append(
            f"{symbol} 存量含 {len(jumps)} 行跳空(|Δ|>{JUMP_THRESHOLD_PCT:.0f}%): {dates}"
        )

    # NaN 残留
    nan_rows = find_nan_rows(df)
    if len(nan_rows) > 0:
        dates = nan_rows["date"].astype(str).tolist()[:5]
        problems.append(f"{symbol} 存量含 {len(nan_rows)} 行 NaN: {dates}")

    return problems
