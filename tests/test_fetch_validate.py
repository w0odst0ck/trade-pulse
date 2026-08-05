#!/usr/bin/env python3
"""fetch_data.py validate_incremental 单元测试（pytest）

覆盖验收要求 4 个场景：
  - 正常 df：校验通过
  - 缺当日：require_date 不满足 → 校验失败
  - volume 异常：0 值 / 超 [0.1, 10] 区间 → 校验失败
  - date 重复 → 校验失败
外加：空 df、close 非法、require_date 格式无效等边界。
"""

import sys
from pathlib import Path

import pandas as pd

# fetch_data.py 内部依赖同目录的 data_quality 与 tools/data_provider
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from fetch_data import validate_incremental


def make_df(n=25, end="2026-08-05", volume=1.0e8, close=1.5):
    """构造 n 行工作日日线 DataFrame，最新日期为 end"""
    dates = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.05,
        "close": close,
        "high": close + 0.05,
        "low": close - 0.1,
        "volume": float(volume),
        "amount": 0.0,
        "amplitude": 1.0,
        "change_pct": 0.0,
        "change": 0.0,
        "turnover": 0.0,
        "symbol": "588000",
    })


class TestValidateIncremental:
    """验收场景 1：正常 df 校验通过"""

    def test_normal_df_passes(self):
        df = make_df()
        result = validate_incremental(df)
        assert result["ok"] is True
        assert result["last_date"] == "2026-08-05"
        assert result["issues"] == []

    def test_normal_df_with_satisfied_require_date(self):
        df = make_df()
        result = validate_incremental(df, require_date="2026-08-05")
        assert result["ok"] is True

    def test_unsorted_df_is_sorted_first(self):
        df = make_df()
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        result = validate_incremental(df)
        assert result["ok"] is True
        assert result["last_date"] == "2026-08-05"


class TestRequireDate:
    """验收场景 2：缺当日（require_date 不满足）→ 校验失败"""

    def test_missing_latest_day_fails(self):
        df = make_df()  # 最新 2026-08-05
        result = validate_incremental(df, require_date="2026-08-06")
        assert result["ok"] is False
        assert any("早于要求日期 2026-08-06" in issue for issue in result["issues"])
        assert result["last_date"] == "2026-08-05"

    def test_far_future_require_date_fails(self):
        df = make_df()
        result = validate_incremental(df, require_date="2099-01-01")
        assert result["ok"] is False
        assert any("2099-01-01" in issue for issue in result["issues"])

    def test_invalid_require_date_reports_issue(self):
        df = make_df()
        result = validate_incremental(df, require_date="not-a-date")
        assert result["ok"] is False
        assert any("格式无效" in issue for issue in result["issues"])


class TestVolume:
    """验收场景 3：volume 异常（0 值、超区间）→ 校验失败"""

    def test_zero_volume_fails(self):
        df = make_df(volume=1.0e8)
        df.loc[df.index[-1], "volume"] = 0.0
        result = validate_incremental(df)
        assert result["ok"] is False
        assert any("volume 非法" in issue for issue in result["issues"])

    def test_volume_ratio_above_range_fails(self):
        df = make_df(volume=1.0e8)
        df.loc[df.index[-1], "volume"] = 20.0 * 1.0e8  # 20 倍均量 > 10
        result = validate_incremental(df)
        assert result["ok"] is False
        assert any("超出 [0.1, 10]" in issue for issue in result["issues"])

    def test_volume_ratio_below_range_fails(self):
        df = make_df(volume=1.0e8)
        df.loc[df.index[-1], "volume"] = 0.01 * 1.0e8  # 0.01 倍均量 < 0.1
        result = validate_incremental(df)
        assert result["ok"] is False
        assert any("超出 [0.1, 10]" in issue for issue in result["issues"])

    def test_volume_ratio_at_boundary_passes(self):
        df = make_df(volume=1.0e8)
        df.loc[df.index[-1], "volume"] = 10.0 * 1.0e8  # 恰好 10 倍，区间含边界
        result = validate_incremental(df)
        assert result["ok"] is True


class TestDates:
    """验收场景 4：date 重复 → 校验失败"""

    def test_duplicate_date_fails(self):
        df = make_df()
        df.loc[df.index[-1], "date"] = df.loc[df.index[-2], "date"]  # 最后一行日期重复
        result = validate_incremental(df)
        assert result["ok"] is False
        assert any("重复" in issue for issue in result["issues"])

    def test_gap_in_recent_window_fails(self):
        df = make_df()
        # 删除最近 3 行（07-31 / 08-03 / 08-04），制造 >5 天自然日缺口
        df = df.drop(df.index[-4:-1]).reset_index(drop=True)
        result = validate_incremental(df)
        assert result["ok"] is False
        assert any("日期不连续" in issue for issue in result["issues"])


class TestEdgeCases:
    """边界场景"""

    def test_empty_df_fails(self):
        result = validate_incremental(pd.DataFrame())
        assert result["ok"] is False
        assert any("空" in issue for issue in result["issues"])

    def test_none_df_fails(self):
        result = validate_incremental(None)
        assert result["ok"] is False

    def test_zero_close_fails(self):
        df = make_df()
        df.loc[df.index[-1], "close"] = 0.0
        result = validate_incremental(df)
        assert result["ok"] is False
        assert any("close 非法" in issue for issue in result["issues"])
