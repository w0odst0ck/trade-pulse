#!/usr/bin/env python3
"""TencentProvider 单元测试（pytest，不联网）

覆盖：
  - 市场前缀映射（6/5/9 -> sh，0/1/3 -> sz，8/4 -> bj）
  - normalize 输出列完整性与顺序
  - 腾讯返回行解析（mock 6 字段样例）
  - 按年分段拼接去重逻辑（mock 响应）
外加：空响应、字段不足行、网络异常等边界。
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))

from data_provider.base import DataProviderError
from data_provider.tencent import TencentProvider
import fetch_data as fetch_data_mod


class FakeResp:
    """模拟 requests.Response（仅暴露 json()）"""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def make_resp(tencent_symbol: str, rows) -> FakeResp:
    """构造带 qfqday 节点的腾讯接口响应"""
    return FakeResp({"code": 0, "data": {tencent_symbol: {"qfqday": rows}}})


class TestName:
    def test_name_is_tencent(self):
        assert TencentProvider().name == "tencent"


class TestMarketPrefix:
    """市场前缀映射：6/5/9 -> sh，0/1/3 -> sz，8/4 -> bj"""

    @pytest.mark.parametrize("symbol,expected", [
        ("588000", "sh"),
        ("510300", "sh"),
        ("900901", "sh"),   # 9 开头 -> sh
        ("000688", "sz"),
        ("159915", "sz"),
        ("300750", "sz"),
        ("000001", "sz"),
        ("830799", "bj"),   # 8/4 开头 -> bj
        ("430047", "bj"),
        ("sh588000", "sh"),  # 已带前缀
        ("sz000688", "sz"),
    ])
    def test_mapping(self, symbol, expected):
        assert TencentProvider._market_prefix(symbol) == expected

    def test_unknown_prefix_raises(self):
        with pytest.raises(DataProviderError):
            TencentProvider._market_prefix("700000")


class TestNormalize:
    """normalize 输出列完整性与顺序"""

    def test_column_order_and_completeness(self):
        provider = TencentProvider()
        df = pd.DataFrame({
            "date": ["2023-01-04", "2023-01-03"],  # 故意乱序
            "open": [1.023, 1.001],
            "close": [1.016, 1.023],
            "high": [1.026, 1.024],
            "low": [1.008, 0.998],
            "volume": [19694224.0, 17785093.0],
        })
        out = provider.normalize(df, "588000")

        # 列完整性与顺序
        assert list(out.columns) == [
            "date", "open", "close", "high", "low", "volume", "amount",
            "amplitude", "change_pct", "change", "turnover", "symbol",
        ]
        # date 转 datetime
        assert pd.api.types.is_datetime64_any_dtype(out["date"])
        # 扩展列补齐为 0
        for col in ["amount", "amplitude", "change_pct", "change", "turnover"]:
            assert (out[col] == 0).all()
        # symbol 列
        assert (out["symbol"] == "588000").all()
        # 按 date 升序排序
        assert list(out["date"]) == sorted(out["date"])
        assert out["date"].iloc[0] == pd.Timestamp("2023-01-03")

    def test_missing_required_column_raises(self):
        provider = TencentProvider()
        with pytest.raises(KeyError):
            provider.normalize(pd.DataFrame({"date": ["2023-01-03"]}), "588000")


class TestFetchDaily:
    """fetch_daily：行解析 + 按年分段拼接去重（mock 响应，不联网）"""

    def test_single_year_parses_rows(self, monkeypatch):
        """真实接口行形式：每行是 list（['2023-01-03', '1.001', ...]）"""
        provider = TencentProvider()
        rows = [
            ["2023-01-03", "1.001", "1.023", "1.024", "0.998", "17785093.000"],
            ["2023-01-04", "1.023", "1.016", "1.026", "1.008", "19694224.000"],
        ]
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(url)
            return make_resp("sh588000", rows)

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        df = provider.fetch_daily("588000", "2023-01-01", "2023-12-31")

        # 单年单次请求，param 格式 {前缀}{symbol},day,{start},{end},{count},qfq
        assert len(calls) == 1
        assert "param=sh588000,day,2023-01-01,2023-12-31,1000,qfq" in calls[0]
        # 解析顺序：日期, 开盘, 收盘, 最高, 最低, 成交量
        assert len(df) == 2
        assert list(df.columns) == ["date", "open", "close", "high", "low", "volume"]
        assert df.iloc[0]["date"] == pd.Timestamp("2023-01-03")
        assert df.iloc[0]["open"] == 1.001
        assert df.iloc[0]["close"] == 1.023
        assert df.iloc[0]["high"] == 1.024
        assert df.iloc[0]["low"] == 0.998
        assert df.iloc[0]["volume"] == 17785093.0

    def test_csv_string_rows_compatible(self, monkeypatch):
        """兼容 CSV 字符串行形式（'2023-01-03,1.001,...'）"""
        provider = TencentProvider()
        rows = ["2023-01-03,1.001,1.023,1.024,0.998,17785093.0"]
        monkeypatch.setattr(
            "data_provider.tencent.requests.get",
            lambda url, headers=None, timeout=None: make_resp("sh588000", rows),
        )
        df = provider.fetch_daily("588000", "2023-01-01", "2023-12-31")
        assert df is not None and len(df) == 1
        assert df.iloc[0]["close"] == 1.023

    def test_multi_year_segment_and_dedup(self, monkeypatch):
        provider = TencentProvider()
        year_2023 = [
            ["2023-01-03", "1.001", "1.023", "1.024", "0.998", "17785093.0"],
            ["2023-12-28", "1.050", "1.060", "1.070", "1.040", "20000000.0"],
            ["2023-12-29", "1.060", "1.070", "1.080", "1.050", "21000000.0"],
        ]
        year_2024 = [
            ["2023-12-29", "1.060", "1.070", "1.080", "1.050", "21000000.0"],  # 与 2023 段重复
            ["2024-01-02", "1.070", "1.080", "1.090", "1.060", "22000000.0"],
        ]
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(url)
            if "2023-01-01" in url:
                return make_resp("sh588000", year_2023)
            return make_resp("sh588000", year_2024)

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        df = provider.fetch_daily("588000", "2023-01-01", "2024-06-30")

        # 按自然年分段各请求一次，首尾年边界正确
        assert len(calls) == 2
        assert "param=sh588000,day,2023-01-01,2023-12-31,1000,qfq" in calls[0]
        assert "param=sh588000,day,2024-01-01,2024-06-30,1000,qfq" in calls[1]
        # 拼接去重：2023 3 行 + 2024 新增 1 行 = 4 行，升序且无重复
        assert len(df) == 4
        assert df["date"].is_monotonic_increasing
        assert not df["date"].duplicated().any()
        assert df.iloc[-1]["date"] == pd.Timestamp("2024-01-02")

    def test_partial_start_year_boundary(self, monkeypatch):
        """start/end 落在年中时，分段边界精确到日"""
        provider = TencentProvider()
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(url)
            return make_resp("sz000688", [])

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        df = provider.fetch_daily("000688", "2024-06-01", "2024-06-30")

        assert len(calls) == 1
        assert "param=sz000688,day,2024-06-01,2024-06-30,1000,qfq" in calls[0]
        assert df is None  # 空数据返回 None

    def test_fallback_to_unadjusted_day(self, monkeypatch):
        """无 qfqday 节点时回退取 day（不复权）"""
        provider = TencentProvider()
        rows = ["2023-01-03,1.0,1.1,1.2,0.9,1000.0"]

        def fake_get(url, headers=None, timeout=None):
            return FakeResp({"code": 0, "data": {"sh588000": {"day": rows}}})

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        df = provider.fetch_daily("588000", "2023-01-01", "2023-12-31")
        assert df is not None and len(df) == 1
        assert df.iloc[0]["close"] == 1.1

    def test_short_rows_skipped(self, monkeypatch):
        """字段不足 6 的行跳过，不影响其余行解析"""
        provider = TencentProvider()
        rows = [
            "2023-01-03,1.0,1.1,1.2,0.9,1000.0",  # 完整 6 字段
            "2023-01-04,1.1,1.2",                 # 字段不足 -> 跳过
            "garbage-line-with-no-commas",        # 无逗号 -> 跳过
        ]
        monkeypatch.setattr(
            "data_provider.tencent.requests.get",
            lambda url, headers=None, timeout=None: make_resp("sh588000", rows),
        )
        df = provider.fetch_daily("588000", "2023-01-01", "2023-12-31")
        assert len(df) == 1

    def test_empty_data_returns_none(self, monkeypatch):
        provider = TencentProvider()
        monkeypatch.setattr(
            "data_provider.tencent.requests.get",
            lambda url, headers=None, timeout=None: make_resp("sh588000", []),
        )
        assert provider.fetch_daily("588000", "2023-01-01", "2023-12-31") is None

    def test_missing_symbol_node_returns_none(self, monkeypatch):
        provider = TencentProvider()
        monkeypatch.setattr(
            "data_provider.tencent.requests.get",
            lambda url, headers=None, timeout=None: FakeResp({"code": 0, "data": {}}),
        )
        assert provider.fetch_daily("588000", "2023-01-01", "2023-12-31") is None

    def test_network_error_raises(self, monkeypatch):
        provider = TencentProvider()

        def fake_get(url, headers=None, timeout=None):
            raise ConnectionError("boom")

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        with pytest.raises(DataProviderError):
            provider.fetch_daily("588000", "2023-01-01")

    def test_unknown_symbol_prefix_raises(self, monkeypatch):
        provider = TencentProvider()
        with pytest.raises(DataProviderError):
            provider.fetch_daily("700000", "2023-01-01", "2023-12-31")


class TestPrefixedSymbol:
    """symbol 已带市场前缀时 fetch_daily 直接使用，不重复拼前缀"""

    def test_prefixed_symbol_no_double_prefix(self, monkeypatch):
        provider = TencentProvider()
        rows = [["2023-01-03", "1.001", "1.023", "1.024", "0.998", "17785093.0"]]
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(url)
            return make_resp("sh588000", rows)

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        df = provider.fetch_daily("sh588000", "2023-01-01", "2023-12-31")

        # 单次请求，param 用 sh588000 而非 shsh588000
        assert len(calls) == 1
        assert "param=sh588000,day,2023-01-01" in calls[0]
        assert "shsh588000" not in calls[0]
        assert df is not None and len(df) == 1
        assert df.iloc[0]["close"] == 1.023

    def test_sh_index_prefixed_symbol_keeps_sh(self, monkeypatch):
        """sh000688（科创50指数）：带前缀直接使用，不被 0->sz 规则错判为深市个股"""
        provider = TencentProvider()
        rows = [["2023-01-03", "1600.0", "1601.0", "1602.0", "1599.0", "100000.0"]]
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(url)
            return make_resp("sh000688", rows)

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        df = provider.fetch_daily("sh000688", "2023-01-01", "2023-12-31")

        assert "param=sh000688,day," in calls[0]
        assert "sz000688" not in calls[0]
        assert df.iloc[0]["close"] == 1601.0

    def test_sz_prefixed_symbol_keeps_sz(self, monkeypatch):
        provider = TencentProvider()
        rows = [["2023-01-03", "28.0", "28.5", "28.8", "27.9", "1000000.0"]]
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(url)
            return make_resp("sz000688", rows)

        monkeypatch.setattr("data_provider.tencent.requests.get", fake_get)
        df = provider.fetch_daily("sz000688", "2023-01-01", "2023-12-31")

        assert "param=sz000688,day," in calls[0]
        assert "shsz000688" not in calls[0]
        assert df.iloc[0]["close"] == 28.5


class TestMarketsMapping:
    """fetch_data.py 的 config['markets'] 映射读取（resolve_tencent_symbol）"""

    @staticmethod
    def _patch_config(monkeypatch, tmp_path, payload: dict):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(fetch_data_mod, "CONFIG_PATH", str(cfg))

    def test_mapped_symbol_gets_prefix(self, monkeypatch, tmp_path):
        self._patch_config(monkeypatch, tmp_path, {"markets": {"588000": "sh", "000688": "sh"}})
        assert fetch_data_mod.resolve_tencent_symbol("588000") == "sh588000"
        assert fetch_data_mod.resolve_tencent_symbol("000688") == "sh000688"

    def test_unmapped_symbol_returned_as_is(self, monkeypatch, tmp_path):
        self._patch_config(monkeypatch, tmp_path, {"markets": {}})
        assert fetch_data_mod.resolve_tencent_symbol("510300") == "510300"

    def test_missing_markets_key_returned_as_is(self, monkeypatch, tmp_path):
        self._patch_config(monkeypatch, tmp_path, {"symbol": "588000"})
        assert fetch_data_mod.resolve_tencent_symbol("000688") == "000688"


def _fake_daily_df(symbol: str) -> pd.DataFrame:
    """构造两行 fake 日线（2023 年，避免盘中剔除逻辑干扰）"""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
        "open": [1.0, 1.1],
        "close": [1.1, 1.2],
        "high": [1.2, 1.3],
        "low": [0.9, 1.0],
        "volume": [1.0e8, 1.0e8],
    })
    df["symbol"] = symbol
    return df


class TestFetchDataTencentSymbol:
    """fetch_data() 调用 tencent provider 时传入带市场前缀 symbol（主源/fallback，不联网）"""

    def _patch_fetch_env(self, monkeypatch, tmp_path, get_provider_impl):
        monkeypatch.setattr(fetch_data_mod, "get_provider", get_provider_impl)
        monkeypatch.setattr(fetch_data_mod, "load_local", lambda path: pd.DataFrame())
        monkeypatch.setattr(
            fetch_data_mod, "quality_gate", lambda new, local, sym: (new, [])
        )
        monkeypatch.setattr(
            fetch_data_mod, "get_data_path", lambda sym: tmp_path / f"{sym}.csv"
        )

    def test_main_provider_tencent_gets_prefixed_symbol(self, monkeypatch, tmp_path):
        seen = []

        class FakeTencent:
            name = "tencent"

            def fetch_daily(self, symbol, start):
                seen.append(symbol)
                return _fake_daily_df(symbol)

            def normalize(self, df, symbol):
                return df

        self._patch_fetch_env(monkeypatch, tmp_path, lambda name: FakeTencent())
        out = fetch_data_mod.fetch_data(
            "588000", "2023-01-01", force=True, provider_name="tencent"
        )

        assert seen == ["sh588000"]
        assert out is not None and len(out) == 2

    def test_fallback_tencent_gets_prefixed_symbol(self, monkeypatch, tmp_path):
        """主源 akshare 失败降级到 tencent 时，同样传带前缀 symbol"""
        seen = []

        class FakeFailing:
            name = "akshare"

            def fetch_daily(self, symbol, start):
                raise RuntimeError("boom")

            def normalize(self, df, symbol):
                return df

        class FakeTencent:
            name = "tencent"

            def fetch_daily(self, symbol, start):
                seen.append(symbol)
                return _fake_daily_df(symbol)

            def normalize(self, df, symbol):
                return df

        def get_provider_impl(name):
            return FakeFailing() if name == "akshare" else FakeTencent()

        self._patch_fetch_env(monkeypatch, tmp_path, get_provider_impl)
        out = fetch_data_mod.fetch_data(
            "000688", "2023-01-01", force=True, provider_name="akshare"
        )

        assert seen == ["sh000688"]
        assert out is not None and len(out) == 2

    def test_non_tencent_provider_gets_raw_symbol(self, monkeypatch, tmp_path):
        """非 tencent 源（如 baostock）仍传原始 symbol"""
        seen = []

        class FakeBao:
            name = "baostock"

            def fetch_daily(self, symbol, start):
                seen.append(symbol)
                return _fake_daily_df(symbol)

            def normalize(self, df, symbol):
                return df

        self._patch_fetch_env(monkeypatch, tmp_path, lambda name: FakeBao())
        out = fetch_data_mod.fetch_data(
            "000688", "2023-01-01", force=True, provider_name="baostock"
        )

        assert seen == ["000688"]
        assert out is not None and len(out) == 2
