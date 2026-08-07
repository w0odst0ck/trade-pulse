#!/usr/bin/env python3
"""fetch_data.py 多源并行调度测试（pytest，mock provider 不联网）

覆盖：
  - 快源命中即停（不碰慢源）
  - 快源全未命中 → 阶段 2 拉慢源
  - 0 命中 → 本地缓存 + FETCH_FAIL
  - ≥2 命中一致 → 用最高优先级源
  - ≥2 命中不一致 → 多数一致组 + 报警标记
  - 差异 >5% → 不覆盖本地
  - 节流：30min 内 max_date 未变 → 跳过
  - 冷却：连续 3 次失败 → cooldown_until 生效
  - 目标日期：now<15:00 → 前一交易日（注入 clock）
  - 修订检测：本地与新数据同日 close 差异 → 报警不覆盖
  - 15:30 场景：腾讯未命中 + 新浪命中 → 成功入库
外加 SinaProvider normalize（volume ÷100、列映射、日期过滤）。
"""

import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_data as fd
from data_provider.sina import SinaProvider

TARGET = "2026-08-06"


def make_daily(end="2026-08-06", n=6, close=1.7, step=0.01, volume=1.0e7, symbol="588000"):
    """构造 n 行标准 12 列日线 DataFrame（close 阶梯递增），最新日期为 end"""
    dates = pd.bdate_range(end=end, periods=n)
    closes = [close + i * step for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "close": closes,
        "high": closes,
        "low": closes,
        "volume": float(volume),
        "amount": 0.0,
        "amplitude": 0.0,
        "change_pct": 0.0,
        "change": 0.0,
        "turnover": 0.0,
        "symbol": symbol,
    })


class FakeProvider:
    """mock 数据源：fetch_daily 记录调用，返回预置 df / None / 抛异常"""

    def __init__(self, name, payload, calls):
        self.name = name
        self._payload = payload  # pd.DataFrame | None | Exception
        self.calls = calls

    def fetch_daily(self, symbol, start):
        self.calls.append((symbol, start))
        if isinstance(self._payload, Exception):
            raise self._payload
        return None if self._payload is None else self._payload.copy()

    def normalize(self, df, symbol):
        df = df.copy()
        df["symbol"] = symbol
        return df


def patch_multi_env(monkeypatch, tmp_path, providers, local_df=None, overrides=None):
    """构造多源环境：tmp config + 隔离 source_health + mock provider

    providers: {src_name: payload(df/None/Exception)}
    返回 (calls, wrapped)：calls[src] 记录 fetch_daily 调用列表；
    wrapped[src] 为对应 FakeProvider。
    """
    cfg = {
        "symbol": "588000", "benchmark": "000688", "provider": "tencent",
        "markets": {"588000": "sh", "000688": "sh"},
        "data_dir": "data",
        "multi_source": True,
        "sources": ["tencent", "sina", "baostock", "akshare", "eastmoney"],
        "fast_sources": ["tencent", "sina"],
        "source_cooldown": {"max_failures": 3, "cooldown_hours": 24, "skip_window_min": 30},
        "retry_count": 2, "retry_delay_sec": 0,
        **(overrides or {}),
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(fd, "CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(fd, "SOURCE_HEALTH_PATH", tmp_path / "source_health.json")
    monkeypatch.setattr(fd, "get_data_path", lambda sym: tmp_path / f"{sym}.csv")
    monkeypatch.setattr(fd, "load_local", lambda path: local_df if local_df is not None else pd.DataFrame())
    monkeypatch.setattr(fd, "compute_target_date", lambda now: TARGET)

    calls = {name: [] for name in providers}
    wrapped = {name: FakeProvider(name, payload, calls[name]) for name, payload in providers.items()}
    monkeypatch.setattr(fd, "get_provider", lambda name: wrapped[name])
    return calls, wrapped


def write_health(tmp_path, payload):
    """预写 source_health.json（payload: {symbol: {src: {...}}}）"""
    path = tmp_path / "source_health.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestTwoPhaseScheduling:
    """两阶段调度：快源命中即停 / 快源未命中才碰慢源"""

    def test_fast_hit_skips_slow(self, monkeypatch, tmp_path, capsys):
        """阶段 1 任一快源命中 → 不碰慢源（东财/baostock 零调用）"""
        tencent = make_daily(end=TARGET)
        slow = RuntimeError("slow source should not be touched")
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": tencent, "sina": None, "baostock": slow,
            "akshare": slow, "eastmoney": slow,
        })
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        captured = capsys.readouterr().out
        assert len(out) == len(tencent)
        assert calls["tencent"]  # 被调用且命中
        assert calls["sina"]     # 快源都尝试
        assert calls["baostock"] == [] and calls["akshare"] == [] and calls["eastmoney"] == []
        # 机器可读 [SRC] 日志
        assert "tencent HIT" in captured
        assert "sina MISS(empty)" in captured
        assert "baostock NOTRUN(fast-hit)" in captured

    def test_fast_miss_triggers_slow(self, monkeypatch, tmp_path):
        """快源全未命中（缺 target bar）→ 阶段 2 拉慢源，慢源命中入库"""
        no_bar = make_daily(end="2026-08-05")  # 缺 08-06
        akshare = make_daily(end=TARGET)
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": no_bar, "sina": no_bar,
            "baostock": RuntimeError("boom"), "akshare": akshare,
            "eastmoney": RuntimeError("boom"),
        })
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        assert len(out) == len(akshare)
        assert calls["akshare"]  # 阶段 2 确实拉慢源
        assert calls["tencent"] and calls["sina"]

    def test_simulate_1530_scenario(self, monkeypatch, tmp_path, capsys):
        """验收场景 4：15:30 后腾讯未命中（盘后未定型）+ 新浪命中 → 成功入库"""
        tencent = make_daily(end="2026-08-05")  # 腾讯 15:30/16:30 无当天 bar
        sina = make_daily(end=TARGET)            # 新浪当天可得
        slow = RuntimeError("should not run")
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": tencent, "sina": sina,
            "baostock": slow, "akshare": slow, "eastmoney": slow,
        })
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        captured = capsys.readouterr().out
        assert "tencent MISS(no-2026-08-06-bar)" in captured
        assert "sina HIT" in captured
        assert calls["baostock"] == [] and calls["akshare"] == []  # 快源命中即停
        assert out["date"].max().strftime("%Y-%m-%d") == TARGET


class TestZeroHit:
    def test_zero_hit_uses_local_cache(self, monkeypatch, tmp_path, capsys):
        """0 命中 → [FETCH_FAIL] 用本地缓存（stale 标记，不写盘）"""
        local = make_daily(end="2026-08-05")
        no_bar = make_daily(end="2026-08-05")
        patch_multi_env(monkeypatch, tmp_path, {
            "tencent": no_bar, "sina": no_bar,
            "baostock": RuntimeError("boom"), "akshare": RuntimeError("boom"),
            "eastmoney": RuntimeError("boom"),
        }, local_df=local)
        out = fd.fetch_data("588000", "2023-01-01")
        captured = capsys.readouterr().out
        assert out is local
        assert out.attrs.get("stale") is True
        assert "[FETCH_FAIL]" in captured
        assert not (tmp_path / "588000.csv").exists()  # 不覆盖本地

    def test_zero_hit_but_local_covers_target(self, monkeypatch, tmp_path, capsys):
        """0 命中但本地已覆盖目标日（快源节流/重复跑）→ [INFO] 数据已到位，非失败"""
        local = make_daily(end=TARGET)
        no_bar = make_daily(end="2026-08-05")
        patch_multi_env(monkeypatch, tmp_path, {
            "tencent": no_bar, "sina": no_bar,
            "baostock": RuntimeError("boom"), "akshare": RuntimeError("boom"),
            "eastmoney": RuntimeError("boom"),
        }, local_df=local)
        out = fd.fetch_data("588000", "2023-01-01")
        captured = capsys.readouterr().out
        assert "数据已到位" in captured
        assert "[FETCH_FAIL]" not in captured
        assert out.attrs.get("stale") is None  # 非 stale（数据未缺失）
        assert out["date"].max().strftime("%Y-%m-%d") == TARGET


class TestAdjudication:
    """多源命中裁决"""

    def test_consistent_hits_use_highest_priority(self, monkeypatch, tmp_path):
        """≥2 命中且一致 → 用最高优先级源（tencent）入库"""
        tencent = make_daily(end=TARGET, close=1.7)
        sina = tencent.copy()
        sina["close"] = sina["close"] * 1.0003   # 相对差 0.03% < 0.1% → 一致
        baostock = tencent.copy()
        # 三源都列为快源（不走两阶段）以测多源一致裁决
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": tencent, "sina": sina, "baostock": baostock,
            "akshare": None, "eastmoney": None,
        }, overrides={"fast_sources": ["tencent", "sina", "baostock"]})
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        # 入库值 = tencent（最高优先级），而非 sina 的 1.0003 倍
        assert out["close"].iloc[-1] == pytest.approx(tencent["close"].iloc[-1])
        assert calls["tencent"] and calls["sina"] and calls["baostock"]
        assert calls["akshare"] == [] and calls["eastmoney"] == []  # 命中后阶段 2 不再拉

    def test_inconsistent_tie_uses_highest_priority_with_alert(self, monkeypatch, tmp_path, capsys):
        """命中不一致（2 源平局）→ 用最高优先级源 + [ALERT] 报警"""
        tencent = make_daily(end=TARGET, close=1.7)
        sina = tencent.copy()
        sina["close"] = sina["close"] * 1.02   # 相对差 2%：不一致但 <5%
        patch_multi_env(monkeypatch, tmp_path, {
            "tencent": tencent, "sina": sina, "baostock": None,
            "akshare": None, "eastmoney": None,
        })
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        captured = capsys.readouterr().out
        assert out["close"].iloc[-1] == pytest.approx(tencent["close"].iloc[-1])  # 平局用 tencent
        assert "[ALERT]" in captured
        assert "平局" in captured and "采用最高优先级源 tencent" in captured

    def test_majority_group_wins_over_highest_priority(self, monkeypatch, tmp_path):
        """不一致时取多数一致组：tencent 少数派，sina+baostock 多数 → 用 sina 入库"""
        tencent = make_daily(end=TARGET, close=1.7)
        sina = tencent.copy()
        sina["close"] = sina["close"] * 1.02       # sina 与 tencent 不一致（2%，<5%）
        baostock = sina.copy()                       # baostock 与 sina 一致
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": tencent, "sina": sina, "baostock": baostock,
            "akshare": None, "eastmoney": None,
        }, overrides={"fast_sources": ["tencent", "sina", "baostock"]})
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        # 多数一致组 = {sina, baostock}，winner = 组内最高优先级 sina
        assert out["close"].iloc[-1] == pytest.approx(sina["close"].iloc[-1])
        assert calls["tencent"] and calls["sina"] and calls["baostock"]

    def test_conflict_over_5pct_no_overwrite(self, monkeypatch, tmp_path, capsys):
        """源间 close 差异 >5% → 不覆盖本地数据 + [FETCH_FAIL] 报警"""
        local = make_daily(end="2026-08-05")
        tencent = make_daily(end=TARGET, close=1.7)
        sina = tencent.copy()
        sina["close"] = sina["close"] * 1.10   # 相对差 10% > 5%
        patch_multi_env(monkeypatch, tmp_path, {
            "tencent": tencent, "sina": sina, "baostock": None,
            "akshare": None, "eastmoney": None,
        }, local_df=local)
        out = fd.fetch_data("588000", "2023-01-01")
        captured = capsys.readouterr().out
        assert out is local
        assert out.attrs.get("stale") is True
        assert "[FETCH_FAIL]" in captured
        assert "不覆盖本地数据" in captured
        assert not (tmp_path / "588000.csv").exists()


class TestThrottleAndCooldown:
    """节流与冷却"""

    def test_throttle_skips_recent_unseen_source(self, monkeypatch, tmp_path, capsys):
        """30min 内刚拉过且 last_max_date 未变（>=target）→ SKIP(throttle)，不再请求"""
        now = time.time()
        write_health(tmp_path, {"588000": {"tencent": {
            "last_fetch_ts": now - 60, "last_max_date": TARGET,
            "consecutive_failures": 0, "cooldown_until": 0,
        }}})
        sina = make_daily(end=TARGET)
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": make_daily(end=TARGET), "sina": sina,
            "baostock": None, "akshare": None, "eastmoney": None,
        })
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        captured = capsys.readouterr().out
        assert calls["tencent"] == []  # 节流跳过，未发起请求
        assert "tencent SKIP(throttle)" in captured
        assert len(out) == len(sina)

    def test_cooldown_skips_source(self, monkeypatch, tmp_path, capsys):
        """cooldown_until 未到期 → SKIP(cooldown)，不再请求"""
        write_health(tmp_path, {"588000": {"tencent": {
            "last_fetch_ts": 0, "last_max_date": None,
            "consecutive_failures": 3, "cooldown_until": time.time() + 24 * 3600,
        }}})
        sina = make_daily(end=TARGET)
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": make_daily(end=TARGET), "sina": sina,
            "baostock": None, "akshare": None, "eastmoney": None,
        })
        out = fd.fetch_data("588000", "2023-01-01", force=True)
        captured = capsys.readouterr().out
        assert calls["tencent"] == []
        assert "tencent SKIP(cooldown)" in captured
        assert len(out) == len(sina)

    def test_health_after_3_failures_enters_cooldown(self):
        """连续 3 次请求异常 → cooldown_until = now + cooldown_hours；成功清零"""
        cfg = {"max_failures": 3, "cooldown_hours": 24}
        h = {}
        fd._update_health(h, "588000", "tencent", "FAIL", None, cfg, 1000.0)
        fd._update_health(h, "588000", "tencent", "FAIL", None, cfg, 1001.0)
        assert h["588000"]["tencent"]["consecutive_failures"] == 2
        assert h["588000"]["tencent"]["cooldown_until"] == 0
        fd._update_health(h, "588000", "tencent", "FAIL", None, cfg, 1002.0)
        assert h["588000"]["tencent"]["consecutive_failures"] == 3
        assert h["588000"]["tencent"]["cooldown_until"] == 1002.0 + 24 * 3600
        # 请求成功（即使未命中）→ 清零，不计冷却
        fd._update_health(h, "588000", "tencent", "MISS", "2026-08-05", cfg, 1003.0)
        assert h["588000"]["tencent"]["consecutive_failures"] == 0
        assert h["588000"]["tencent"]["cooldown_until"] == 0
        assert h["588000"]["tencent"]["last_max_date"] == "2026-08-05"


class TestTargetDate:
    """目标日期语义：盘中 → 前一交易日；盘后 → 最近交易日"""

    def test_before_1500_uses_prev_trading_day(self):
        # 2026-08-07 是周五（交易日），盘中 → 前一交易日 08-06
        assert fd.compute_target_date(datetime(2026, 8, 7, 10, 0)) == "2026-08-06"
        assert fd.compute_target_date(datetime(2026, 8, 7, 14, 59)) == "2026-08-06"

    def test_after_1500_uses_latest_trading_day(self):
        assert fd.compute_target_date(datetime(2026, 8, 7, 15, 0)) == "2026-08-07"
        assert fd.compute_target_date(datetime(2026, 8, 7, 16, 30)) == "2026-08-07"

    def test_weekend_rolls_back(self):
        # 周六 15:30 → 最近交易日 = 周五 08-07
        assert fd.compute_target_date(datetime(2026, 8, 8, 15, 30)) == "2026-08-07"
        # 周日 → 回退周五
        assert fd.compute_target_date(datetime(2026, 8, 9, 10, 0)) == "2026-08-07"

    def test_holiday_rolls_back(self):
        # 2026-10-01 国庆假日，15:30 后 → 最近交易日 09-30
        assert fd.compute_target_date(datetime(2026, 10, 1, 15, 30)) == "2026-09-30"


class TestMergeIncremental:
    """合并增量：修订检测（同日不覆盖 + [ALERT]）"""

    def test_revision_detected_keeps_local(self, monkeypatch, tmp_path, capsys):
        local = make_daily(end="2026-08-06", n=6, close=1.7)
        new = make_daily(end="2026-08-07", n=7, close=1.7)
        new.loc[new["date"] == "2026-08-06", "close"] = 1.9  # 同日 close 差异 >0.1%
        combined = fd.merge_incremental(local, new, "588000")
        captured = capsys.readouterr().out
        assert "[ALERT]" in captured
        assert "修订差异" in captured
        assert combined["date"].nunique() == 7            # 只追加 08-07
        local_close = local.loc[local["date"] == "2026-08-06", "close"].iloc[0]
        assert combined.loc[combined["date"] == "2026-08-06", "close"].iloc[0] == local_close  # 保留本地

    def test_same_close_no_alert(self, capsys):
        local = make_daily(end="2026-08-06", n=6, close=1.7)
        new = make_daily(end="2026-08-07", n=7, close=1.7)
        # 差异 0.01% < 0.1%：以本地 08-06 实际 close 为基准
        local_close = local.loc[local["date"] == "2026-08-06", "close"].iloc[0]
        new.loc[new["date"] == "2026-08-06", "close"] = local_close * 1.0001
        combined = fd.merge_incremental(local, new, "588000")
        captured = capsys.readouterr().out
        assert "[ALERT]" not in captured
        assert combined["date"].nunique() == 7


class TestLegacySingleSource:
    """单源模式（--provider）：sina 也需带市场前缀（000688 → sh000688 科创50指数）"""

    def test_legacy_provider_sina_gets_prefixed_symbol(self, monkeypatch, tmp_path):
        """避免拉到 sz000688 深市个股：legacy 主源 sina 同样 resolve_tencent_symbol"""
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "sina": make_daily(end=TARGET), "tencent": None,
            "baostock": None, "akshare": None, "eastmoney": None,
        })
        fd.fetch_data("000688", "2023-01-01", force=True, provider_name="sina")
        assert calls["sina"] == [("sh000688", "2023-01-01")]

    def test_legacy_provider_tencent_gets_prefixed_symbol(self, monkeypatch, tmp_path):
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "tencent": make_daily(end=TARGET), "sina": None,
            "baostock": None, "akshare": None, "eastmoney": None,
        })
        fd.fetch_data("000688", "2023-01-01", force=True, provider_name="tencent")
        assert calls["tencent"] == [("sh000688", "2023-01-01")]

    def test_legacy_provider_baostock_uses_raw_symbol(self, monkeypatch, tmp_path):
        """非 tencent/sina 源仍传原始 symbol"""
        calls, _ = patch_multi_env(monkeypatch, tmp_path, {
            "baostock": make_daily(end=TARGET), "tencent": None,
            "sina": None, "akshare": None, "eastmoney": None,
        })
        fd.fetch_data("000688", "2023-01-01", force=True, provider_name="baostock")
        assert calls["baostock"] == [("000688", "2023-01-01")]


class TestSinaProvider:
    """SinaProvider：normalize 单位换算、列映射、日期过滤"""

    def _sina_df(self):
        return pd.DataFrame({
            "date": ["2023-01-04", "2023-01-03"],  # 故意乱序
            "open": [1.023, 1.001],
            "high": [1.026, 1.024],
            "low": [1.008, 0.998],
            "close": [1.016, 1.023],
            "volume": [1969422400.0, 1778509300.0],  # 股
            "amount": [2.0e9, 1.8e9],
            "postVol": [0.0, 0.0],
            "postAmt": [0.0, 0.0],
        })

    def test_normalize_etf_volume_divided_by_100(self):
        provider = SinaProvider()
        out = provider.normalize(self._sina_df(), "588000")
        assert list(out.columns) == [
            "date", "open", "close", "high", "low", "volume", "amount",
            "amplitude", "change_pct", "change", "turnover", "symbol",
        ]
        # volume 股→手（÷100）；normalize 按 date 升序 → iloc[0] 为 01-03
        assert out["volume"].iloc[0] == 17785093.0
        assert out["volume"].iloc[1] == 19694224.0
        assert (out["symbol"] == "588000").all()
        assert pd.api.types.is_datetime64_any_dtype(out["date"])
        assert list(out["date"]) == sorted(out["date"])  # 按 date 升序

    def test_normalize_index_volume_divided_by_100(self):
        provider = SinaProvider()
        df = self._sina_df()
        # 指数（stock_zh_index_daily）实测 volume 单位也是股 → 同样 ÷100 对齐手
        out = provider.normalize(df, "000688")
        assert out["volume"].iloc[0] == 17785093.0
        assert out["volume"].iloc[1] == 19694224.0

    def test_normalize_missing_column_raises(self):
        provider = SinaProvider()
        with pytest.raises(KeyError):
            provider.normalize(pd.DataFrame({"date": ["2023-01-03"]}), "588000")

    def test_market_prefix(self):
        provider = SinaProvider()
        assert provider._market_prefix("588000") == "sh"
        assert provider._market_prefix("159915") == "sz"
        assert provider._market_prefix("sh000688") == "sh"
        assert provider._is_etf("sh588000") is True
        assert provider._is_etf("000688") is False

    def test_fetch_daily_filters_dates_and_selects_interface(self, monkeypatch):
        """ETF 走 fund_etf_hist_sina；本地按 start~end 过滤"""
        import types as _types
        ak = _types.ModuleType("akshare")
        seen = {}

        def fake_etf(symbol):
            seen["symbol"] = symbol
            return self._sina_df()

        ak.fund_etf_hist_sina = fake_etf
        monkeypatch.setitem(sys.modules, "akshare", ak)
        provider = SinaProvider()
        df = provider.fetch_daily("588000", "2023-01-03", "2023-01-04")
        assert seen["symbol"] == "sh588000"  # 自动补市场前缀
        # 日期过滤（fetch_daily 不排序，normalize 才排序）
        assert sorted(df["date"]) == ["2023-01-03", "2023-01-04"]

        # 指数走 stock_zh_index_daily
        def fake_index(symbol):
            seen["index_symbol"] = symbol
            return self._sina_df()

        ak.stock_zh_index_daily = fake_index
        df2 = provider.fetch_daily("sh000688", "2023-01-01", "2023-12-31")
        assert seen["index_symbol"] == "sh000688"  # 已带前缀直接使用
        assert len(df2) == 2
