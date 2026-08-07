#!/usr/bin/env python3
"""health_check.py 五源健康度 + 特征滞后单元测试（pytest，不联网）

覆盖验收：
  - 五源各探活逻辑：tencent fqkline 拉 1 根 / sina 新浪 / akshare 东财 /
    eastmoney 东财 / baostock login
  - 每源 OK/FAIL + 原因；全部 OK 汇总一行；有 FAIL 才列出异常源
  - 冷却源跳过：source_health.json 中 cooldown_until 未到期的源不真实探活，
    标 ok=True + SKIP(cooldown)；文件缺失/损坏降级全部真实探活并 WARN
  - 特征滞后检查：features_cache 最新日期 vs 最近交易日
  - 代理环境变量清除（东财系直连，与 fetch_data 一致）
"""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import health_check as hc


class FakeResp:
    """模拟 requests.Response（仅暴露 json()）"""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def make_tencent_ok_payload(tsym="sh588000"):
    return {"code": 0, "data": {tsym: {"qfqday": [["2026-08-05", "1.0", "1.0", "1.0", "1.0", "1000"]]}}}


class TestClearProxyEnv:
    def test_clears_all_proxy_vars(self, monkeypatch):
        for k in ("http_proxy", "HTTPS_PROXY", "all_proxy"):
            monkeypatch.setenv(k, "http://proxy:8080")
        hc._clear_proxy_env()
        for k in ("http_proxy", "https_proxy", "all_proxy",
                  "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            assert k not in os.environ


class TestCheckTencent:
    @pytest.fixture(autouse=True)
    def _patch_resolve(self, monkeypatch):
        """与真实 config 解耦：resolve_tencent_symbol 固定返回 sh588000（测试 hermetic）"""
        monkeypatch.setattr(
            "fetch_data.resolve_tencent_symbol",
            lambda symbol: "sh588000" if symbol == "588000" else symbol,
        )

    def test_ok_with_klines(self, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            assert "param=sh588000,day" in url
            assert ",1,qfq" in url  # 拉 1 根
            return FakeResp(make_tencent_ok_payload())

        monkeypatch.setattr("requests.get", fake_get)
        r = hc._check_tencent("588000")
        assert r["ok"] is True
        assert "2026-08-05" in r["msg"]

    def test_empty_klines_fails(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda url, headers=None, timeout=None: FakeResp(
                {"code": 0, "data": {"sh588000": {"qfqday": []}}}),
        )
        r = hc._check_tencent("588000")
        assert r["ok"] is False
        assert "空" in r["msg"]

    def test_network_error_fails_with_reason(self, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            raise ConnectionError("boom")

        monkeypatch.setattr("requests.get", fake_get)
        r = hc._check_tencent("588000")
        assert r["ok"] is False
        assert "boom" in r["msg"]


class TestCheckAkShare:
    def test_ok_with_df(self, monkeypatch):
        df = pd.DataFrame({"date": [pd.Timestamp("2026-08-05")], "close": [1.0]})

        def fake_fetch(self, symbol, start_date, end_date="20500101"):
            return df

        monkeypatch.setattr("data_provider.akshare.AkShareProvider.fetch_daily", fake_fetch)
        r = hc._check_akshare("588000")
        assert r["ok"] is True
        assert "2026-08-05" in r["msg"]

    def test_empty_df_fails(self, monkeypatch):
        monkeypatch.setattr(
            "data_provider.akshare.AkShareProvider.fetch_daily",
            lambda self, symbol, start_date, end_date="20500101": None,
        )
        r = hc._check_akshare("588000")
        assert r["ok"] is False
        assert "空数据" in r["msg"]

    def test_exception_fails_as_source_fault(self, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date="20500101"):
            raise RuntimeError("IP 风控拒绝")

        monkeypatch.setattr("data_provider.akshare.AkShareProvider.fetch_daily", fake_fetch)
        r = hc._check_akshare("588000")
        assert r["ok"] is False
        assert "IP 风控拒绝" in r["msg"]  # 标注原因


class TestCheckEastMoney:
    def test_ok_with_df(self, monkeypatch):
        df = pd.DataFrame({"date": [pd.Timestamp("2026-08-05")], "close": [1.0]})
        monkeypatch.setattr(
            "data_provider.fallback.EastMoneyProvider.fetch_daily",
            lambda self, symbol, start_date, end_date="20500101": df,
        )
        r = hc._check_eastmoney("588000")
        assert r["ok"] is True

    def test_exception_fails(self, monkeypatch):
        monkeypatch.setattr(
            "data_provider.fallback.EastMoneyProvider.fetch_daily",
            lambda self, symbol, start_date, end_date="20500101": (_ for _ in ()).throw(
                RuntimeError("eastmoney down")),
        )
        r = hc._check_eastmoney("588000")
        assert r["ok"] is False
        assert "eastmoney down" in r["msg"]


class TestCheckBaoStock:
    class FakeBS:
        def logout(self):
            pass

    def test_login_ok(self, monkeypatch):
        monkeypatch.setattr(
            "data_provider.baostock.BaoStockProvider._login",
            lambda self: TestCheckBaoStock.FakeBS(),
        )
        r = hc._check_baostock("588000")
        assert r["ok"] is True
        assert "login" in r["msg"]

    def test_login_fail(self, monkeypatch):
        monkeypatch.setattr(
            "data_provider.baostock.BaoStockProvider._login",
            lambda self: (_ for _ in ()).throw(RuntimeError("login 拒绝")),
        )
        r = hc._check_baostock("588000")
        assert r["ok"] is False
        assert "login" in r["msg"]


class TestCheckSina:
    @pytest.fixture(autouse=True)
    def _patch_resolve(self, monkeypatch):
        """与真实 config 解耦：resolve_tencent_symbol 固定返回 sh588000"""
        monkeypatch.setattr(
            "fetch_data.resolve_tencent_symbol",
            lambda symbol: "sh588000" if symbol == "588000" else symbol,
        )

    def test_ok_with_df(self, monkeypatch):
        df = pd.DataFrame({"date": [pd.Timestamp("2026-08-06")], "close": [1.0]})

        def fake_fetch(self, symbol, start_date, end_date="20500101"):
            assert symbol == "sh588000"  # 与 fetch_data 同口径：带市场前缀
            return df

        monkeypatch.setattr("data_provider.sina.SinaProvider.fetch_daily", fake_fetch)
        r = hc._check_sina("588000")
        assert r["ok"] is True
        assert "2026-08-06" in r["msg"]

    def test_empty_df_fails(self, monkeypatch):
        monkeypatch.setattr(
            "data_provider.sina.SinaProvider.fetch_daily",
            lambda self, symbol, start_date, end_date="20500101": None,
        )
        r = hc._check_sina("588000")
        assert r["ok"] is False
        assert "空数据" in r["msg"]

    def test_exception_fails_as_source_fault(self, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date="20500101"):
            raise RuntimeError("sina 限流")

        monkeypatch.setattr("data_provider.sina.SinaProvider.fetch_daily", fake_fetch)
        r = hc._check_sina("588000")
        assert r["ok"] is False
        assert "sina 限流" in r["msg"]  # 标注原因


class TestLoadCooldown:
    def _health_json(self, **cooldowns):
        """构造 source_health.json 结构：{symbol: {src: {cooldown_until: ...}}}，
        未列出的源 cooldown_until=0"""
        health = {}
        for symbol, srcs in {"588000": cooldowns, "000688": cooldowns}.items():
            health[symbol] = {
                src: {"last_fetch_ts": 0, "last_max_date": None,
                      "consecutive_failures": 0, "cooldown_until": cu}
                for src, cu in srcs.items()
            }
        return health

    def test_future_cooldown_collected(self, tmp_path, monkeypatch):
        future = time.time() + 3600
        payload = self._health_json(akshare=future, eastmoney=future,
                                    tencent=0, sina=0, baostock=0)
        (tmp_path / "source_health.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(hc, "SCRIPT_DIR", tmp_path)
        cooldown, warn = hc._load_cooldown()
        assert cooldown == {"akshare", "eastmoney"}
        assert warn == ""

    def test_expired_cooldown_not_collected(self, tmp_path, monkeypatch):
        payload = self._health_json(akshare=time.time() - 3600, eastmoney=0)
        (tmp_path / "source_health.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(hc, "SCRIPT_DIR", tmp_path)
        cooldown, warn = hc._load_cooldown()
        assert cooldown == set()
        assert warn == ""

    def test_missing_file_falls_back_with_warn(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hc, "SCRIPT_DIR", tmp_path)
        cooldown, warn = hc._load_cooldown()
        assert cooldown == set()
        assert "缺失" in warn  # 不静默：警告降级为全部真实探活

    def test_corrupt_file_falls_back_with_warn(self, tmp_path, monkeypatch):
        (tmp_path / "source_health.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(hc, "SCRIPT_DIR", tmp_path)
        cooldown, warn = hc._load_cooldown()
        assert cooldown == set()
        assert "解析失败" in warn


class TestCheckProviderHealth:
    def test_all_ok_silent_summary(self, monkeypatch):
        monkeypatch.setattr(hc, "_load_cooldown", lambda: (set(), ""))
        for name in ("_check_tencent", "_check_sina", "_check_akshare",
                     "_check_eastmoney", "_check_baostock"):
            monkeypatch.setattr(hc, name, lambda sym: {"ok": True, "msg": "fine"})

        r = hc.check_provider_health()
        assert r["ok"] is True
        assert "五源健康" in r["msg"]
        assert set(r["sources"].keys()) == {
            "tencent", "sina", "akshare", "eastmoney", "baostock"}

    def test_partial_fail_lists_failed_sources(self, monkeypatch):
        monkeypatch.setattr(hc, "_load_cooldown", lambda: (set(), ""))
        monkeypatch.setattr(hc, "_check_tencent", lambda sym: {"ok": True, "msg": "fine"})
        monkeypatch.setattr(hc, "_check_sina", lambda sym: {"ok": True, "msg": "fine"})
        monkeypatch.setattr(hc, "_check_akshare", lambda sym: {"ok": False, "msg": "风控"})
        monkeypatch.setattr(hc, "_check_eastmoney", lambda sym: {"ok": False, "msg": "风控"})
        monkeypatch.setattr(hc, "_check_baostock", lambda sym: {"ok": True, "msg": "fine"})

        r = hc.check_provider_health()
        assert r["ok"] is False
        assert "2 个数据源异常" in r["msg"]
        assert "akshare" in r["msg"] and "eastmoney" in r["msg"]
        assert "tencent" not in r["msg"].split("异常")[1]

    def test_missing_health_file_falls_back_to_all_live(self, tmp_path, monkeypatch):
        """source_health.json 缺失 → 降级为全部真实探活（不静默跳过），并 WARN"""
        monkeypatch.setattr(hc, "SCRIPT_DIR", tmp_path)  # 目录内无 source_health.json
        called = []

        def fake_check(name):
            def _inner(sym):
                called.append(name)
                return {"ok": True, "msg": "fine"}
            return _inner

        for name in ("_check_tencent", "_check_sina", "_check_akshare",
                     "_check_eastmoney", "_check_baostock"):
            monkeypatch.setattr(hc, name, fake_check(name))

        r = hc.check_provider_health()
        assert r["ok"] is True
        assert set(called) == {"_check_tencent", "_check_sina", "_check_akshare",
                               "_check_eastmoney", "_check_baostock"}

    def test_cooldown_and_fail_mixed(self, monkeypatch):
        """冷却源 SKIP（ok=True 不计故障）+ 另一源真实 FAIL → 整体异常且只列 FAIL 源"""
        monkeypatch.setattr(hc, "_load_cooldown", lambda: ({"akshare"}, ""))
        monkeypatch.setattr(hc, "_check_tencent", lambda sym: {"ok": True, "msg": "fine"})
        monkeypatch.setattr(hc, "_check_sina", lambda sym: {"ok": False, "msg": "sina down"})
        monkeypatch.setattr(hc, "_check_akshare", lambda sym: (_ for _ in ()).throw(
            AssertionError("冷却中的 akshare 不应被探活")))
        monkeypatch.setattr(hc, "_check_eastmoney", lambda sym: {"ok": True, "msg": "fine"})
        monkeypatch.setattr(hc, "_check_baostock", lambda sym: {"ok": True, "msg": "fine"})

        r = hc.check_provider_health()
        assert r["ok"] is False
        assert "1 个数据源异常" in r["msg"]
        assert "sina" in r["msg"]
        assert "akshare" not in r["msg"]  # 冷却源不计入异常
        assert r["sources"]["akshare"]["ok"] is True
        assert "SKIP" in r["sources"]["akshare"]["msg"]

    def test_cooldown_sources_skipped_without_request(self, tmp_path, monkeypatch):
        """source_health.json 中 cooldown_until 未来 → 对应源不真实探活：
        ok=True 且 msg 含 SKIP；且探活函数未被调用（此处直接抛异常验证）"""
        future = time.time() + 3600
        health = {"588000": {
            "tencent": {"cooldown_until": 0},
            "sina": {"cooldown_until": 0},
            "akshare": {"cooldown_until": future},
            "eastmoney": {"cooldown_until": future},
            "baostock": {"cooldown_until": 0},
        }}
        (tmp_path / "source_health.json").write_text(
            json.dumps(health, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(hc, "SCRIPT_DIR", tmp_path)

        # 非冷却源正常探活；冷却源若被调用则抛异常 → 用例失败
        monkeypatch.setattr(hc, "_check_tencent", lambda sym: {"ok": True, "msg": "fine"})
        monkeypatch.setattr(hc, "_check_sina", lambda sym: {"ok": True, "msg": "fine"})
        monkeypatch.setattr(hc, "_check_akshare", lambda sym: (_ for _ in ()).throw(
            AssertionError("冷却中的 akshare 不应被探活")))
        monkeypatch.setattr(hc, "_check_eastmoney", lambda sym: (_ for _ in ()).throw(
            AssertionError("冷却中的 eastmoney 不应被探活")))
        monkeypatch.setattr(hc, "_check_baostock", lambda sym: {"ok": True, "msg": "fine"})

        r = hc.check_provider_health()
        assert r["ok"] is True
        assert "五源健康" in r["msg"]
        assert r["sources"]["akshare"] == {"ok": True, "msg": "SKIP(cooldown)，跳过探活"}
        assert r["sources"]["eastmoney"] == {"ok": True, "msg": "SKIP(cooldown)，跳过探活"}

    def test_main_symbol_reads_config(self, tmp_path, monkeypatch):
        """_main_symbol 读 config.json 的 symbol"""
        cfg_path = hc.CONFIG_PATH
        monkeypatch.setattr(hc, "CONFIG_PATH", tmp_path / "config.json")
        (tmp_path / "config.json").write_text('{"symbol": "515050"}', encoding="utf-8")
        assert hc._main_symbol() == "515050"
        assert cfg_path.exists()  # 真实 config 未被改动


class TestCheckFeaturesStaleness:
    """特征滞后检测：features_cache 最新日期 vs 最近交易日（严格判定）"""

    @pytest.fixture(autouse=True)
    def _patch_env(self, tmp_path, monkeypatch):
        """隔离真实 data/ 与真实日历：固定 DATA_DIR 与最近交易日"""
        monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
        monkeypatch.setattr(hc, "_recent_trading_day", lambda: date(2026, 8, 6))

    def _write_features(self, tmp_path, dates):
        df = pd.DataFrame({"date": pd.to_datetime(dates), "close": [1.0] * len(dates)})
        df.to_csv(tmp_path / "features_cache.csv", index=False)

    def test_lagged_features_fail(self, tmp_path):
        """特征末日 08-05 < 最近交易日 08-06 → 滞后 1 天，ok=False"""
        self._write_features(tmp_path, ["2026-08-04", "2026-08-05"])
        r = hc.check_features_staleness()
        assert r["ok"] is False
        assert "滞后 1 天" in r["msg"]
        assert r["lag_days"] == 1
        assert r["latest"] == "2026-08-05"

    def test_up_to_date_features_ok(self, tmp_path):
        """特征末日 == 最近交易日 → ok=True"""
        self._write_features(tmp_path, ["2026-08-05", "2026-08-06"])
        r = hc.check_features_staleness()
        assert r["ok"] is True
        assert r["lag_days"] == 0

    def test_ahead_features_ok(self, tmp_path):
        """特征末日 > 最近交易日（数据超前等）→ 不算滞后，ok=True"""
        self._write_features(tmp_path, ["2026-08-06", "2026-08-07"])
        r = hc.check_features_staleness()
        assert r["ok"] is True
        assert r["lag_days"] == -1
        assert "已更新至 2026-08-07" in r["msg"]

    def test_missing_file_fails(self, tmp_path):
        r = hc.check_features_staleness()
        assert r["ok"] is False
        assert "不存在" in r["msg"]

    def test_empty_file_fails(self, tmp_path):
        (tmp_path / "features_cache.csv").write_text(
            "date,close\n", encoding="utf-8")
        r = hc.check_features_staleness()
        assert r["ok"] is False
        assert "空" in r["msg"]


class TestCheckState:
    """check_state：损坏/结构异常的 state.json 应报 ok=False 而非崩溃"""

    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
        r = hc.check_state()
        assert r["ok"] is False
        assert "不存在" in r["msg"]

    def test_broken_json(self, tmp_path, monkeypatch):
        (tmp_path / "state.json").write_text("{broken", encoding="utf-8")
        monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
        r = hc.check_state()
        assert r["ok"] is False
        assert "解析失败" in r["msg"]

    def test_missing_state_field(self, tmp_path, monkeypatch):
        (tmp_path / "state.json").write_text('{"foo": 1}', encoding="utf-8")
        monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
        r = hc.check_state()
        assert r["ok"] is False
        assert "结构异常" in r["msg"]

    def test_normal(self, tmp_path, monkeypatch):
        (tmp_path / "state.json").write_text('{"state": "空仓"}', encoding="utf-8")
        monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
        r = hc.check_state()
        assert r["ok"] is True
        assert "空仓" in r["msg"]


class TestCheckGit:
    """check_git：git 命令失败应报 ok=False（部署链路异常），而非静默 ok=True"""

    def test_git_failure_reports_false(self, tmp_path, monkeypatch):
        """非 git 目录 → ok=False + 明确报错"""
        monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)  # tmp_path 非 git 仓库
        r = hc.check_git()
        assert r["ok"] is False
        assert "git 检查失败" in r["msg"]
