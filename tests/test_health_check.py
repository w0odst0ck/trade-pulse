#!/usr/bin/env python3
"""health_check.py 四源健康度单元测试（pytest，不联网）

覆盖验收：
  - 四源各探活逻辑：tencent fqkline 拉 1 根 / akshare 东财 / eastmoney 东财 / baostock login
  - 每源 OK/FAIL + 原因；全部 OK 汇总一行；有 FAIL 才列出异常源
  - 代理环境变量清除（东财系直连，与 fetch_data 一致）
"""

import os
import sys
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


class TestCheckProviderHealth:
    def test_all_ok_silent_summary(self, monkeypatch):
        for name in ("_check_tencent", "_check_akshare", "_check_eastmoney", "_check_baostock"):
            monkeypatch.setattr(hc, name, lambda sym: {"ok": True, "msg": "fine"})

        r = hc.check_provider_health()
        assert r["ok"] is True
        assert "四源健康" in r["msg"]
        assert set(r["sources"].keys()) == {"tencent", "akshare", "eastmoney", "baostock"}

    def test_partial_fail_lists_failed_sources(self, monkeypatch):
        monkeypatch.setattr(hc, "_check_tencent", lambda sym: {"ok": True, "msg": "fine"})
        monkeypatch.setattr(hc, "_check_akshare", lambda sym: {"ok": False, "msg": "风控"})
        monkeypatch.setattr(hc, "_check_eastmoney", lambda sym: {"ok": False, "msg": "风控"})
        monkeypatch.setattr(hc, "_check_baostock", lambda sym: {"ok": True, "msg": "fine"})

        r = hc.check_provider_health()
        assert r["ok"] is False
        assert "2 个数据源异常" in r["msg"]
        assert "akshare" in r["msg"] and "eastmoney" in r["msg"]
        assert "tencent" not in r["msg"].split("异常")[1]

    def test_main_symbol_reads_config(self, tmp_path, monkeypatch):
        """_main_symbol 读 config.json 的 symbol"""
        cfg_path = hc.CONFIG_PATH
        monkeypatch.setattr(hc, "CONFIG_PATH", tmp_path / "config.json")
        (tmp_path / "config.json").write_text('{"symbol": "515050"}', encoding="utf-8")
        assert hc._main_symbol() == "515050"
        assert cfg_path.exists()  # 真实 config 未被改动
