#!/usr/bin/env python3
"""compute_features.py 多标的支持单元测试（pytest，不联网）

覆盖验收：
  - --symbol/--benchmark 参数覆盖 config 默认值（compute_all_features symbol 参数）
  - 特征缓存按 data/{symbol}/features_cache.csv 读写（与 588000 同结构）
  - 默认（symbol=None）回落到 config['symbol']，原 588000 行为不变
  - 相对强度因子用传入 benchmark 计算，不依赖 588000 专属逻辑
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import compute_features as cf

FEATURE_COLS = [
    "date", "close", "volume", "momentum", "trend", "volume_price", "rsrs",
    "relative_strength", "weekly_modifier", "ma60_slope", "total_score",
]


def make_daily(n=200, end="2026-08-05", base_close=1.5, seed=42) -> pd.DataFrame:
    """构造 n 个交易日日线（工作日近似交易日），列与 daily.csv 对齐"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n)
    close = base_close * (1 + rng.normal(0, 0.01, n)).cumprod()
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.uniform(1e7, 2e7, n)
    return pd.DataFrame({
        "date": dates, "open": open_, "close": close,
        "high": high, "low": low, "volume": volume,
        "amount": volume * close,
    })


def load_test_config(tmp_path) -> dict:
    """真实 config + data_dir 指向临时目录（缓存读写落在 tmp，不动工作区 data/）"""
    with open(cf.CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = dict(cfg)
    cfg["data_dir"] = str(tmp_path)
    return cfg


class TestSymbolParam:
    """compute_all_features 的 symbol 参数：默认回落 / 覆盖写目录"""

    def test_default_symbol_falls_back_to_config(self, tmp_path, monkeypatch):
        cfg = load_test_config(tmp_path)
        monkeypatch.setattr(cf, "load_config", lambda: cfg)
        df = make_daily()
        bench = make_daily(seed=1, base_close=900.0)
        out = cf.compute_all_features(df, bench, cfg, force=True)  # symbol=None

        # 缓存写到 config['symbol']（588000）目录，与 daily 行数对齐
        path = tmp_path / cfg["symbol"] / "features_cache.csv"
        assert path.exists()
        cached = pd.read_csv(path, parse_dates=["date"])
        assert len(cached) == len(df)
        # 与 588000 同结构：列齐全
        for col in FEATURE_COLS:
            assert col in cached.columns
        # 返回对象与缓存一致
        assert len(out) == len(df)

    def test_symbol_override_writes_to_symbol_dir(self, tmp_path, monkeypatch):
        cfg = load_test_config(tmp_path)
        monkeypatch.setattr(cf, "load_config", lambda: cfg)
        df = make_daily()
        bench = make_daily(seed=1, base_close=900.0)
        out = cf.compute_all_features(df, bench, cfg, force=True, symbol="515050")

        assert (tmp_path / "515050" / "features_cache.csv").exists()
        # 未误写默认标的目录
        assert not (tmp_path / cfg["symbol"] / "features_cache.csv").exists()
        cached = pd.read_csv(tmp_path / "515050" / "features_cache.csv",
                             parse_dates=["date"])
        assert len(cached) == len(df)
        # 与 588000 缓存结构一致（列集合相同）
        assert set(FEATURE_COLS) == set(cached.columns)

    def test_relative_strength_uses_benchmark(self, tmp_path, monkeypatch):
        """相对强度因子与 benchmark 对齐计算（515050 用 000688，非 588000 专属逻辑）"""
        cfg = load_test_config(tmp_path)
        monkeypatch.setattr(cf, "load_config", lambda: cfg)
        df = make_daily(seed=3)
        bench = make_daily(seed=7, base_close=900.0)
        out = cf.compute_all_features(df, bench, cfg, force=True, symbol="515050")

        # 对齐后应产生非空相对强度值（merge 后覆盖大部分交易日）
        assert out["relative_strength"].notna().sum() > len(out) * 0.5

    def test_incremental_mode_uses_symbol_dir(self, tmp_path, monkeypatch):
        """增量模式（force=False，已有缓存）读写同一 symbol 目录"""
        cfg = load_test_config(tmp_path)
        monkeypatch.setattr(cf, "load_config", lambda: cfg)
        df = make_daily(n=100)
        bench = make_daily(seed=1, base_close=900.0)
        cf.compute_all_features(df, bench, cfg, force=True, symbol="515050")

        # 追加 10 天新数据 → 增量计算
        extra = make_daily(n=10, end="2026-08-19", seed=5)
        new_df = pd.concat([df, extra], ignore_index=True).drop_duplicates(
            subset=["date"]).sort_values("date").reset_index(drop=True)
        out = cf.compute_all_features(new_df, bench, cfg, force=False, symbol="515050")
        assert len(out) == len(new_df)
        cached = pd.read_csv(tmp_path / "515050" / "features_cache.csv",
                             parse_dates=["date"])
        assert cached["date"].max() == new_df["date"].max()


class TestMainArgs:
    """main() 的 --symbol/--benchmark 参数传导（monkeypatch 数据与计算，不联网不写盘）"""

    def test_symbol_and_benchmark_propagate(self, tmp_path, monkeypatch, capsys):
        cfg = load_test_config(tmp_path)
        monkeypatch.setattr(cf, "load_config", lambda: cfg)
        df = make_daily()
        monkeypatch.setattr(cf, "load_data", lambda sym: df)
        calls = {}

        def spy_compute(df_sym, df_bench, config, force=False, symbol=None):
            calls["symbol"] = symbol
            calls["bench_df_len"] = len(df_bench)
            return df_sym.copy()

        monkeypatch.setattr(cf, "compute_all_features", spy_compute)
        monkeypatch.setattr(cf, "get_latest_features", lambda sym: {"date": "2026-08-05"})
        monkeypatch.setattr(sys, "argv",
                            ["compute_features.py", "--symbol", "515050",
                             "--benchmark", "000688", "--output", "panel"])
        cf.main()

        assert calls["symbol"] == "515050"
        assert calls["bench_df_len"] == len(df)

    def test_default_args_fall_back_to_config(self, tmp_path, monkeypatch, capsys):
        """不带 --symbol/--benchmark 时回落 config 默认值（symbol=None 由内部回落）"""
        cfg = load_test_config(tmp_path)
        monkeypatch.setattr(cf, "load_config", lambda: cfg)
        df = make_daily()
        monkeypatch.setattr(cf, "load_data", lambda sym: df)
        calls = {}

        def spy_compute(df_sym, df_bench, config, force=False, symbol=None):
            calls["symbol"] = symbol
            return df_sym.copy()

        monkeypatch.setattr(cf, "compute_all_features", spy_compute)
        monkeypatch.setattr(cf, "get_latest_features", lambda sym: {})
        monkeypatch.setattr(sys, "argv", ["compute_features.py"])
        cf.main()

        # main 层把 config 默认值解析进 symbol 再传给 compute_all_features
        assert calls["symbol"] == cfg["symbol"]

    def test_output_panel_prints_latest(self, tmp_path, monkeypatch, capsys):
        """--output panel（默认）打印最新因子面板"""
        cfg = load_test_config(tmp_path)
        monkeypatch.setattr(cf, "load_config", lambda: cfg)
        df = make_daily()
        monkeypatch.setattr(cf, "load_data", lambda sym: df)
        monkeypatch.setattr(cf, "compute_all_features",
                            lambda *a, **k: df.copy())
        monkeypatch.setattr(cf, "get_latest_features",
                            lambda sym: {"date": "2026-08-05", "momentum": 0.5})
        monkeypatch.setattr(sys, "argv",
                            ["compute_features.py", "--symbol", "515050"])
        cf.main()
        out = capsys.readouterr().out
        assert "515050" in out or "momentum" in out
