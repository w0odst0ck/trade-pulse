#!/usr/bin/env python3
"""backtest.py 多标的支持单元测试（pytest，不联网、不跑真实回测）

覆盖验收：
  - --symbol 参数覆盖 config['symbol']，输出 data/{symbol}/backtest/metrics.json
  - 报告标题动态显示 symbol
  - 非默认标的未指定策略时提示不加载 588000 调优参数（默认行为不变）
  - 不带 --symbol 时回落 config['symbol']（588000），行为与改动前一致
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import backtest as bt

TRADE_COLS = ["action", "entry_date", "entry_price", "entry_value", "entry_shares",
              "exit_date", "exit_price", "exit_value", "return",
              "signal_date", "signal_score"]


def make_features(n=40, end="2026-08-05") -> pd.DataFrame:
    """与 features_cache.csv 同结构的日线特征（date/close/total_score 等）"""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range(end=end, periods=n)
    close = 1.5 * (1 + rng.normal(0, 0.01, n)).cumprod()
    return pd.DataFrame({
        "date": dates, "close": close, "open": close, "high": close * 1.01,
        "low": close * 0.99, "volume": 1e7, "amount": 1e7 * close,
        "momentum": rng.uniform(-1, 1, n), "trend": rng.uniform(-1, 1, n),
        "volume_price": rng.uniform(-1, 1, n), "rsrs": rng.uniform(-1, 1, n),
        "relative_strength": rng.uniform(-1, 1, n),
        "weekly_modifier": 0.0, "ma60_slope": 0.0,
        "total_score": rng.uniform(-0.5, 0.5, n),
    })


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLS)


def load_test_config(tmp_path) -> dict:
    """真实 config + data_dir 指向临时目录（输出落在 tmp，不动工作区 data/）"""
    with open(bt.CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = dict(cfg)
    cfg["data_dir"] = str(tmp_path)
    return cfg


def stub_backtest_pipeline(monkeypatch, tmp_path, features_df, cfg):
    """把 main() 的回测/基准/加载环节全部替换为轻量 stub，仅验证 symbol 传导与输出"""
    def fake_load_features(symbol):
        return features_df.copy()

    def fake_run_backtest(features_df, config, start, end, cost_rate):
        eq = features_df[["date"]].copy()
        eq["equity"] = 1.0
        eq["signal"] = "空仓"
        eq["total_score"] = 0.0
        eq["position"] = 0.0
        eq["cash"] = 1.0
        eq["hold_value"] = 0.0
        return {"trades": empty_trades(), "equity_curve": eq,
                "final_value": 1.0, "risk_events": []}

    def fake_buy_hold(features_df, start, end):
        df = features_df[["date"]].copy()
        df["benchmark_equity"] = 1.0
        return df

    def fake_ma_crossover(features_df, start, end, cost_rate):
        eq = features_df[["date"]].copy()
        eq["equity"] = 1.0
        return {"trades": empty_trades(), "equity_curve": eq, "final_value": 1.0}

    monkeypatch.setattr(bt, "load_config", lambda: cfg)
    monkeypatch.setattr(bt, "load_features_df", fake_load_features)
    monkeypatch.setattr(bt, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(bt, "run_buy_hold", fake_buy_hold)
    monkeypatch.setattr(bt, "run_ma_crossover", fake_ma_crossover)


class TestSymbolArg:
    def test_symbol_override_outputs_to_symbol_dir(self, tmp_path, monkeypatch, capsys):
        cfg = load_test_config(tmp_path)
        features = make_features()
        stub_backtest_pipeline(monkeypatch, tmp_path, features, cfg)
        monkeypatch.setattr(sys, "argv",
                            ["backtest.py", "--symbol", "515050"])

        bt.main()
        out = capsys.readouterr().out

        # 输出目录 data/515050/backtest/
        metrics_path = tmp_path / "515050" / "backtest" / "metrics.json"
        assert metrics_path.exists()
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        assert "annual_return" in m
        # 报告标题动态
        assert "515050 日线择时回测报告" in out
        # 未指定策略时提示不加载 588000 调优参数
        assert "[INFO] --symbol 515050" in out
        assert "strategies/588000.yaml" in out

    def test_default_symbol_falls_back_to_config(self, tmp_path, monkeypatch, capsys):
        """不带 --symbol → config['symbol']（588000），输出原默认目录"""
        cfg = load_test_config(tmp_path)
        features = make_features()
        stub_backtest_pipeline(monkeypatch, tmp_path, features, cfg)
        monkeypatch.setattr(sys, "argv", ["backtest.py"])

        bt.main()
        out = capsys.readouterr().out

        metrics_path = tmp_path / cfg["symbol"] / "backtest" / "metrics.json"
        assert metrics_path.exists()
        assert "588000 日线择时回测报告" in out
        # 默认标的不打印 INFO 提示（与改动前输出一致）
        assert "[INFO] --symbol" not in out

    def test_print_report_symbol_default(self, capsys):
        """print_report 不传 symbol 时默认 588000 标题（回归保护）"""
        features = make_features()
        eq = features[["date"]].copy()
        eq["equity"] = 1.0
        m = bt.compute_metrics(eq["equity"], trades_df=empty_trades(), n_days=len(eq))
        bt.print_report(m, m, m, {}, "2026-01-01", "2026-08-05", len(eq), 0.00055)
        out = capsys.readouterr().out
        assert "588000 日线择时回测报告" in out

    def test_print_report_symbol_custom(self, capsys):
        """print_report 传 symbol 时标题用该标的"""
        features = make_features()
        eq = features[["date"]].copy()
        eq["equity"] = 1.0
        m = bt.compute_metrics(eq["equity"], trades_df=empty_trades(), n_days=len(eq))
        bt.print_report(m, m, m, {}, "2026-01-01", "2026-08-05", len(eq), 0.00055,
                        symbol="515050")
        out = capsys.readouterr().out
        assert "515050 日线择时回测报告" in out
