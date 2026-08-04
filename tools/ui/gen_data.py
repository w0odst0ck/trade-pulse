#!/usr/bin/env python3
"""
gen_data.py — 生成共享数据文件 docs/assets/data.json

原子写 + 校验：先写临时文件，校验 JSON 合法且关键字段存在，再 rename。
失败返回非零退出码（build_ui.py / cron 据此判断是否提交）。

用法：
  python gen_data.py
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "588000"
OUT_DIR = PROJECT_ROOT / "docs" / "assets"
OUT_PATH = OUT_DIR / "data.json"
STATE_PATH = DATA_DIR / "state.json"


def load_csv(path: Path):
    """读 CSV，容忍不存在 → 返回空列表"""
    import pandas as pd
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"  [WARN] 读取 {path.name} 失败: {e}")
        return []


def map_backtest_trades(records: list) -> list:
    """把回测 trades 映射为 trade_log 的展示 schema（占位模式前端统一读取）

    trade_log 列: date/symbol/action/score/position_pct/entry_price/exit_price/pnl_pct/note
    backtest 列: action/entry_date/entry_price/.../exit_date/exit_price/return/signal_score
    只有已平仓批次（exit_date 非空）才展示盈亏，未平仓批次 pnl_pct 留空。
    """
    out = []
    for r in records:
        closed = r.get("exit_date") is not None and str(r.get("exit_date", "")) != ""
        pnl = None
        if closed:
            ret = r.get("return")
            if isinstance(ret, (int, float)) and ret == ret:  # 非 NaN
                pnl = round(float(ret) * 100, 2)
        out.append({
            "date": str(r.get("entry_date", ""))[:10],
            "symbol": "588000",
            "action": r.get("action", ""),
            "score": r.get("signal_score"),
            "position_pct": r.get("entry_value"),
            "entry_price": r.get("entry_price"),
            "exit_price": r.get("exit_price") if closed else None,
            "pnl_pct": pnl,
            "note": "",
        })
    return out


def build_data() -> dict:
    import pandas as pd

    state = {}
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)

    daily = load_csv(DATA_DIR / "daily.csv")
    features = load_csv(DATA_DIR / "features_cache.csv")
    equity = load_csv(DATA_DIR / "backtest" / "equity_curve.csv")
    paper_equity = load_csv(DATA_DIR / "paper" / "paper_equity.csv")
    trades = load_csv(DATA_DIR / "backtest" / "trades.csv")
    trade_log = load_csv(PROJECT_ROOT / "data" / "trade_log.csv")

    # 绩效指标（backtest 输出 metrics.json；不存在则不展示）
    metrics = {}
    metrics_path = DATA_DIR / "backtest" / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"  [WARN] 读取 metrics.json 失败: {e}")

    latest = features[-1] if features else {}
    sig_state = state.get("state", "空仓")

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbols": ["588000"],
        "signals": {
            "588000": {
                "state": sig_state,
                "score": round(float(latest.get("total_score", 0)), 4),
                "date": str(latest.get("date", ""))[:10],
                "factors": {
                    k: round(float(latest.get(k, 0)), 3)
                    for k in ["momentum", "trend", "volume_price", "rsrs"]
                },
                "factor_names": ["动量", "趋势", "量价", "RSRS"],
            }
        },
        "equity": equity[-300:],      # 最近 300 天权益
        "paper_equity": paper_equity[-300:],  # 最近 300 天纸面盘权益（无则空列表）
        "prices": daily[-200:],       # 最近 200 天价格
        "features": features[-60:],   # 最近 60 天特征
        "trades": map_backtest_trades(trades[-15:]),  # 最近 15 笔回测交易（映射为展示 schema）
        "trade_log": trade_log[-30:], # 最近 30 条实盘记录（决策留痕）
        "metrics": metrics,           # 回测绩效指标（sortino/omega/max_dd_duration 等）
    }
    return data


def sanitize(obj):
    """递归清洗：NaN/Infinity → None（JSON 规范要求，浏览器 JSON.parse 会拒绝裸 NaN）"""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def atomic_write(data: dict, out_path: Path) -> bool:
    """原子写 + 校验"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 清洗 NaN（pandas CSV 读入的 float NaN 序列化为裸 NaN，浏览器解析失败）
    data = sanitize(data)

    # 写临时文件
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, allow_nan=False)
    except Exception as e:
        os.unlink(tmp)
        print(f"  [ERR] 写入临时文件失败: {e}")
        return False

    # 校验：JSON 可解析 + 关键字段存在
    try:
        with open(tmp, encoding="utf-8") as f:
            check = json.load(f)
        assert "generated_at" in check, "缺 generated_at"
        assert "symbols" in check, "缺 symbols"
        assert "signals" in check, "缺 signals"
        assert isinstance(check["symbols"], list) and len(check["symbols"]) > 0, "symbols 为空"
    except Exception as e:
        os.unlink(tmp)
        print(f"  [ERR] 校验失败: {e}")
        return False

    # 原子替换
    os.replace(tmp, out_path)
    return True


def main():
    print("  [UI] 生成共享数据 docs/assets/data.json")
    data = build_data()
    ok = atomic_write(data, OUT_PATH)
    if not ok:
        print("  [FAIL] data.json 生成失败")
        sys.exit(1)
    kb = os.path.getsize(OUT_PATH) / 1024
    print(f"  [OK] {OUT_PATH.resolve()} ({kb:.0f}KB, symbols={data['symbols']})")


if __name__ == "__main__":
    main()
