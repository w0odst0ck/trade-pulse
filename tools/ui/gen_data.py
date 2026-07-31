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


def build_data() -> dict:
    import pandas as pd

    state = {}
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)

    daily = load_csv(DATA_DIR / "daily.csv")
    features = load_csv(DATA_DIR / "features_cache.csv")
    equity = load_csv(DATA_DIR / "backtest" / "equity_curve.csv")
    trades = load_csv(DATA_DIR / "backtest" / "trades.csv")

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
        "prices": daily[-200:],       # 最近 200 天价格
        "features": features[-60:],   # 最近 60 天特征
        "trades": trades[-15:],       # 最近 15 笔交易
    }
    return data


def atomic_write(data: dict, out_path: Path) -> bool:
    """原子写 + 校验"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 写临时文件
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
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
