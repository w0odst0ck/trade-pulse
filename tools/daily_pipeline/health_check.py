#!/usr/bin/env python3
"""
health_check.py — trade-pulse 系统健康检查

检查项：
  1. 数据滞后：daily.csv 最新日期 vs 今天（交易日）差距
  2. 特征/信号是否正常：features_cache 最新日期
  3. data.json 生成时间：是否过期（>2 天）
  4. 状态机文件存在
  5. 最近一次 git 提交时间（UI 部署是否在跑）

输出：
  --json：机器可读（供 cron 判断）
  默认：人类可读 + 非零退出码表示有异常

用法：
  python health_check.py                  # 人类可读
  python health_check.py --json           # JSON 输出
"""

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "588000"
DOCS_DIR = PROJECT_ROOT / "docs"

# 交易日历（简化：周末不算交易日）
def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def _latest_date(df: pd.DataFrame, path: Path):
    """安全取最新日期：空表/缺列/解析失败 → 返回 None"""
    if df is None or len(df) == 0:
        return None
    if "date" not in df.columns:
        return None
    try:
        return pd.to_datetime(df["date"]).max()
    except Exception:
        return None


def check_data_staleness() -> dict:
    """检查数据滞后"""
    path = DATA_DIR / "daily.csv"
    if not path.exists():
        return {"ok": False, "msg": "daily.csv 不存在"}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"ok": False, "msg": f"daily.csv 解析失败: {e}"}

    latest = _latest_date(df, path)
    if latest is None:
        return {"ok": False, "msg": "daily.csv 为空或缺少 date 列"}
    latest = latest.date()
    today = date.today()

    # 找最近一个交易日（跳过周末）
    cursor = today
    while is_weekend(cursor):
        cursor = date.fromordinal(cursor.toordinal() - 1)

    lag_days = (cursor - latest).days
    # 周末允许 +2 天滞后（周五数据在周末检查时滞后 2 天正常）
    ok = lag_days <= 2
    return {
        "ok": ok,
        "msg": f"数据最新 {latest}（最近交易日 {cursor}，滞后 {lag_days} 天）",
        "latest": str(latest),
        "lag_days": lag_days,
    }


def check_features() -> dict:
    """检查特征缓存（含滞后检查）"""
    path = DATA_DIR / "features_cache.csv"
    if not path.exists():
        return {"ok": False, "msg": "features_cache.csv 不存在"}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"ok": False, "msg": f"features_cache.csv 解析失败: {e}"}

    latest = _latest_date(df, path)
    if latest is None:
        return {"ok": False, "msg": "features_cache.csv 为空或缺少 date 列"}
    latest = latest.date()

    # 滞后检查：与最近交易日对比，允许 2 天（特征跟着数据走）
    cursor = date.today()
    while is_weekend(cursor):
        cursor = date.fromordinal(cursor.toordinal() - 1)
    lag_days = (cursor - latest).days
    ok = lag_days <= 2
    return {
        "ok": ok,
        "msg": f"特征最新 {latest}（滞后 {lag_days} 天）",
        "latest": str(latest),
        "lag_days": lag_days,
    }


def check_data_json() -> dict:
    """检查 data.json 是否过期"""
    path = DOCS_DIR / "assets" / "data.json"
    if not path.exists():
        return {"ok": False, "msg": "docs/assets/data.json 不存在（UI 未构建）"}
    try:
        with open(path) as f:
            d = json.load(f)
        gen = d.get("generated_at", "")
        gen_dt = datetime.strptime(gen, "%Y-%m-%d %H:%M").date()
        lag = (date.today() - gen_dt).days
        ok = lag <= 2
        return {"ok": ok, "msg": f"data.json 生成于 {gen}（{lag} 天前）", "latest": gen}
    except Exception as e:
        return {"ok": False, "msg": f"data.json 解析失败: {e}"}


def check_state() -> dict:
    """检查状态机"""
    path = DATA_DIR / "state.json"
    if not path.exists():
        return {"ok": False, "msg": "state.json 不存在"}
    with open(path) as f:
        s = json.load(f)
    return {"ok": True, "msg": f"状态机: {s.get('state', '?')}"}


def check_git() -> dict:
    """检查最近 git 提交时间（UI 部署是否在跑）"""
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        last = r.stdout.strip()
        if not last:
            return {"ok": True, "msg": "无 git 记录"}
        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
        hours = (datetime.now() - last_dt).total_seconds() / 3600
        # 工作日 14:25 后应有提交；周末允许 48h
        ok = hours <= 72
        return {"ok": ok, "msg": f"最近提交 {last}（{hours:.0f} 小时前）", "latest": last}
    except Exception as e:
        return {"ok": True, "msg": f"git 检查跳过: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = {
        "data": check_data_staleness(),
        "features": check_features(),
        "data_json": check_data_json(),
        "state": check_state(),
        "git": check_git(),
    }

    if args.json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        print(f"\n🔧 trade-pulse 健康检查 — {date.today()}")
        print("=" * 45)
        for name, c in checks.items():
            mark = "✅" if c["ok"] else "❌"
            print(f"  {mark} [{name:<10}] {c['msg']}")

    # 退出码：有异常返回 1
    failed = [k for k, v in checks.items() if not v["ok"]]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
