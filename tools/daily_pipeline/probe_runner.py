#!/usr/bin/env python3
"""probe_runner.py — 实盘数据源探测执行器（纯 shell 调用的轻量探针）

5 个关键时点探测（probe.sh 调用）：
  09:05  日线源（腾讯/新浪）+ 实时源 → 预期 max_date ≥ 最近交易日（数据可得性）
  14:20  实时源重点（qt.gtimg/hq.sinajs）→ 保护 14:25 盘中预览
  14:45  实时源重点 → 保护 14:50 尾盘确认（决策点，最高优先）
  15:25  日线源 → 预判 15:30 收盘增量
  16:25  日线源 → 预判 16:30 收盘补齐

设计：
  - 探测只写 source_health.json 的 probe 字段（不动 fetch 的 cooldown/last_max_date）
  - 首次失败重试 1 次（间隔 2s）→ 区分偶发抖动 vs 真实故障
  - 探测结果不进冷却（冷却由 fetch_data 实际拉取结果驱动）
  - 数据可得性：09:05 校验返回最新日期 ≥ 预期（最近交易日）；
    15:25/16:25 只测接口可达 + 日期不倒退（今日未发布是常态，不算故障）
  - 非交易日跳过（trading_calendar）
"""

import argparse
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from trading_calendar import is_trading_day, prev_trading_day

PROBE_HEALTH_PATH = SCRIPT_DIR / "probe_health.json"     # 探测专用（独立文件零并发冲突）

RETRY_COUNT = 1
RETRY_DELAY_SEC = 2.0
TIMEOUT_SEC = 10


def _now_ts() -> float:
    return time.time()


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _load_health() -> dict:
    if PROBE_HEALTH_PATH.exists():
        try:
            return json.loads(PROBE_HEALTH_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _expected_date(mode: str) -> date:
    """预期数据日期 = 最近已收盘交易日（由探测时点决定，不用墙钟）

    morning（09:05）：今天还没收盘 → 预期 = prev_trading_day(today)
    after_close（15:25/16:25）：预期 = today（但调用方不校验新鲜度，
      今日未发布是常态——腾讯盘后定型晚于 16:30）
    """
    today = date.today()
    if is_trading_day(today):
        return prev_trading_day(today) if mode == "morning" else today
    return prev_trading_day(today)


def probe_source(name: str, fn, expect_fresh: bool, symbol: str, mode: str = "morning") -> dict:
    """探测单源：调用探活函数，失败重试 1 次

    expect_fresh=True：校验返回最新日期 ≥ 预期（数据可得性，09:05 用）
    expect_fresh=False：只测接口可达（15:25/16:25 用，今日未发布是常态）
    """
    for attempt in range(RETRY_COUNT + 1):
        try:
            result = fn(symbol)
            if result.get("ok"):
                # 从 msg 提取最新日期（"fqkline 正常（最近 2026-08-11）"）
                max_date = None
                m = re.search(r"(\d{4}-\d{2}-\d{2})", result.get("msg", ""))
                if m:
                    max_date = m.group(1)
                if expect_fresh:
                    # 数据可得性校验：新鲜度正是 morning 探测的目的，
                    # 缺日期必须显式失败（不能静默通过——msg 格式漂移会漏报陈旧）
                    expected = _expected_date(mode)
                    if not max_date:
                        return {"ok": False, "name": name, "max_date": None,
                                "msg": "数据陈旧校验失败：无法从响应提取最新日期"}
                    try:
                        md = date.fromisoformat(max_date)
                        if md < expected:
                            return {"ok": False, "name": name, "max_date": max_date,
                                    "msg": f"数据陈旧：最新 {max_date} < 预期 {expected}"}
                    except ValueError:
                        return {"ok": False, "name": name, "max_date": max_date,
                                "msg": f"数据陈旧校验失败：日期格式异常 {max_date}"}
                return {"ok": True, "name": name, "max_date": max_date, "msg": result.get("msg", "")}
            # 失败：重试
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_SEC)
            else:
                return {"ok": False, "name": name, "msg": result.get("msg", "探活失败")}
        except Exception as e:
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_SEC)
            else:
                return {"ok": False, "name": name, "msg": f"{type(e).__name__}: {e}"}
    return {"ok": False, "name": name, "msg": "探活失败"}


def run_probe(symbol: str, mode: str) -> dict:
    """执行探测，写 source_health.json probe 字段

    mode: morning(09:05) / intraday(14:20,14:45) / after_close(15:25,16:25)
    """
    from health_check import (
        _check_tencent, _check_sina,
        _check_realtime_tencent, _check_realtime_sina,
    )

    results = {}
    if mode in ("morning", "after_close"):
        # 日线源 + 实时源
        expect_fresh = (mode == "morning")
        results["tencent"] = probe_source("tencent", _check_tencent, expect_fresh, symbol, mode)
        results["sina"] = probe_source("sina", _check_sina, expect_fresh, symbol, mode)
        results["realtime_tencent"] = probe_source("realtime_tencent", _check_realtime_tencent, False, symbol, mode)
        results["realtime_sina"] = probe_source("realtime_sina", _check_realtime_sina, False, symbol, mode)
    elif mode == "intraday":
        # 实时源重点（决策保护）+ 日线源快速检查
        results["realtime_tencent"] = probe_source("realtime_tencent", _check_realtime_tencent, False, symbol, mode)
        results["realtime_sina"] = probe_source("realtime_sina", _check_realtime_sina, False, symbol, mode)
        results["tencent"] = probe_source("tencent", _check_tencent, False, symbol, mode)
        results["sina"] = probe_source("sina", _check_sina, False, symbol, mode)

    # 写 probe 专用文件（probe_health.json，与 fetch_data 的 source_health.json 分离，
    # 互不覆盖——probe 只记录探测结果，不动 fetch 的 cooldown/last_max_date）
    health = _load_health()
    sym = health.setdefault(symbol, {})
    for name, r in results.items():
        sym[name] = {
            "ok": r.get("ok", False),
            "max_date": r.get("max_date"),
            "msg": r.get("msg", ""),
            "ts": _now_ts(),
        }
    sym["probe_ts"] = _now_ts()
    sym["probe_mode"] = mode
    _atomic_write(PROBE_HEALTH_PATH, health)

    return results


def main():
    parser = argparse.ArgumentParser(description="数据源探测执行器")
    parser.add_argument("--symbol", default="588000")
    parser.add_argument("--mode", choices=["morning", "intraday", "after_close"], required=True)
    args = parser.parse_args()

    # 非交易日跳过（cron 1-5 也会撞法定假日）
    if not is_trading_day(date.today()):
        print("NO_REPLY")
        return 0

    results = run_probe(args.symbol, args.mode)

    # 输出汇总（probe.sh 解析判断告警）
    failed = [name for name, r in results.items() if not r.get("ok")]
    print(f"probe {args.mode} {args.symbol}: ok={len(results)-len(failed)}/{len(results)}")
    for name, r in results.items():
        status = "OK " if r.get("ok") else "FAIL"
        detail = r.get("msg", "")
        max_d = r.get("max_date") or ""
        print(f"  [{status}] {name}: {detail} {max_d}")
    if failed:
        print("PROBE_FAIL: " + ",".join(failed))
        return 1
    print("PROBE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
