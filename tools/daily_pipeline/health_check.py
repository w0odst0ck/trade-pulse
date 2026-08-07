#!/usr/bin/env python3
"""
health_check.py — trade-pulse 系统健康检查

检查项：
  1. 数据滞后：daily.csv 最新日期 vs 今天（交易日）差距
  2. 特征/信号是否正常：features_cache 最新日期
  3. 特征滞后检查：features_cache 最新日期 vs 最近交易日（严格判定）
  4. data.json 生成时间：是否过期（>2 天）
  5. 状态机文件存在
  6. 最近一次 git 提交时间（UI 部署是否在跑）
  7. 五源健康度：tencent / sina / akshare / eastmoney / baostock 各拉一次探活；
     冷却中的源（source_health.json 的 cooldown_until 未到期）跳过探活不算故障，
     全 OK 时一行带过，有 FAIL 才输出每源状态表告警（源故障标注，退出码非 0）

输出：
  --json：机器可读（供 cron 判断）
  默认：人类可读 + 非零退出码表示有异常

用法：
  python health_check.py                  # 人类可读
  python health_check.py --json           # JSON 输出
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data_quality import scan_quality

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
SOURCE_HEALTH_PATH = SCRIPT_DIR / "source_health.json"
DATA_DIR = PROJECT_ROOT / "data" / "588000"
DOCS_DIR = PROJECT_ROOT / "docs"

# 腾讯 fqkline 接口（与 data_provider/tencent.py 同源）
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
TENCENT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

# 交易日历：统一走 trading_calendar（is_trading_day/prev_trading_day，识别节假日）


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

    # 最近交易日（trading_calendar 识别节假日，与 features_staleness 口径一致）
    cursor = _recent_trading_day()

    lag_days = (cursor - latest).days
    # 周末/节假日允许 +2 天滞后（周五数据在周末检查时滞后 2 天正常）
    ok = lag_days <= 2
    return {
        "ok": ok,
        "msg": f"数据最新 {latest}（最近交易日 {cursor}，滞后 {lag_days} 天）",
        "latest": str(latest),
        "lag_days": lag_days,
    }


def _recent_trading_day() -> date:
    """最近一个交易日（复用 trading_calendar 的 is_trading_day/prev_trading_day）"""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from trading_calendar import is_trading_day, prev_trading_day
    today = date.today()
    if is_trading_day(today):
        return today
    return prev_trading_day(today)


def check_features_staleness() -> dict:
    """检查特征缓存滞后：特征末日 < 数据末日 → 滞后

    语义：特征应跟上已到位的数据（数据可能因腾讯定型晚而滞后交易日，
    特征跟随数据即可，无需追赶尚未发布的交易日）。
    """
    feat_path = DATA_DIR / "features_cache.csv"
    data_path = DATA_DIR / "daily.csv"
    if not feat_path.exists():
        return {"ok": False, "msg": "features_cache.csv 不存在"}
    if not data_path.exists():
        return {"ok": False, "msg": "daily.csv 不存在（无法对比）"}

    try:
        feat_df = pd.read_csv(feat_path)
        data_df = pd.read_csv(data_path)
    except Exception as e:
        return {"ok": False, "msg": f"features_cache.csv 解析失败: {e}"}

    feat_latest = _latest_date(feat_df, feat_path)
    data_latest = _latest_date(data_df, data_path)
    if feat_latest is None:
        return {"ok": False, "msg": "features_cache.csv 为空或缺少 date 列"}
    if data_latest is None:
        return {"ok": False, "msg": "daily.csv 为空或缺少 date 列"}
    feat_latest = feat_latest.date()
    data_latest = data_latest.date()

    lag_days = (data_latest - feat_latest).days
    if feat_latest < data_latest:
        return {
            "ok": False,
            "msg": f"特征滞后数据 {lag_days} 天（特征最新 {feat_latest}，数据最新 {data_latest}）",
            "latest": str(feat_latest),
            "data_latest": str(data_latest),
            "lag_days": lag_days,
        }
    return {
        "ok": True,
        "msg": f"特征已跟上数据（特征/数据均 {feat_latest}）",
        "latest": str(feat_latest),
        "data_latest": str(data_latest),
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
    # 用 trading_calendar 识别节假日，与 features_staleness 口径一致
    cursor = _recent_trading_day()
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


def check_data_quality() -> dict:
    """检查存量数据质量：跳空残留 / NaN 残留（与 fetch 时拦截同判据）"""
    path = DATA_DIR / "daily.csv"
    if not path.exists():
        return {"ok": False, "msg": "daily.csv 不存在"}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"ok": False, "msg": f"daily.csv 解析失败: {e}"}

    problems = scan_quality(df, symbol="588000")
    if problems:
        return {"ok": False, "msg": "; ".join(problems)}
    return {"ok": True, "msg": "数据质量正常（无跳空/NaN 残留）"}


def check_state() -> dict:
    """检查状态机"""
    path = DATA_DIR / "state.json"
    if not path.exists():
        return {"ok": False, "msg": "state.json 不存在"}
    try:
        with open(path) as f:
            s = json.load(f)
    except Exception as e:
        return {"ok": False, "msg": f"state.json 解析失败: {e}"}
    if not isinstance(s, dict) or "state" not in s:
        return {"ok": False, "msg": "state.json 结构异常（缺 state 字段）"}
    return {"ok": True, "msg": f"状态机: {s.get('state', '?')}"}


def check_git() -> dict:
    """检查最近 git 提交时间（UI 部署是否在跑）"""
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        if r.returncode != 0:
            # git 命令本身失败（非仓库/权限等）：部署链路异常，如实报告
            return {"ok": False, "msg": f"git 检查失败: {(r.stderr or r.stdout).strip()[:100]}"}
        last = r.stdout.strip()
        if not last:
            return {"ok": True, "msg": "无 git 记录"}
        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
        hours = (datetime.now() - last_dt).total_seconds() / 3600
        # 工作日 14:25 后应有提交；周末允许 48h
        ok = hours <= 72
        return {"ok": ok, "msg": f"最近提交 {last}（{hours:.0f} 小时前）", "latest": last}
    except Exception as e:
        return {"ok": False, "msg": f"git 检查异常: {e}"}


# ── 五源健康度 ─────────────────────────────────────────


def _clear_proxy_env():
    """东财系（AkShare/EastMoney）走代理必失败（mihomo 规则拒绝），
    与 fetch_data.py 一致强制直连"""
    for k in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(k, None)


def _main_symbol() -> str:
    """当前主标的（config.json 的 symbol，与 fetch_data 拉取对象一致）"""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f).get("symbol", "588000")
    except Exception:
        return "588000"


def _recent_start(days: int = 7) -> str:
    """最近 N 天起点（YYYY-MM-DD），探活只需近期 1 根"""
    return (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def _check_tencent(symbol: str) -> dict:
    """腾讯源：web.ifzq.gtimg.cn fqkline 拉 1 根"""
    _clear_proxy_env()
    try:
        import requests
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from fetch_data import resolve_tencent_symbol
        tsym = resolve_tencent_symbol(symbol)  # 588000 -> sh588000（config['markets'] 映射）
        end = date.today().strftime("%Y-%m-%d")
        start = _recent_start()
        url = (f"{TENCENT_KLINE_URL}?param={tsym},day,{start},{end},1,qfq")
        resp = requests.get(url, headers=TENCENT_HEADERS, timeout=10)
        data = resp.json()
        node = (data.get("data") or {}).get(tsym) or {}
        klines = node.get("qfqday") or node.get("day") or []
        if klines:
            return {"ok": True, "msg": f"fqkline 正常（最近 {klines[-1][0]}）"}
        return {"ok": False, "msg": "fqkline 返回空（接口可达但无数据）"}
    except Exception as e:
        return {"ok": False, "msg": f"请求失败: {e}"}


def _check_akshare(symbol: str) -> dict:
    """AkShare 源（东财系）：被动观察，不主动请求（东财 IP 风控重点）"""
    return _observe_source(symbol, "akshare")


def _check_eastmoney(symbol: str) -> dict:
    """EastMoney 源（东财系）：被动观察，不主动请求（东财 IP 风控重点）"""
    return _observe_source(symbol, "eastmoney")


def _observe_source(symbol: str, src: str) -> dict:
    """被动观察东财系源：读 source_health.json 展示最近状态，零主动请求

    东财是 IP 风控重点：主动探活可能重新触发风控（退烧了又去撩病毒）。
    状态完全由 fetch_data 实际使用结果驱动（fetch 成功/失败时写入）。
    """
    try:
        with open(SOURCE_HEALTH_PATH, encoding="utf-8") as f:
            health = json.load(f)
    except Exception:
        return {"ok": False, "msg": "观察源不可用：source_health.json 缺失/损坏"}
    entry = health.get(symbol, {}).get(src, {})
    if not entry:
        return {"ok": True, "msg": "未使用（fetch_data 尚未请求过此源）"}
    now_ts = time.time()
    cooldown_until = entry.get("cooldown_until", 0) or 0
    if cooldown_until > now_ts:
        remain_h = (cooldown_until - now_ts) / 3600
        return {"ok": True, "msg": f"SKIP(cooldown) 冷却中（剩 {remain_h:.1f}h，预期状态）"}
    fails = entry.get("consecutive_failures", 0)
    last_date = entry.get("last_max_date") or "无"
    if fails > 0:
        return {"ok": False, "msg": f"最近失败 {fails} 次（观察中，最近数据 {last_date}）"}
    return {"ok": True, "msg": f"正常（观察，最近数据 {last_date}）"}


def _check_sina(symbol: str) -> dict:
    """新浪源：SinaProvider 拉最近数据（与 fetch_data 同实现：带市场前缀）"""
    _clear_proxy_env()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from data_provider.sina import SinaProvider
        from fetch_data import resolve_tencent_symbol
        call_symbol = resolve_tencent_symbol(symbol)  # 588000 -> sh588000（config['markets'] 映射）
        df = SinaProvider().fetch_daily(call_symbol, _recent_start())
        if df is not None and len(df) > 0:
            latest = df["date"].max()
            latest = latest.date() if hasattr(latest, "date") else latest
            return {"ok": True, "msg": f"新浪接口正常（最新 {latest}）"}
        return {"ok": False, "msg": "新浪接口返回空数据"}
    except Exception as e:
        return {"ok": False, "msg": f"源故障: {e}"}


def _check_baostock(symbol: str) -> dict:
    """BaoStock 源：login 测试（登录式 API，login 成功即链路可用）"""
    _clear_proxy_env()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from data_provider.baostock import BaoStockProvider
        provider = BaoStockProvider()
        bs = provider._login()
        bs.logout()
        return {"ok": True, "msg": "login/logout 正常"}
    except Exception as e:
        return {"ok": False, "msg": f"login 失败: {e}"}


def _load_cooldown() -> tuple:
    """读取 source_health.json，返回 (冷却中的源名集合, 警告信息)

    冷却判定与 fetch_data.py 一致：entry["cooldown_until"] > now → 冷却中。
    遍历所有 symbol：任一 symbol 下某源处于冷却即认为该源冷却中（跳过探活）。
    文件缺失/损坏 → 返回 (空集, 警告)，由调用方降级为全部真实探活（不静默跳过）。
    """
    path = SCRIPT_DIR / "source_health.json"
    if not path.exists():
        warn = "source_health.json 缺失，冷却跳过失效，降级为全部真实探活"
        return set(), warn
    try:
        with open(path, encoding="utf-8") as f:
            health = json.load(f)
    except Exception as e:
        warn = f"source_health.json 解析失败（{e}），冷却跳过失效，降级为全部真实探活"
        return set(), warn

    now_ts = time.time()
    cooldown = set()
    if not isinstance(health, dict):
        warn = "source_health.json 结构异常，冷却跳过失效，降级为全部真实探活"
        return set(), warn
    for symbol_health in health.values():
        if not isinstance(symbol_health, dict):
            continue
        for src, entry in symbol_health.items():
            if isinstance(entry, dict) and entry.get("cooldown_until", 0) > now_ts:
                cooldown.add(src)
    return cooldown, ""


def check_provider_health() -> dict:
    """五源健康度：{ok, msg, sources: {name: {ok, msg}}}

    每源一次探活请求；冷却中的源（source_health.json 的 cooldown_until 未到期）
    不真实请求，直接标 ok=True + SKIP(cooldown)（冷却中属预期状态，非故障）；
    source_health.json 缺失/损坏时降级为全部真实探活并输出 WARN。
    全部 OK 时 msg 汇总一行，有 FAIL 时 msg 列出异常源，
    明细在各源的 sources[name] 中（人类可读输出时展开为状态表）。
    """
    symbol = _main_symbol()
    cooldown_sources, warn = _load_cooldown()
    if warn:
        print(f"  [WARN] {warn}", file=sys.stderr)

    checkers = {
        "tencent": _check_tencent,
        "sina": _check_sina,
        "akshare": _check_akshare,
        "eastmoney": _check_eastmoney,
        "baostock": _check_baostock,
    }
    sources = {}
    for name, fn in checkers.items():
        if name in cooldown_sources:
            sources[name] = {"ok": True, "msg": "SKIP(cooldown)，跳过探活"}
        else:
            sources[name] = fn(symbol)

    failed = [n for n, s in sources.items() if not s["ok"]]
    if not failed:
        msg = "五源健康（tencent/sina/akshare/eastmoney/baostock）"
    else:
        msg = f"{len(failed)} 个数据源异常: {', '.join(failed)}"
    return {"ok": not failed, "msg": msg, "sources": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = {
        "data": check_data_staleness(),
        "features": check_features(),
        "features_staleness": check_features_staleness(),
        "data_json": check_data_json(),
        "state": check_state(),
        "quality": check_data_quality(),
        "git": check_git(),
        "providers": check_provider_health(),
    }

    if args.json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        print(f"\n🔧 trade-pulse 健康检查 — {date.today()}")
        print("=" * 45)
        for name, c in checks.items():
            mark = "✅" if c["ok"] else "❌"
            print(f"  {mark} [{name:<10}] {c['msg']}")
            # 四源有 FAIL 时才展开每源状态表（全 OK 时上面一行带过，保持静默）
            if name == "providers" and not c["ok"]:
                for sname, s in c.get("sources", {}).items():
                    smark = "✅" if s["ok"] else "❌"
                    print(f"      {smark} {sname:<9} {s['msg']}")

    # 退出码：有异常返回 1
    failed = [k for k, v in checks.items() if not v["ok"]]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
