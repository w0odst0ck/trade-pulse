#!/usr/bin/env python3
"""
realtime_quote.py — 盘中实时行情（方案 B 核心模块）

用途：14:25 盘中预览 / 14:50 尾盘确认时，拉取当日实时 bar
      （今开/最高/最低/现价/累计量/累计额），构造当日未收盘 K 线，
      参与特征计算，消除「信号落后一天」的结构性滞后。

数据源（免费、无 key、互不依赖）：
  - 腾讯  qt.gtimg.cn/q=shXXXXXX   （GBK，成交量=手，与 daily.csv 主源口径一致）
  - 新浪  hq.sinajs.cn/list=shXXXXXX（GBK，ETF/股票=股需 ÷100，指数=手不换算；需 Referer）

稳定性设计（实盘兜底）：
  - 双源互备：腾讯主 + 新浪备；双源均失败 → 返回 None（上层回退昨日收盘信号）
  - 交叉验证：双源 close 相对差 < 0.5% 判一致（用腾讯）；> 2% 判冲突（取腾讯+告警）
  - sanity check：价格 > 0、high >= max(open,close)、low <= min(open,close)、
    close/high/low 与昨收相对偏差 < 20%（防脏数据/接口异常值）
  - 时段守卫：仅交易日 09:30-11:30 / 13:00-15:00 返回实时数据，其余返回 None
  - 幂等：每次调用独立拉取，不落盘、不污染 daily.csv / features_cache
"""

import re
from datetime import datetime, date as date_cls, time as dtime
from pathlib import Path
from typing import Dict, Optional

import requests

from trading_calendar import is_trading_day

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

# config['markets'] 缓存（热路径：14:25/14:50 每次调用都解析前缀，避免重复读盘）
_MARKETS_CACHE = None
_MARKETS_LOADED = False


def _load_markets() -> dict:
    """读取 config['markets'] 映射，带缓存；加载失败不缓存（下次调用重试）。"""
    global _MARKETS_CACHE, _MARKETS_LOADED
    if _MARKETS_LOADED:
        return _MARKETS_CACHE
    try:
        import json
        with open(CONFIG_PATH) as f:
            _MARKETS_CACHE = (json.load(f).get("markets") or {})
        _MARKETS_LOADED = True
    except (OSError, json.JSONDecodeError):
        # 失败不置 _MARKETS_LOADED：config 恢复后自动重试，避免永久用空映射
        _MARKETS_CACHE = {}
    return _MARKETS_CACHE


def _resolve_prefix(symbol: str) -> str:
    """解析市场前缀：优先 config['markets'] 显式映射（000688=sh 科创50指数），
    缺失时按代码首字符兜底（6/5/9→sh，0/1/3→sz，8/4→bj）。

    与 fetch_data.resolve_tencent_symbol 同源逻辑：纯首字符无法区分
    「000xxx 上证指数」与「000xxx 深市股票」，必须显式映射。
    """
    if symbol[:2].lower() in ("sh", "sz", "bj"):
        return symbol[:2].lower()
    m = _load_markets().get(symbol)
    if m:
        return m
    first = symbol[0]
    if first in "659":
        return "sh"
    if first in "013":
        return "sz"
    if first in "84":
        return "bj"
    return "sh"

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TENCENT_URL = "http://qt.gtimg.cn/q={symbol}"
SINA_URL = "https://hq.sinajs.cn/list={symbol}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
SINA_HEADERS = {**HEADERS, "Referer": "https://finance.sina.com.cn/"}

TIMEOUT_SEC = 8

# 双源一致性判定（close 相对差）
CLOSE_CONSISTENT_PCT = 0.005    # < 0.5% 一致
CLOSE_CONFLICT_PCT = 0.02       # > 2% 严重冲突（取腾讯 + 告警）

# sanity check：与昨收最大相对偏差（防接口脏数据/异常值）
MAX_PRICE_DEV_PCT = 0.20

# 腾讯实时接口字段索引（按 ~ 分割，0-based；实测验证过）
TX_IDX = {
    "name": 1, "code": 2, "price": 3, "prev_close": 4, "open": 5,
    "volume": 6, "ts": 30, "change": 31, "change_pct": 32,
    "high": 33, "low": 34, "quote": 35,  # quote = 价/量/额（斜杠分隔）
}

# 新浪实时接口字段索引（按 , 分割，0-based）
SINA_IDX = {
    "name": 0, "open": 1, "prev_close": 2, "price": 3, "high": 4,
    "low": 5, "volume": 8, "amount": 9, "date": 30, "time": 31,
}


# ---------------------------------------------------------------------------
# 时段守卫
# ---------------------------------------------------------------------------

def is_market_open(now: Optional[datetime] = None) -> bool:
    """交易日 + 连续竞价时段（09:30-11:30 / 13:00-15:00）"""
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return False
    t = now.time()
    return dtime(9, 30) <= t <= dtime(11, 30) or dtime(13, 0) <= t <= dtime(15, 0)


# ---------------------------------------------------------------------------
# 单源拉取
# ---------------------------------------------------------------------------

def _fetch_tencent(symbol: str) -> Optional[Dict]:
    """腾讯实时行情 → bar dict（成交量单位=手，与 daily.csv 主源口径一致）"""
    # 市场前缀：优先 config['markets'] 显式映射，缺失按首字符兜底
    if symbol[:2].lower() not in ("sh", "sz", "bj"):
        symbol = f"{_resolve_prefix(symbol)}{symbol}"
    url = TENCENT_URL.format(symbol=symbol)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        resp.encoding = "gbk"
        text = resp.text
    except Exception as e:
        print(f"  [RT] 腾讯实时请求失败: {type(e).__name__}: {e}")
        return None

    m = re.search(r'="(.*)"', text)
    if not m:
        return None
    parts = m.group(1).split("~")
    if len(parts) < 36:
        return None
    try:
        price = float(parts[TX_IDX["price"]])
        prev_close = float(parts[TX_IDX["prev_close"]])
        open_ = float(parts[TX_IDX["open"]])
        high = float(parts[TX_IDX["high"]])
        low = float(parts[TX_IDX["low"]])
        volume = float(parts[TX_IDX["volume"]])
        ts_raw = parts[TX_IDX["ts"]]
        quote = parts[TX_IDX["quote"]]
    except (ValueError, IndexError):
        return None

    # 未开盘/停牌：price<=0 或时间戳非当日 → 无效
    if price <= 0 or open_ <= 0:
        return None

    # 时间戳 YYYYMMDDHHMMSS → 日期
    bar_date = date_cls.today()
    if len(ts_raw) >= 8 and ts_raw.isdigit():
        try:
            bar_date = datetime.strptime(ts_raw[:8], "%Y%m%d").date()
        except ValueError:
            pass

    # 成交额：quote 字段 "price/volume/amount"（amount 单位=元）
    # 解析失败置 NaN + 告警（amount 仅展示用不参与因子，但避免静默 0 误导）
    amount = float('nan')
    q_parts = quote.split("/") if quote else []
    if len(q_parts) >= 3:
        try:
            amount = float(q_parts[2])
        except ValueError:
            print(f"  [WARN] 腾讯 quote 格式异常，amount 置 NaN: {quote}")
    else:
        print(f"  [WARN] 腾讯 quote 字段缺失，amount 置 NaN: {quote!r}")

    return {
        "date": bar_date, "open": open_, "high": high, "low": low,
        "close": price, "volume": volume, "amount": amount,
        "prev_close": prev_close, "source": "tencent",
    }


def _fetch_sina(symbol: str) -> Optional[Dict]:
    """新浪实时行情 → bar dict

    volume 单位处理（实测验证 2026-08-11）：
      - ETF/股票（5/1 开头）：新浪返回=股，÷100 对齐腾讯手口径
      - 指数（000688 等）：新浪返回=手（与腾讯一致），不换算
    错误换算会导致基准 volume 差 100 倍（静默污染 relative_strength 展示因子）。
    """
    if symbol[:2].lower() not in ("sh", "sz", "bj"):
        symbol = f"{_resolve_prefix(symbol)}{symbol}"
    url = SINA_URL.format(symbol=symbol)
    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        resp.encoding = "gbk"
        text = resp.text
    except Exception as e:
        print(f"  [RT] 新浪实时请求失败: {type(e).__name__}: {e}")
        return None

    m = re.search(r'="(.*)"', text)
    if not m:
        return None
    parts = m.group(1).split(",")
    if len(parts) < 32:
        return None
    try:
        price = float(parts[SINA_IDX["price"]])
        prev_close = float(parts[SINA_IDX["prev_close"]])
        open_ = float(parts[SINA_IDX["open"]])
        high = float(parts[SINA_IDX["high"]])
        low = float(parts[SINA_IDX["low"]])
        volume = float(parts[SINA_IDX["volume"]])
        amount = float(parts[SINA_IDX["amount"]])
    except (ValueError, IndexError):
        return None

    if price <= 0 or open_ <= 0:
        return None

    # 成交量单位：ETF/股票=股（÷100 对齐手），指数=手（不换算）——实测验证
    code = symbol[2:] if symbol[:2].lower() in ("sh", "sz", "bj") else symbol
    if code[:1] in ("5", "1"):
        volume = volume / 100.0

    bar_date = date_cls.today()
    d_raw = parts[SINA_IDX["date"]]
    if d_raw:
        try:
            bar_date = datetime.strptime(d_raw.strip(), "%Y-%m-%d").date()
        except ValueError:
            pass

    return {
        "date": bar_date, "open": open_, "high": high, "low": low,
        "close": price, "volume": volume, "amount": amount,
        "prev_close": prev_close, "source": "sina",
    }


# ---------------------------------------------------------------------------
# sanity check
# ---------------------------------------------------------------------------

def sanity_check(bar: Dict, prev_close: Optional[float] = None) -> tuple:
    """数据合理性校验 → (ok: bool, issues: list[str])

    校验项：
      - 价格字段 > 0
      - high >= max(open, close)、low <= min(open, close)、high >= low
      - volume > 0
      - 与昨收相对偏差 < 20%（prev_close 传入时；close/high/low 都查）
    """
    issues: list = []
    for f in ("open", "high", "low", "close"):
        v = bar.get(f)
        if v is None or v != v or v <= 0:  # 含 NaN 检查
            issues.append(f"{f} 非法: {v}")

    if not issues:
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        if h < max(o, c) - 1e-9:
            issues.append(f"high {h} < max(open {o}, close {c})")
        if l > min(o, c) + 1e-9:
            issues.append(f"low {l} > min(open {o}, close {c})")
        if h < l - 1e-9:
            issues.append(f"high {h} < low {l}")

    vol = bar.get("volume")
    if vol is None or vol != vol or vol <= 0:
        issues.append(f"volume 非法: {vol}")

    if prev_close and prev_close > 0:
        for f in ("close", "high", "low"):
            v = bar.get(f)
            if v and abs(v - prev_close) / prev_close > MAX_PRICE_DEV_PCT:
                issues.append(
                    f"{f} {v} 与昨收 {prev_close} 偏差 "
                    f"{abs(v - prev_close) / prev_close * 100:.1f}% > 20%"
                )

    return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# 双源互备 + 交叉验证（对外主入口）
# ---------------------------------------------------------------------------

def fetch_realtime_bar(symbol: str, prev_close: Optional[float] = None,
                       now: Optional[datetime] = None,
                       _fetch_fns=None) -> Optional[Dict]:
    """拉取当日实时 bar（腾讯主 + 新浪备，交叉验证）

    返回 bar dict（含 source 字段），全部失败或数据不可信返回 None。
    调用方收到 None 应回退「昨日收盘信号」并标注实时源不可用。

    参数：
      _fetch_fns: 注入拉取函数（测试用）；默认 (tencent, sina) 真实请求
    """
    if not is_market_open(now):
        print("  [RT] 非交易时段，实时行情不可用（回退收盘信号）")
        return None

    fetch_tencent, fetch_sina = _fetch_fns or (_fetch_tencent, _fetch_sina)

    # 腾讯主源
    bar_tx = None
    try:
        bar_tx = fetch_tencent(symbol)
    except Exception as e:
        print(f"  [RT] 腾讯实时异常: {type(e).__name__}: {e}")
    if bar_tx:
        ok, issues = sanity_check(bar_tx, prev_close)
        if not ok:
            print(f"  [RT] 腾讯实时数据不合法: {'; '.join(issues)}")
            bar_tx = None
        elif bar_tx.get('date') != date_cls.today():
            # 实盘安全：接口返回的 bar 日期必须等于今天（交易日）。
            # 若数据源返回昨日/异常日期（脏数据），绝不能当作今日盘中 bar 驱动信号
            print(f"  [RT] 腾讯实时 bar 日期 {bar_tx.get('date')} != 今日 {date_cls.today()}，拒绝")
            bar_tx = None

    # 新浪备用
    bar_sina = None
    try:
        bar_sina = fetch_sina(symbol)
    except Exception as e:
        print(f"  [RT] 新浪实时异常: {type(e).__name__}: {e}")
    if bar_sina:
        ok, issues = sanity_check(bar_sina, prev_close)
        if not ok:
            print(f"  [RT] 新浪实时数据不合法: {'; '.join(issues)}")
            bar_sina = None
        elif bar_sina.get('date') != date_cls.today():
            print(f"  [RT] 新浪实时 bar 日期 {bar_sina.get('date')} != 今日 {date_cls.today()}，拒绝")
            bar_sina = None

    # 双源都失败
    if not bar_tx and not bar_sina:
        print("  [RT] 双源实时行情均不可用（回退收盘信号）")
        return None

    # 双源交叉验证
    if bar_tx and bar_sina and bar_tx["close"] > 0:
        rel = abs(bar_sina["close"] - bar_tx["close"]) / bar_tx["close"]
        if rel > CLOSE_CONFLICT_PCT:
            print(f"  [ALERT] 实时双源冲突: 腾讯 close {bar_tx['close']} vs "
                  f"新浪 {bar_sina['close']}（差 {rel * 100:.1f}%），采用腾讯")
        elif rel > CLOSE_CONSISTENT_PCT:
            print(f"  [WARN] 实时双源 close 差 {rel * 100:.2f}%（<2%），采用腾讯")
        # 一致 → 用腾讯
    if bar_tx:
        return bar_tx
    return bar_sina


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="盘中实时行情自测")
    parser.add_argument("symbol", nargs="?", default="588000")
    parser.add_argument("--prev-close", type=float, default=None)
    args = parser.parse_args()

    bar = fetch_realtime_bar(args.symbol, prev_close=args.prev_close)
    if bar is None:
        print("  [RESULT] None（实时源不可用）")
        return 1
    print(f"  [RESULT] {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
