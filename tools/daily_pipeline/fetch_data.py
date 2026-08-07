#!/usr/bin/env python3
"""
fetch_data.py — 588000 日线数据拉取（Provider 版本）

通过 DataProvider 抽象接口获取数据，支持自动降级。

用法：
  python fetch_data.py                    # 增量更新
  python fetch_data.py --force            # 强制全量拉取
  python fetch_data.py --start 2024-01-01 # 指定起始日期
  python fetch_data.py --provider akshare # 指定数据源
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data_quality import MAX_GAP_DAYS, quality_gate
from trading_calendar import is_trading_day, prev_trading_day

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

# 导入 Provider
import sys
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from data_provider import AkShareProvider, EastMoneyProvider, BaoStockProvider, TencentProvider
from data_provider.sina import SinaProvider

# Provider 固定 fallback 顺序（单源模式：取除主源外的其余源按序降级）
PROVIDER_ORDER = ["akshare", "tencent", "eastmoney", "baostock"]

# 多源模式默认配置（config.json 未显式配置时的兜底）
DEFAULT_SOURCES = ["tencent", "sina", "baostock", "akshare", "eastmoney"]
DEFAULT_FAST_SOURCES = ["tencent", "sina"]

# 数据源健康记录（节流/冷却），按 symbol 分源记录
SOURCE_HEALTH_PATH = SCRIPT_DIR / "source_health.json"

# 多源并行每源超时（秒）
SOURCE_TIMEOUT_SEC = 15

# 交叉验证阈值
CLOSE_CONSISTENT_PCT = 0.001   # close 相对差 < 0.1% 判一致
RET_CONSISTENT_PCT = 0.001     # 日收益差 < 0.1% 判一致（对复权基准不敏感）
CLOSE_CONFLICT_PCT = 0.05      # close 相对差 > 5% 判严重冲突（不覆盖本地）


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_data_path(symbol: str) -> Path:
    """数据缓存路径：data/{symbol}/daily.csv"""
    config = load_config()
    data_dir = PROJECT_ROOT / config["data_dir"]
    return data_dir / symbol / "daily.csv"


def load_local(path: Path) -> pd.DataFrame:
    """读取本地缓存数据"""
    if path.exists():
        # symbol 列强制 str：避免 '000688' 被 pandas 推断为 int 688（前导零丢失）
        df = pd.read_csv(path, parse_dates=["date"], dtype={"symbol": str})
        return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame()


def validate_incremental(df: pd.DataFrame, require_date: str = None) -> dict:
    """校验增量结果。返回 {ok: bool, last_date: str, issues: list[str]}

    校验项：
      - df 非空
      - 最新日期 >= require_date（若传入，格式 YYYY-MM-DD）
      - 最新一行 close 非空且 > 0
      - 最新一行 volume 非空且 > 0，且与最近 20 日（不含当日）均量之比在
        [0.1, 10] 区间内（防半日/异常量数据；均量为去首尾 10% 极端值后的
        稳健均值，避免个别巨量日干扰；历史不足 20 日用可用行计算，
        无历史则跳过区间检查）
      - date 无重复、连续无缺失：全量查重复；相邻日期自然日间隔超
        MAX_GAP_DAYS 记不连续（判据与 data_quality.py 风格一致，仅检查
        最近 21 行增量核心窗口，避免 A 股长假间隔误报；存量历史由
        health_check 的 scan_quality 覆盖）
    """
    issues: List[str] = []
    if df is None or len(df) == 0:
        issues.append("数据为空（df 无任何行）")
        return {"ok": False, "last_date": "", "issues": issues}

    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    last_ts = dates.iloc[-1]
    last_date = last_ts.strftime("%Y-%m-%d")

    # 1. require_date 门槛：最新日期必须 >= 要求日期
    if require_date:
        try:
            require_ts = pd.to_datetime(require_date)
        except (ValueError, TypeError):
            issues.append(f"require_date 格式无效: {require_date}（应为 YYYY-MM-DD）")
        else:
            if last_ts < require_ts:
                issues.append(f"最新日期 {last_date} 早于要求日期 {require_date}")

    # 2. 最新一行 close 非空且 > 0
    if "close" not in df.columns:
        issues.append("缺少 close 列")
    else:
        last_close = df["close"].iloc[-1]
        if pd.isna(last_close) or float(last_close) <= 0:
            issues.append(f"最新一行 close 非法: {last_close}（应为 > 0）")

    # 3. 最新一行 volume 非空且 > 0，且与最近 20 日均量之比在 [0.1, 10]
    if "volume" not in df.columns:
        issues.append("缺少 volume 列")
    else:
        last_vol = df["volume"].iloc[-1]
        if pd.isna(last_vol) or float(last_vol) <= 0:
            issues.append(f"最新一行 volume 非法: {last_vol}（应为 > 0）")
        else:
            prior = pd.to_numeric(df["volume"].iloc[:-1], errors="coerce").tail(20).dropna()
            if len(prior) > 0:
                # 去首尾各 10% 极端值后求均量：个别巨量/异常日会拉高均值，
                # 导致正常缩量日被误判（如 2026-08 初 588000 缩量 vs 7 月末巨量）
                p = prior.sort_values()
                k = max(1, len(p) // 10)
                trimmed = p.iloc[k:-k] if len(p) > 2 * k else p
                avg = float(trimmed.mean()) if len(trimmed) > 0 else float(p.mean())
                if avg > 0:
                    ratio = float(last_vol) / avg
                    if not 0.1 <= ratio <= 10.0:
                        issues.append(
                            f"最新 volume {last_vol} 与 20 日均量 {avg:.1f} 比值 "
                            f"{ratio:.2f} 超出 [0.1, 10]"
                        )

    # 4. date 无重复、连续无缺失
    dups = dates[dates.duplicated(keep=False)]
    if len(dups) > 0:
        dup_days = sorted({d.strftime("%Y-%m-%d") for d in dups})
        issues.append(
            f"date 存在 {len(dup_days)} 个重复日: "
            f"{dup_days[:5]}{'...' if len(dup_days) > 5 else ''}"
        )

    recent = dates.tail(21)  # 增量核心窗口：当日 + 20 日均量所需历史
    prev = None
    for d in recent:
        if prev is not None:
            gap = (d - prev).days
            if gap > MAX_GAP_DAYS:
                issues.append(
                    f"日期不连续: {prev.strftime('%Y-%m-%d')} 与 {d.strftime('%Y-%m-%d')} "
                    f"间隔 {gap} 天 > {MAX_GAP_DAYS}"
                )
        prev = d

    return {"ok": len(issues) == 0, "last_date": last_date, "issues": issues}


def get_provider(name: str = "akshare"):
    """获取 Provider 实例"""
    providers = {
        "akshare": AkShareProvider(),
        "tencent": TencentProvider(),
        "eastmoney": EastMoneyProvider(),
        "baostock": BaoStockProvider(),
        "sina": SinaProvider(),
    }
    return providers.get(name, providers["akshare"])


def resolve_tencent_symbol(symbol: str) -> str:
    """按 config['markets'] 将 symbol 映射为腾讯带前缀形式（如 588000 -> sh588000）

    纯代码前缀规则无法区分「000xxx 上证指数」与「000xxx 深市股票」
    （如 000688 既可能是 sh000688 科创50指数，也可能是 sz000688 深市个股），
    必须显式指定市场。无映射时原样返回，由 TencentProvider._market_prefix 兜底。
    """
    market = (load_config().get("markets") or {}).get(symbol)
    return f"{market}{symbol}" if market else symbol


# ---------------------------------------------------------------------------
# 多源调度（S2 数据源稳定性方案）
# ---------------------------------------------------------------------------

def compute_target_date(now: datetime) -> str:
    """计算本次拉取的目标日期（命中判定基准），格式 YYYY-MM-DD

    - now < 15:00 → 最近交易日的前一交易日（盘中防护：不拿当日半日 bar）
    - now >= 15:00 → 最近交易日（今天若是交易日则今天，否则回退）

    最近交易日判定复用 trading_calendar（is_trading_day/prev_trading_day）。
    """
    today = now.date()
    if now.time() < dtime(15, 0):
        return prev_trading_day(today).strftime("%Y-%m-%d")
    if is_trading_day(today):
        return today.strftime("%Y-%m-%d")
    return prev_trading_day(today).strftime("%Y-%m-%d")


def load_source_health() -> dict:
    """读取 source_health.json（节流/冷却状态），损坏或缺失时返回空 dict"""
    if SOURCE_HEALTH_PATH.exists():
        try:
            with open(SOURCE_HEALTH_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_source_health(health: dict) -> None:
    try:
        with open(SOURCE_HEALTH_PATH, "w") as f:
            json.dump(health, f, indent=1)
    except OSError as e:
        print(f"  [WARN] 写入 source_health 失败: {e}")


def _update_health(health: dict, symbol: str, src: str, status: str, max_date: Optional[str],
                   cooldown_cfg: dict, now_ts: float) -> None:
    """主线程串行更新单源健康记录（worker 不写 health，避免并发竞态）

    status 语义：
      HIT/MISS（请求成功，无论是否命中）→ 清零失败计数、更新 last_max_date；
        请求成功但未命中不计冷却（数据时序问题非源健康问题）
      FAIL（请求异常）→ consecutive_failures+1，达阈值进入冷却（cooldown_until）
      SKIP（节流/冷却跳过，未发请求）→ 不记录本次，保持上次状态
    """
    if status == "SKIP":
        return
    entry = health.setdefault(symbol, {}).setdefault(src, {
        "last_fetch_ts": 0, "last_max_date": None,
        "consecutive_failures": 0, "cooldown_until": 0,
    })
    # 冷却到期 → 清零失败计数，恢复尝试
    if entry.get("cooldown_until", 0) and entry["cooldown_until"] <= now_ts:
        entry["consecutive_failures"] = 0
        entry["cooldown_until"] = 0
    entry["last_fetch_ts"] = now_ts
    if status == "FAIL":
        entry["consecutive_failures"] = entry.get("consecutive_failures", 0) + 1
        if entry["consecutive_failures"] >= cooldown_cfg.get("max_failures", 3):
            entry["cooldown_until"] = now_ts + cooldown_cfg.get("cooldown_hours", 24) * 3600
    else:
        entry["consecutive_failures"] = 0
        entry["cooldown_until"] = 0
        if max_date:
            entry["last_max_date"] = max_date


def _run_source(src: str, symbol: str, start: str, target_date: str, local_df: pd.DataFrame,
                health: dict, cooldown_cfg: dict, config: dict):
    """单次调用一个数据源（无重试循环——多源冗余替代重试；单源模式保留原重试）

    只读 health（节流/冷却判定），不写——健康状态由主线程在 _run_phase 统一提交，
    避免 worker 线程并发写与 save_source_health 序列化竞争。

    返回 (status, df, detail, max_date)：
      status:  HIT（命中）/ MISS（数据未到位）/ FAIL（请求异常）/ SKIP（冷却或节流）
      df:      命中时的标准化 DataFrame，否则 None
      detail:  机器可读状态描述（HIT=max_date；MISS/SKIP/FAIL 为原因）
      max_date: 请求成功时源数据的实际最新日期（None 表示无数据），用于节流记录
    """
    entry = health.get(symbol, {}).get(src) or {}
    now_ts = time.time()

    # 冷却：cooldown_until 未到期 → 直接跳过（到期后的清零由主线程 _update_health 处理）
    if entry.get("cooldown_until", 0) > now_ts:
        return "SKIP", None, "cooldown", None

    # 节流：skip_window_min 内刚拉过且上次已覆盖目标日期 → 跳过（结果不会变）
    skip_win = cooldown_cfg.get("skip_window_min", 30) * 60
    last_ts = entry.get("last_fetch_ts", 0)
    last_max = entry.get("last_max_date")
    if last_ts and (now_ts - last_ts) < skip_win and last_max and last_max >= target_date:
        return "SKIP", None, "throttle", last_max

    try:
        provider = get_provider(src)
        # tencent/sina 需带市场前缀（config['markets'] 映射，无映射按代码首字符推断）
        call_symbol = resolve_tencent_symbol(symbol) if src in ("tencent", "sina") else symbol
        df = provider.fetch_daily(call_symbol, start)
        if df is None or len(df) == 0:
            return "MISS", None, "empty", None
        df = provider.normalize(df, symbol)
        df, q_warnings = quality_gate(df, local_df, symbol)
        for w in q_warnings:
            print(f"  ⚠️ [QUALITY] {w}")
        if df is None or len(df) == 0:
            return "MISS", None, "quality-gate", None
        max_date = df["date"].max().strftime("%Y-%m-%d")
        if max_date >= target_date:
            return "HIT", df, max_date, max_date
        return "MISS", None, f"no-{target_date}-bar", max_date
    except Exception as e:
        return "FAIL", None, type(e).__name__, None


def _run_phase(sources: List[str], symbol: str, start: str, target_date: str,
               local_df: pd.DataFrame, health: dict, cooldown_cfg: dict, config: dict) -> dict:
    """并行拉取一批源（每源独立超时 SOURCE_TIMEOUT_SEC）

    返回 {src: (status, df, detail, max_date)}；超时未完成的源标记 FAIL(timeout)。
    所有健康状态在 worker 全部返回后由主线程串行提交（_update_health），
    阶段 2 的节流/冷却判定基于阶段 1 已提交的最新状态。
    """
    results: dict = {}
    if not sources:
        return results
    executor = ThreadPoolExecutor(max_workers=len(sources))
    try:
        futures = {
            executor.submit(_run_source, s, symbol, start, target_date, local_df,
                            health, cooldown_cfg, config): s
            for s in sources
        }
        try:
            for fut in as_completed(futures, timeout=SOURCE_TIMEOUT_SEC):
                src = futures[fut]
                try:
                    results[src] = fut.result()
                except Exception as e:
                    results[src] = ("FAIL", None, type(e).__name__, None)
        except TimeoutError:
            pass  # 未完成的 future 统一标记 timeout
        for fut, src in futures.items():
            if src not in results:
                results[src] = ("FAIL", None, "timeout", None)
    finally:
        executor.shutdown(wait=False)  # 不阻塞等挂起线程（CLI 短进程可接受）
    # 主线程统一提交健康状态
    for src, (status, _, _, max_date) in results.items():
        _update_health(health, symbol, src, status, max_date, cooldown_cfg, time.time())
    return results


def _pair_consistency(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """两源 close 对齐比较（inner join by date）

    一致判定：close 相对差 < 0.1% 且日收益差 < 0.1%（收益差对复权基准不敏感）。
    返回 True（一致）/ False（不一致）/ None（无可对齐日期，无法判定）/ "conflict"（>5%）。
    """
    ac = df_a.set_index("date")["close"]
    bc = df_b.set_index("date")["close"]
    joined = pd.concat([ac, bc], axis=1, join="inner").dropna()
    if len(joined) == 0:
        return None
    rel = (joined.iloc[:, 1] - joined.iloc[:, 0]).abs() / joined.iloc[:, 0].abs()
    max_rel = float(rel.max())
    if max_rel > CLOSE_CONFLICT_PCT:
        return "conflict"
    ar, br = ac.pct_change(), bc.pct_change()
    jr = pd.concat([ar, br], axis=1, join="inner").dropna()
    max_ret = float((jr.iloc[:, 1] - jr.iloc[:, 0]).abs().max()) if len(jr) > 0 else 0.0
    return max_rel < CLOSE_CONSISTENT_PCT and max_ret < RET_CONSISTENT_PCT


def _adjudicate(hits: Dict[str, pd.DataFrame], priority: List[str], symbol: str):
    """多源命中交叉验证与裁决

    - 1 源命中：直接采用
    - ≥2 源命中：先做冲突检测——任一源对 close 相对差 > 5% → 严重冲突
      （conflict=True）：调用方不得覆盖本地数据并报警
    - 按 close 聚类（以每源为锚求一致集）取多数一致组：
      - 全部一致 → 用最高优先级源（priority[0]），记录 n_agree
      - 有不一致 → 用多数一致组内最高优先级源；无严格多数（平局）→
        用最高优先级源所在组 + [ALERT] 报警

    Returns (winner_df, info)；info["conflict"]=True 时 winner_df=None，调用方不得入库。
    """
    base_name = priority[0]
    base = hits[base_name].copy().sort_values("date").reset_index(drop=True)
    n = len(hits)
    if n == 1:
        return base, {"n_sources": 1, "n_agree": 1, "winner": base_name,
                      "conflict": False, "alerts": []}

    # 冲突检测：任一源对 close 相对差 >5% → 数据不可信，不覆盖本地
    for i in range(n):
        for j in range(i + 1, n):
            if _pair_consistency(hits[priority[i]], hits[priority[j]]) == "conflict":
                return None, {
                    "conflict": True, "winner": base_name, "n_sources": n, "n_agree": 0,
                    "detail": f"{priority[i]} 与 {priority[j]} close 相对差 > 5%",
                    "alerts": [],
                }

    # 按 close 聚类：以每源为锚求一致集，取最大组为多数一致组
    groups: Dict[str, set] = {}
    for a in priority:
        members = {a}
        for b in priority:
            if b != a and _pair_consistency(hits[a], hits[b]) is True:
                members.add(b)
        groups[a] = members

    max_size = max(len(m) for m in groups.values())
    best = [a for a in priority if len(groups[a]) == max_size]
    # 平局：优先取含最高优先级源（base_name）的组
    anchor = base_name if base_name in best else best[0]
    majority = groups[anchor]
    winner = min(majority, key=priority.index)  # 组内最高优先级源

    alerts: List[str] = []
    disagree = [s for s in priority if s not in majority]
    for s in disagree:
        alerts.append(f"{s} 与多数一致组（{sorted(majority)}）不一致")
    if len(majority) <= n - len(majority):
        # 平局或基准组非多数：用最高优先级源 + 报警
        alerts.append(
            f"命中源平局（{len(majority)} 一致 vs {n - len(majority)} 不一致），"
            f"采用最高优先级源 {winner}"
        )
    return hits[winner].copy().sort_values("date").reset_index(drop=True), {
        "n_sources": n, "n_agree": len(majority), "winner": winner,
        "conflict": False, "alerts": alerts,
    }


def _notify_alert(text: str) -> None:
    """报警：打印 [ALERT] 标记；飞书推送可用时同步推送（推送失败不影响主流程）"""
    print(f"  [ALERT] {text}")
    try:
        from feishu_push import push_text
        push_text(f"[trade-pulse] {text}")
    except Exception as e:
        print(f"  [WARN] 飞书推送失败: {e}")


def _fetch_multi_source(symbol: str, start: str, local_df: pd.DataFrame,
                        config: dict, target_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """多源并行拉取（两阶段）+ 裁决

    - 阶段 1：并行拉快源（tencent/sina，当天可得性最强）
    - 任一快源命中 → 进裁决，不碰慢源（东财/baostock 少暴露）
    - 快源全未命中 → 阶段 2 并行拉慢源
    - 命中判定：df 非空 且 normalize 后 max_date >= target_date 且过质量闸门
    - target_date 可注入（测试用）；None 时按 compute_target_date(now) 计算

    0 命中或源间严重冲突返回 None（调用方回退本地缓存，不覆盖）。
    """
    target_date = target_date or compute_target_date(datetime.now())
    sources = config.get("sources", DEFAULT_SOURCES)
    fast = config.get("fast_sources", DEFAULT_FAST_SOURCES)
    slow = [s for s in sources if s not in fast]
    cooldown_cfg = config.get("source_cooldown", {})

    health = load_source_health()
    results: dict = {}

    # 阶段 1：快源并行
    results.update(_run_phase(fast, symbol, start, target_date, local_df, health, cooldown_cfg, config))

    # 阶段 2：快源全未命中才碰慢源
    if not any(status == "HIT" for status, _, _, _ in results.values()):
        results.update(_run_phase(slow, symbol, start, target_date, local_df, health, cooldown_cfg, config))

    # 每源一行机器可读日志
    for src in sources:
        if src in results:
            status, _, detail, _ = results[src]
            print(f"  [SRC] {src} {status}({detail})")
        else:
            print(f"  [SRC] {src} NOTRUN(fast-hit)")

    save_source_health(health)

    hits = {src: r[1] for src, r in results.items() if r[0] == "HIT"}
    if not hits:
        # 本地已覆盖目标日（如快源节流跳过后重复跑）→ 数据实际已到位，非真失败
        if len(local_df) > 0:
            local_last = pd.to_datetime(local_df["date"]).max().strftime("%Y-%m-%d")
            if local_last >= target_date:
                print(f"  [INFO] {symbol} 数据已到位（本地最新 {local_last} >= 目标 {target_date}），返回本地缓存")
                return local_df
        print(f"  [FETCH_FAIL] {symbol} 多源均未命中目标日 {target_date}（尝试 {len(results)} 源）")
        return None

    priority = [s for s in sources if s in hits]
    winner, info = _adjudicate(hits, priority, symbol)
    if info.get("conflict"):
        print(f"  [FETCH_FAIL] {symbol} 数据源冲突（{info['detail']}），不覆盖本地数据")
        _notify_alert(f"{symbol} 多源数据冲突，不覆盖本地：{info['detail']}")
        return None
    for alert in info.get("alerts", []):
        _notify_alert(f"{symbol} {alert}")
    if info["n_agree"] < info["n_sources"]:
        print(f"  [INFO] 多源裁决：{info['n_agree']}/{info['n_sources']} 源一致，采用 {info['winner']}")
    return winner


def _fetch_legacy(symbol: str, start: str, local_df: pd.DataFrame, provider_name: str,
                  config: dict) -> Optional[pd.DataFrame]:
    """单源模式：原顺序降级链（主源重试 + 按 PROVIDER_ORDER 降级），行为与改造前一致"""
    provider = get_provider(provider_name)
    df_new = None
    retries = config.get("retry_count", 2)
    # 腾讯/新浪源需带市场前缀的 symbol（config['markets'] 映射），其余源用原始 symbol
    call_symbol = resolve_tencent_symbol(symbol) if provider_name in ("tencent", "sina") else symbol

    for attempt in range(retries + 1):
        try:
            df_new = provider.fetch_daily(call_symbol, start)
            if df_new is not None and len(df_new) > 0:
                df_new = provider.normalize(df_new, symbol)
                break
        except Exception as e:
            print(f"  [WARN] {provider.name} 失败: {e}")
            df_new = None
        if attempt < retries:
            time.sleep(config.get("retry_delay_sec", 3))

    # 备用 Provider：按固定顺序取除主源外的其余源降级
    fallbacks = [p for p in PROVIDER_ORDER if p != provider_name]
    for fb_name in fallbacks:
        if df_new is not None and len(df_new) > 0:
            break
        print(f"  [FALLBACK] 切 {fb_name}...")
        fb = get_provider(fb_name)
        # 主源/备用统一在调用点处理：tencent/sina 源用带市场前缀的 symbol
        fb_symbol = resolve_tencent_symbol(symbol) if fb_name in ("tencent", "sina") else symbol
        for attempt in range(retries + 1):
            try:
                df_new = fb.fetch_daily(fb_symbol, start)
                if df_new is not None and len(df_new) > 0:
                    df_new = fb.normalize(df_new, symbol)
                    break
            except Exception as e:
                print(f"  [WARN] {fb.name} 失败: {e}")
                df_new = None
            if attempt < retries:
                time.sleep(config.get("retry_delay_sec", 3))

    if df_new is None or len(df_new) == 0:
        print(f"  [FETCH_FAIL] {symbol} 数据源均不可用（主源 {provider_name} + 备用均失败）")
        return None

    # 数据质量闸门（合并前最后一道校验）
    df_new, q_warnings = quality_gate(df_new, local_df, symbol)
    for w in q_warnings:
        print(f"  ⚠️ [QUALITY] {w}")
    if df_new is None or len(df_new) == 0:
        print(f"  [FETCH_FAIL] {symbol} 数据全部被质量闸门拦截")
        return None
    return df_new


def merge_incremental(local_df: pd.DataFrame, new_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """合并增量：同日不覆盖 + 修订检测

    新数据与本地同日期 close 相对差 > 0.1% → 打印 [ALERT] 修订差异，
    保留本地值不静默改写；本地没有的日期追加新数据（按 date 排序）。
    """
    local = local_df.copy().sort_values("date").reset_index(drop=True)
    new = new_df.copy()
    local_dates = pd.to_datetime(local["date"]).dt.strftime("%Y-%m-%d")
    new_dates = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")

    local_close = dict(zip(local_dates, pd.to_numeric(local["close"], errors="coerce")))
    overlap = new[new_dates.isin(local_dates)]
    for _, row in overlap.iterrows():
        d = str(row["date"])[:10]
        lc = local_close.get(d)
        nc = float(row["close"])
        if lc and abs(nc - lc) / abs(lc) > CLOSE_CONSISTENT_PCT:
            print(f"  [ALERT] {symbol} 修订差异: {d} 本地 close {lc} vs 新数据 close {nc}（保留本地值）")

    keep = new[~new_dates.isin(local_dates)]
    if len(keep) == 0:
        # 无新增日：原样返回本地（避免 concat 引发列 dtype 提升导致 CSV 重写不一致，
        # 保证「连续跑两次 md5 一致」的增量幂等）
        return local
    combined = pd.concat([local, keep], ignore_index=True).sort_values("date").reset_index(drop=True)
    return combined


def _merge_and_save(symbol: str, df_new: pd.DataFrame, local_df: pd.DataFrame,
                    data_path: Path, force: bool) -> pd.DataFrame:
    """合并增量 + 盘中防护 + 写盘（多源/单源共用尾部）"""
    if len(local_df) > 0 and not force:
        combined = merge_incremental(local_df, df_new, symbol)
    else:
        combined = df_new

    # 盘中防护（双保险）：收盘前剔除当日未完成 bar（避免半日数据污染信号/特征）
    # 15:00 收盘后允许保留当日完整日线；target_date 语义已保证多源命中不含半日 bar，
    # 此逻辑对 legacy 路径仍必要，对多源路径兜底。
    if datetime.now().strftime("%H:%M") < "15:00":
        today_str = datetime.now().strftime("%Y-%m-%d")
        n_before = len(combined)
        combined = combined[combined["date"] != today_str].reset_index(drop=True)
        if len(combined) < n_before:
            print(f"  [INFO] 剔除当日盘中未收盘数据（{today_str}）")

    combined.to_csv(data_path, index=False)
    print(f"  [OK] {symbol}: {len(combined)} 条 ({combined['date'].min().date()} ~ {combined['date'].max().date()})")
    return combined


def fetch_data(
    symbol: str,
    start_date: str,
    force: bool = False,
    provider_name: str = None,
) -> pd.DataFrame:
    """拉取数据，自动增量更新

    provider_name 为 None 且 config['multi_source'] 非 false 时走多源并行调度；
    否则走原顺序降级链（行为与改造前一致）。
    """
    # 东财系（AkShare/EastMoney）走代理必失败（mihomo 规则拒绝），强制直连
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("all_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)

    config = load_config()
    data_path = get_data_path(symbol)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    local_df = load_local(data_path)

    if force or len(local_df) == 0:
        start = start_date
        print(f"  [INFO] 全量拉取 {symbol} 从 {start_date}")
    else:
        last_date = local_df["date"].max().strftime("%Y-%m-%d")
        start = last_date
        print(f"  [INFO] 增量更新 {symbol} 从 {start}（已有 {len(local_df)} 条）")

    # 模式路由：显式 --provider 或 config['multi_source']=false → 单源（原逻辑）
    multi = provider_name is None and config.get("multi_source", True)
    if multi:
        df_new = _fetch_multi_source(symbol, start, local_df, config)
    else:
        df_new = _fetch_legacy(symbol, start, local_df,
                                provider_name or config.get("provider", "akshare"), config)

    # 全部失败 → 用本地缓存（FETCH_FAIL 详情已在上层打印）
    if df_new is None or len(df_new) == 0:
        if len(local_df) > 0:
            print(f"  ⚠️ 数据源均不可用，使用本地缓存（{len(local_df)} 条）")
            local_df.attrs["stale"] = True
            return local_df
        print(f"  [FETCH_FAIL] {symbol} 无法获取数据，且无本地缓存")
        raise RuntimeError(f"无法获取 {symbol} 数据，且无本地缓存")

    return _merge_and_save(symbol, df_new, local_df, data_path, force)


def main():
    parser = argparse.ArgumentParser(description="拉取 588000 + 000688 日线数据")
    parser.add_argument("--force", action="store_true", help="强制全量重拉")
    parser.add_argument("--start", default="2023-01-01", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--provider", default=None,
                        choices=["akshare", "tencent", "eastmoney", "baostock", "sina"],
                        help="数据源（默认取 config.json 的 provider 字段；多源模式忽略）")
    parser.add_argument("--require-date", default=None,
                        help="增量完成后校验最新日期 >= 该日期 (YYYY-MM-DD)，不满足则退出码 1")
    args = parser.parse_args()

    config = load_config()
    print("\n📥 数据拉取")
    df_sym = fetch_data(config["symbol"], args.start, args.force, args.provider)
    df_bench = fetch_data(config["benchmark"], args.start, args.force, args.provider)

    if args.require_date:
        for name, df in ((config["symbol"], df_sym), (config["benchmark"], df_bench)):
            result = validate_incremental(df, require_date=args.require_date)
            if not result["ok"]:
                print(f"  [FAIL] {name} 数据完整性校验失败（require-date={args.require_date}）")
                for issue in result["issues"]:
                    print(f"    - {issue}")
                sys.exit(1)
            print(f"  [OK] {name} 数据完整，最新日期 {result['last_date']}")

    return df_sym, df_bench


if __name__ == "__main__":
    main()
