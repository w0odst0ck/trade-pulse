#!/usr/bin/env python3
"""link_health.py — 信号链路可信度 → 仓位乘数（实盘风控）

探测结果（probe_runner.py 写入 source_health.json 的 probe 字段）→
链路可信度分级 → 仓位乘数。daily_panel 用它给信号仓位打折，
避免基于降级/陈旧数据的重仓决策。

分级（滞回：连续 2 次同向才切换，防单次抖动横跳）：
  🟢 1.0  full   双源正常 + 数据新鲜（盘中实时可用）
  🟡 0.75 degraded  单源降级（日线或实时仅 1 源可用）
  🟠 0.6  stale  数据陈旧 1 天（max_date < 预期 1 天）
  🔴 0.3  broken 双源全挂 或 数据陈旧 ≥2 天
  ⛔ 0    blind  连续探测失败(≥3) 且 数据陈旧 ≥2 天——信号不可信，不操作

设计要点：
  - 探测失败 ≠ 信号错误：只降信任，不写 state（状态机照常）
  - 乘数取值经验值（0.75/0.6/0.3/0），跑 2-4 周看实际降级频率再校准
  - 已持仓：只影响新决策（开仓/加仓），不触发减仓（降级不代表持仓信号错了）
"""

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from trading_calendar import is_trading_day, prev_trading_day

# fetch_data 专用（cooldown/命中）；探测用 PROBE_HEALTH_PATH（分离防并发冲突）
PROBE_HEALTH_PATH = SCRIPT_DIR / "probe_health.json"     # 探测专用（与 fetch 完全分离，零并发冲突）
PROBE_STATE_PATH = SCRIPT_DIR / "probe_state.json"

# 分级 → 乘数（经验值）
LEVEL_MULTIPLIER = {
    "full": 1.0,
    "degraded": 0.75,
    "stale": 0.6,
    "broken": 0.3,
    "blind": 0.0,
}
LEVEL_EMOJI = {"full": "🟢", "degraded": "🟡", "stale": "🟠", "broken": "🔴", "blind": "⛔"}

# 滞回：连续 N 次同向探测才切换等级
HYSTERESIS_N = 2
# blind 判定：连续探测失败次数阈值（get_link_confidence 里用 fail_streak 累积，
# 单次全挂不直接跳 0.0——绕过滞回的最破坏性等级必须有连续证据）
BLIND_FAIL_THRESHOLD = 3
# 数据陈旧判定（天）
STALE_DAYS = 1
BROKEN_STALE_DAYS = 2
# 探测数据时效（小时）：probe_ts 距今超过此值 → 探测数据视为过期，
# 不参与 stale 判定（周末/隔夜残留探测不误判陈旧）
PROBE_FRESH_HOURS = 26


def load_probe_health() -> Dict:
    """读 probe_health.json（探测专用文件，与 fetch_data 的 source_health.json 分离）

    结构：{symbol: {"tencent": {"ok": bool, "max_date": str, "ts": float}, ...}}
    """
    if not PROBE_HEALTH_PATH.exists():
        return {}
    try:
        with open(PROBE_HEALTH_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data


def load_probe_state() -> Dict:
    if PROBE_STATE_PATH.exists():
        try:
            return json.loads(PROBE_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"level": None, "ok_streak": 0, "fail_streak": 0, "ts": 0}


def save_probe_state(state: Dict) -> None:
    """原子写（临时文件 + rename），防并发读半截"""
    tmp = PROBE_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(PROBE_STATE_PATH)


def compute_raw_level(symbol: str) -> Dict:
    """根据 probe 健康数据算原始分级（不滞回）

    数据新鲜度原则：
      - 探测数据必须新鲜（probe_ts 距今 < PROBE_FRESH_HOURS）才参与 stale 判定
      - 旧探测数据（周末/隔夜残留：09:05 跑过，15:00 后再读）→ 不判 stale，
        只根据「当时探测的 ok 状态」判降级（接口状态短期有效）
      - 这避免「读时墙钟」与「探测时点」混用导致周末后误判陈旧
    """
    health = load_probe_health()
    sym = health.get(symbol, {})
    probes = {k: v for k, v in sym.items() if isinstance(v, dict) and "ok" in v}
    if not probes:
        return {"level": "full", "reason": "无探测数据（默认信任）"}

    tencent_ok = probes.get("tencent", {}).get("ok", False)
    sina_ok = probes.get("sina", {}).get("ok", False)
    rt_tencent_ok = probes.get("realtime_tencent", {}).get("ok", False)
    rt_sina_ok = probes.get("realtime_sina", {}).get("ok", False)

    # 探测数据时效：probe_ts 距今 < 26h 视为新鲜（覆盖隔夜 + 单日间隔）
    probe_ts = sym.get("probe_ts") or 0
    probe_fresh = (time.time() - probe_ts) < PROBE_FRESH_HOURS * 3600

    stale_days = None
    if probe_fresh:
        # 预期数据日期由**探测时点**决定（不用读时墙钟——避免 morning 探测的
        # 「昨日数据（正常）」被读成陈旧，也避免跨交易日读取漂移）：
        #   morning/intraday 探测 → 预期 = 探测日的最近已收盘交易日（今天 bar 未出）
        #   after_close 探测 → 预期 = 探测日（收盘后应有今日数据或走兜底链）
        # 基准日 = probe 时间戳的日期（探测视角），与 probe_runner 的预期语义一致
        probe_date = date.fromtimestamp(probe_ts)
        probe_mode = sym.get("probe_mode") or "morning"
        if probe_mode == "after_close" and is_trading_day(probe_date):
            expected = probe_date
        else:
            expected = prev_trading_day(probe_date)
        max_date_str = None
        for k in ("tencent", "sina"):
            md = probes.get(k, {}).get("max_date")
            if md:
                max_date_str = max(max_date_str or "", md)
        if max_date_str:
            try:
                md = date.fromisoformat(max_date_str[:10])
                stale_days = (expected - md).days
            except ValueError:
                stale_days = None

    daily_alive = tencent_ok + sina_ok          # 0/1/2
    realtime_alive = rt_tencent_ok + rt_sina_ok  # 0/1/2

    # blind：日线全挂且实时全挂，且数据陈旧（须连续失败阈值——见 apply_hysteresis
    # 的 fail_streak 累积；这里只给原始状态，不直接跳到 0.0）
    if daily_alive == 0 and realtime_alive == 0 and stale_days is not None and stale_days >= BROKEN_STALE_DAYS:
        return {"level": "blind", "reason": "双源全挂+数据陈旧", "stale_days": stale_days}
    # broken：双源全挂 或 数据陈旧 ≥2 天
    if daily_alive == 0 or (stale_days is not None and stale_days >= BROKEN_STALE_DAYS):
        if daily_alive == 0:
            reason = "日线双源全挂"
        else:
            reason = f"数据陈旧{stale_days}天"
        return {"level": "broken", "reason": reason, "stale_days": stale_days}
    # stale：数据陈旧 1 天
    if stale_days is not None and stale_days >= STALE_DAYS:
        return {"level": "stale", "reason": f"数据陈旧{stale_days}天", "stale_days": stale_days}
    # degraded：单源降级（日线或实时）
    if daily_alive < 2 or realtime_alive < 2:
        parts = []
        if daily_alive < 2:
            parts.append(f"日线{int(daily_alive)}/2源")
        if realtime_alive < 2:
            parts.append(f"实时{int(realtime_alive)}/2源")
        return {"level": "degraded", "reason": "单源降级：" + "+".join(parts)}
    return {"level": "full", "reason": "双源正常+数据新鲜"}


def apply_hysteresis(raw_level: str, prev_state: Dict) -> Dict:
    """滞回：连续 N 次同向才切换等级/告警

    模型：prev_state 记录当前等级 + 连续同向证据计数。
      - 首次（prev=None）：直接采纳 raw；仅 broken/blind 视为变更（风控优先）
      - 恶化/同等级持续非满级：fail_streak+1，**恰达阈值**（== need）才 changed
        （此后 fail_streak 继续涨但不再触发——防重复轰炸，恢复后重置）
      - 改善：ok_streak+1，恰达阈值才升回
      - 回到 full：两计数归零
    """
    prev = prev_state.get("level")
    if prev is None:
        # 首次探测：直接采纳当前等级；仅 broken/blind 视为变更（风控优先，
        # 首次双源全挂即告警），degraded/stale 首次不告警（等连续确认）。
        # 非满级时 fail_streak 记 1（本次已算一次失败证据），同等级后续累积。
        return {"level": raw_level, "ok_streak": 1, "fail_streak": 1 if raw_level != "full" else 0,
                "changed": raw_level in ("broken", "blind")}
    rank = ["full", "degraded", "stale", "broken", "blind"]
    prev_idx = rank.index(prev) if prev in rank else 0
    raw_idx = rank.index(raw_level) if raw_level in rank else 0

    def _need(level: str) -> int:
        if level == "blind":
            return BLIND_FAIL_THRESHOLD  # 最破坏性：连续 3 次防误跳 0
        if level == "broken":
            return 1  # 风控优先：首次即切换（探测已重试过，真实故障）
        return HYSTERESIS_N  # degraded/stale：连续 2 次

    if raw_idx == prev_idx:
        # 同等级：非满级持续 → fail_streak 累积（恰达阈值触发一次告警，
        # 超过后不再触发——防重复轰炸）；full 稳定 → 计数归零
        if raw_level == "full":
            return {"level": "full", "ok_streak": 0, "fail_streak": 0, "changed": False}
        fail_streak = prev_state.get("fail_streak", 0) + 1
        need = _need(raw_level)
        # 首次进入该等级时 fail_streak 已=1（见恶化分支），
        # 故这里恰达阈值只在「从更低累积到阈值」时触发一次
        changed = (fail_streak == need and prev_state.get("fail_streak", 0) < need)
        return {"level": raw_level, "ok_streak": 0, "fail_streak": fail_streak,
                "changed": changed}
    if raw_idx < prev_idx:  # 改善（full ← degraded/...）
        ok_streak = prev_state.get("ok_streak", 0) + 1
        if ok_streak >= HYSTERESIS_N:
            return {"level": raw_level, "ok_streak": ok_streak, "fail_streak": 0, "changed": True}
        return {"level": prev, "ok_streak": ok_streak, "fail_streak": 0, "changed": False}
    # 恶化（full → degraded/...）
    fail_streak = prev_state.get("fail_streak", 0) + 1
    need = _need(raw_level)
    if fail_streak >= need:
        return {"level": raw_level, "ok_streak": 0, "fail_streak": fail_streak, "changed": True}
    return {"level": prev, "ok_streak": 0, "fail_streak": fail_streak, "changed": False}


def _compute_confidence(symbol: str) -> Dict:
    """核心：读 probe 数据 → raw 分级 → 滞回 → 持久化 → 返回完整可信度

    probe.sh（探测后）与 daily_panel（信号时）都走这里——唯一状态机，
    避免多 writer 各自维护一套滞回状态互相覆盖。
    """
    health = load_probe_health()
    sym = health.get(symbol, {})
    raw = compute_raw_level(symbol)
    prev = load_probe_state()

    # 有探测数据才滞回；无探测数据（未跑过）默认 full
    has_probe = any(isinstance(v, dict) and "ok" in v for v in sym.values())
    if not has_probe:
        return {"level": "full", "multiplier": 1.0, "emoji": "🟢",
                "reason": "无探测数据（默认信任）", "stale_days": None,
                "ts": time.time(), "changed": False}

    state = apply_hysteresis(raw["level"], prev)
    state["ts"] = time.time()
    save_probe_state(state)

    level = state["level"]
    return {
        "level": level,
        "multiplier": LEVEL_MULTIPLIER[level],
        "emoji": LEVEL_EMOJI[level],
        "reason": raw.get("reason", ""),
        "stale_days": raw.get("stale_days"),
        "ts": state["ts"],
        "changed": state.get("changed", False),
    }


def get_link_confidence(symbol: str = "588000") -> Dict:
    """对外主入口：返回当前链路可信度 + 仓位乘数（daily_panel 用）

    返回: {level, multiplier, emoji, reason, stale_days, ts}
    """
    return _compute_confidence(symbol)


def evaluate_alert(symbol: str = "588000") -> Dict:
    """probe.sh 专用：探测后决定是否告警 + 级别（唯一状态机，防双 writer）

    返回: {should_alert, severity, level, emoji, reason, changed}
      severity: emergency（broken/blind）/ normal（degraded/stale）/ none（full）
      should_alert: changed 且 severity != none（broken 首次即告警；
        degraded/stale 连续 2 次；blind 连续 3 次——滞回阈值统一在此）
    """
    c = _compute_confidence(symbol)
    severity = {
        "broken": "emergency", "blind": "emergency",
        "degraded": "normal", "stale": "normal",
    }.get(c["level"], "none")
    return {
        "should_alert": bool(c.get("changed")) and severity != "none",
        "severity": severity,
        "level": c["level"],
        "emoji": c["emoji"],
        "reason": c.get("reason", ""),
        "changed": c.get("changed", False),
        "stale_days": c.get("stale_days"),
    }


def apply_multiplier(signal_position: float, confidence: Dict) -> Dict:
    """信号仓位 × 链路乘数 → 建议仓位

    返回: {signal_position, multiplier, advised_position, level, emoji, reason}
    """
    mult = confidence.get("multiplier", 1.0)
    advised = round(signal_position * mult, 2)
    return {
        "signal_position": signal_position,
        "multiplier": mult,
        "advised_position": advised,
        "level": confidence.get("level"),
        "emoji": confidence.get("emoji", "🟢"),
        "reason": confidence.get("reason", ""),
        "stale_days": confidence.get("stale_days"),
    }


if __name__ == "__main__":
    c = get_link_confidence(sys.argv[1] if len(sys.argv) > 1 else "588000")
    print(f"链路可信度: {c['emoji']} {c['level']} ×{c['multiplier']}（{c['reason']}）")
