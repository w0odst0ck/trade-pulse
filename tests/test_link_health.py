#!/usr/bin/env python3
"""link_health 单元测试（pytest，不联网）

覆盖：
  - compute_raw_level 分级：full / degraded / stale / broken / blind
  - probe_ts 时效：旧探测数据不参与 stale 判定（周末/隔夜残留不误判）
  - apply_hysteresis 滞回：连续 2 次同向才切换，单次抖动不横跳；
    blind 需连续 3 次（BLIND_FAIL_THRESHOLD）
  - apply_multiplier 乘数：信号仓位 × 链路乘数
  - get_link_confidence 无探测数据默认 full
"""

import sys
import time
from datetime import date as date_cls
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import link_health as lh


def make_probe(tencent=True, sina=True, rt_t=True, rt_s=True,
               max_date="2026-08-11", probe_ts=None, probe_mode="morning"):
    """构造 probe_health 结构；probe_ts 默认当前时间（新鲜）"""
    ts = probe_ts if probe_ts is not None else time.time()
    return {
        "588000": {
            "tencent": {"ok": tencent, "max_date": max_date, "ts": ts},
            "sina": {"ok": sina, "max_date": max_date, "ts": ts},
            "realtime_tencent": {"ok": rt_t, "max_date": None, "ts": ts},
            "realtime_sina": {"ok": rt_s, "max_date": None, "ts": ts},
            "probe_ts": ts,
            "probe_mode": probe_mode,
        }
    }


class TestRawLevel:
    @staticmethod
    def _today_iso() -> str:
        return date_cls.today().isoformat()

    def test_full_when_all_ok(self, monkeypatch):
        monkeypatch.setattr(lh, "load_probe_health", lambda: make_probe(max_date=self._today_iso()))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "full"

    def test_degraded_single_daily(self, monkeypatch):
        monkeypatch.setattr(lh, "load_probe_health", lambda: make_probe(tencent=False, max_date=self._today_iso()))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "degraded"

    def test_degraded_single_realtime(self, monkeypatch):
        monkeypatch.setattr(lh, "load_probe_health", lambda: make_probe(rt_t=False, max_date=self._today_iso()))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "degraded"

    def test_stale_one_day(self, monkeypatch):
        # after_close 探测：预期=今日，max_date=昨日 → 陈旧 1 天
        from datetime import timedelta
        yday = (date_cls.today() - timedelta(days=1)).isoformat()
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(max_date=yday, probe_mode="after_close"))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "stale"

    def test_morning_yesterday_data_not_stale(self, monkeypatch):
        # morning 探测：预期=昨日，max_date=昨日 → 正常（今天 bar 未出）不判 stale
        from datetime import timedelta
        yday = (date_cls.today() - timedelta(days=1)).isoformat()
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(max_date=yday, probe_mode="morning"))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "full"

    def test_stale_skipped_when_probe_stale(self, monkeypatch):
        # probe_ts 是 3 天前（周末残留）→ 不参与 stale 判定 → 即使 max_date 旧也判 full
        from datetime import timedelta
        old_ts = time.time() - 72 * 3600
        yday = (date_cls.today() - timedelta(days=1)).isoformat()
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(max_date=yday, probe_ts=old_ts))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "full"

    def test_broken_two_days_stale(self, monkeypatch):
        from datetime import timedelta
        d3 = (date_cls.today() - timedelta(days=3)).isoformat()
        monkeypatch.setattr(lh, "load_probe_health", lambda: make_probe(max_date=d3))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "broken"

    def test_broken_both_daily_down(self, monkeypatch):
        monkeypatch.setattr(lh, "load_probe_health", lambda: make_probe(tencent=False, sina=False))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "broken"

    def test_blind_all_down_and_stale(self, monkeypatch):
        from datetime import timedelta
        d3 = (date_cls.today() - timedelta(days=3)).isoformat()
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(tencent=False, sina=False, rt_t=False, rt_s=False,
                                               max_date=d3))
        r = lh.compute_raw_level("588000")
        assert r["level"] == "blind"


class TestHysteresis:
    def test_first_probe_accepts(self):
        r = lh.apply_hysteresis("degraded", {"level": None, "ok_streak": 0, "fail_streak": 0})
        assert r["level"] == "degraded"

    def test_single_failure_keeps_full(self):
        r = lh.apply_hysteresis("degraded", {"level": "full", "ok_streak": 1, "fail_streak": 0})
        assert r["level"] == "full"  # 1 次失败不降级
        assert r["fail_streak"] == 1

    def test_two_failures_downgrade(self):
        r1 = lh.apply_hysteresis("degraded", {"level": "full", "ok_streak": 1, "fail_streak": 0})
        r2 = lh.apply_hysteresis("degraded", {"level": "full", "ok_streak": 1, "fail_streak": r1["fail_streak"]})
        assert r2["level"] == "degraded"
        assert r2.get("changed") is True

    def test_blind_needs_three_failures(self):
        # blind 需连续 3 次（BLIND_FAIL_THRESHOLD），1-2 次不跳
        r1 = lh.apply_hysteresis("blind", {"level": "full", "ok_streak": 1, "fail_streak": 0})
        assert r1["level"] == "full"
        r2 = lh.apply_hysteresis("blind", {"level": "full", "ok_streak": 1, "fail_streak": r1["fail_streak"]})
        assert r2["level"] == "full"
        r3 = lh.apply_hysteresis("blind", {"level": "full", "ok_streak": 1, "fail_streak": r2["fail_streak"]})
        assert r3["level"] == "blind"
        assert r3.get("changed") is True

    def test_broken_first_failure_switches(self):
        # broken（双源全挂/陈旧≥2）风控优先：1 次即切换
        r = lh.apply_hysteresis("broken", {"level": "full", "ok_streak": 1, "fail_streak": 0})
        assert r["level"] == "broken"
        assert r.get("changed") is True

    def test_single_improve_keeps_degraded(self):
        r = lh.apply_hysteresis("full", {"level": "degraded", "ok_streak": 0, "fail_streak": 2})
        assert r["level"] == "degraded"  # 1 次恢复不升级
        assert r["ok_streak"] == 1

    def test_two_improves_upgrade(self):
        r1 = lh.apply_hysteresis("full", {"level": "degraded", "ok_streak": 0, "fail_streak": 2})
        r2 = lh.apply_hysteresis("full", {"level": "degraded", "ok_streak": r1["ok_streak"], "fail_streak": 2})
        assert r2["level"] == "full"
        assert r2.get("changed") is True

    def test_same_level_degraded_accumulates(self):
        # 同等级持续降级：fail_streak 累积（第二次达阈值触发一次告警）
        r1 = lh.apply_hysteresis("degraded", {"level": "degraded", "ok_streak": 0, "fail_streak": 1})
        assert r1["level"] == "degraded"
        assert r1["fail_streak"] == 2
        assert r1.get("changed") is True  # 恰达阈值（2）触发
        r2 = lh.apply_hysteresis("degraded", {"level": "degraded", "ok_streak": 0, "fail_streak": 2})
        assert r2["fail_streak"] == 3
        assert r2.get("changed") is False  # 超过阈值不再触发（防轰炸）

    def test_full_stable_resets(self):
        # full 稳定：计数归零
        r = lh.apply_hysteresis("full", {"level": "full", "ok_streak": 3, "fail_streak": 1})
        assert r["fail_streak"] == 0
        assert r["ok_streak"] == 0


class TestMultiplier:
    def test_full_keeps_position(self):
        c = {"level": "full", "multiplier": 1.0}
        r = lh.apply_multiplier(0.6, c)
        assert r["advised_position"] == 0.6

    def test_degraded_discounts(self):
        c = {"level": "degraded", "multiplier": 0.75}
        r = lh.apply_multiplier(0.6, c)
        assert r["advised_position"] == 0.45

    def test_broken_heavy_discount(self):
        c = {"level": "broken", "multiplier": 0.3}
        r = lh.apply_multiplier(0.5, c)
        assert r["advised_position"] == 0.15

    def test_blind_zero(self):
        c = {"level": "blind", "multiplier": 0.0}
        r = lh.apply_multiplier(0.5, c)
        assert r["advised_position"] == 0.0


class TestGetConfidence:
    def test_no_probe_defaults_full(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lh, "load_probe_health", lambda: {})
        monkeypatch.setattr(lh, "PROBE_STATE_PATH", tmp_path / "probe_state.json")
        c = lh.get_link_confidence("588000")
        assert c["level"] == "full"
        assert c["multiplier"] == 1.0


class TestEvaluateAlert:
    def test_no_probe_no_alert(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lh, "load_probe_health", lambda: {})
        monkeypatch.setattr(lh, "PROBE_STATE_PATH", tmp_path / "probe_state.json")
        a = lh.evaluate_alert("588000")
        assert a["should_alert"] is False
        assert a["severity"] == "none"

    def test_full_no_alert(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(max_date=date_cls.today().isoformat()))
        monkeypatch.setattr(lh, "PROBE_STATE_PATH", tmp_path / "probe_state.json")
        a = lh.evaluate_alert("588000")
        assert a["should_alert"] is False
        assert a["severity"] == "none"

    def test_broken_first_failure_emergency(self, monkeypatch, tmp_path):
        # 双源日线全挂 → broken，首次即 emergency 告警
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(tencent=False, sina=False,
                                               max_date=date_cls.today().isoformat()))
        monkeypatch.setattr(lh, "PROBE_STATE_PATH", tmp_path / "probe_state.json")
        a = lh.evaluate_alert("588000")
        assert a["should_alert"] is True
        assert a["severity"] == "emergency"

    def test_broken_second_run_no_repeat(self, monkeypatch, tmp_path):
        # 持续挂：第二次不再告警（防轰炸）
        p = tmp_path / "probe_state.json"
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(tencent=False, sina=False,
                                               max_date=date_cls.today().isoformat()))
        monkeypatch.setattr(lh, "PROBE_STATE_PATH", p)
        a1 = lh.evaluate_alert("588000")
        assert a1["should_alert"] is True
        a2 = lh.evaluate_alert("588000")
        assert a2["should_alert"] is False

    def test_degraded_needs_two_runs(self, monkeypatch, tmp_path):
        # 单源降级：第一次不告警（fail_streak=1），第二次才告警
        p = tmp_path / "probe_state.json"
        monkeypatch.setattr(lh, "load_probe_health",
                            lambda: make_probe(tencent=False,
                                               max_date=date_cls.today().isoformat()))
        monkeypatch.setattr(lh, "PROBE_STATE_PATH", p)
        a1 = lh.evaluate_alert("588000")
        assert a1["should_alert"] is False
        a2 = lh.evaluate_alert("588000")
        assert a2["should_alert"] is True
        assert a2["severity"] == "normal"
