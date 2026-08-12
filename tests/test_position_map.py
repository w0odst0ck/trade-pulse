#!/usr/bin/env python3
"""position_map 单元测试（pytest，不联网）

覆盖：
  - 默认映射与历史行为同构（score≤cap_score 线性 / >cap_score 封顶）
  - 候选映射差异化生效（square/flat/base/slope/cap）
  - config 覆盖优先于文件默认；非法值回退
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "daily_pipeline"))

import position_map as pm


class TestDefaultLinear:
    """默认映射必须与历史 calc_position 行为完全一致（基线同构）"""

    def test_empty_position_zero(self):
        assert pm.calc_position('空仓', 0.9) == 0.0

    def test_linear_below_cap_score(self):
        # score=0.3 → 0.3 + 0.3*0.4 = 0.42
        assert pm.calc_position('持仓', 0.3) == 0.42

    def test_linear_at_cap_score_boundary(self):
        # score=0.69 ≤ 0.7 → 0.3 + 0.69*0.4 = 0.576 → 0.58
        assert pm.calc_position('持仓', 0.69) == 0.58

    def test_cap_score_override(self):
        # score=0.71 > 0.7 → 封顶 0.7
        assert pm.calc_position('持仓', 0.71) == 0.7

    def test_high_score_capped(self):
        assert pm.calc_position('持仓', 1.0) == 0.7

    def test_negative_score_floor(self):
        # score=-0.5 → 钳到 0 → 0.3 + 0*0.4 = 0.3
        assert pm.calc_position('持仓', -0.5) == 0.3

    def test_watch_state_uses_map(self):
        # 观望也按映射算（非空仓）
        assert pm.calc_position('观望', 0.5) == 0.5


class TestConfigOverride:
    """config['position_map'] 覆盖优先于文件默认"""

    def test_square_map(self):
        cfg = {"position_map": {"type": "square", "base": 0.3, "slope": 0.7, "cap": 0.7}}
        # score=0.5 → 0.3 + 0.25*0.7 = 0.475 → 0.48? round(0.475,2)=0.47（银行家舍入）
        assert pm.calc_position('持仓', 0.5, cfg) == 0.47

    def test_flat_map(self):
        cfg = {"position_map": {"type": "flat", "base": 0.5}}
        assert pm.calc_position('持仓', 0.9, cfg) == 0.5

    def test_different_base_slope_cap(self):
        # cap_score=1.0 时不做分段封顶（候选映射语义）
        cfg = {"position_map": {"type": "linear", "base": 0.2, "slope": 0.4, "cap": 0.6, "cap_score": 1.0}}
        # score=0.9 → 0.2 + 0.36 = 0.56 ≤ 0.6
        assert pm.calc_position('持仓', 0.9, cfg) == 0.56

    def test_high_base_capped_by_cap(self):
        cfg = {"position_map": {"type": "linear", "base": 0.5, "slope": 0.2, "cap": 0.7, "cap_score": 1.0}}
        # score=0.9 → 0.5 + 0.18 = 0.68 ≤ 0.7
        assert pm.calc_position('持仓', 0.9, cfg) == 0.68

    def test_invalid_cap_falls_back(self):
        # cap > 1 非法 → 回退默认 0.7
        cfg = {"position_map": {"type": "linear", "base": 0.3, "slope": 0.4, "cap": 1.5}}
        assert pm.calc_position('持仓', 1.0, cfg) == 0.7

    def test_empty_position_map_falls_back_to_file(self):
        # config 无 position_map → 读文件默认
        assert pm.calc_position('持仓', 0.3, {}) == 0.42


class TestResolvePm:
    def test_load_position_map_default(self):
        p = pm.load_position_map()
        assert p["type"] == "linear"
        assert p["base"] == 0.3
        assert p["cap"] == 0.7

    def test_resolve_priority(self):
        # 传入 config 优先
        cfg = {"position_map": {"type": "flat", "base": 0.6}}
        p = pm._resolve_pm(cfg)
        assert p["type"] == "flat"
        assert p["base"] == 0.6
