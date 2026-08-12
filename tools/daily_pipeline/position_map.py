#!/usr/bin/env python3
"""position_map.py — 仓位映射（参数化，独立模块）

signal_rules / backtest / paper_trade 三方统一引用（保证回测=纸面=实盘同构）。
映射参数从 config['position_map'] 读取，带默认值兜底。

映射类型：
  linear   pos = base + score * slope，cap 封顶（现状 0.3 + score×0.4，cap 0.7）
  square   pos = base + score² * slope，cap 封顶（低分更保守）
  flat     pos = base 常数（信号转多即固定仓位）

用法：
  from position_map import calc_position
  pos = calc_position(state, score, config)   # 0.0 ~ cap
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

# 默认映射（与历史行为一致：score≤0.7 时 0.3+score×0.4，score>0.7 封顶 0.7）
DEFAULT_POSITION_MAP = {
    "type": "linear",
    "base": 0.3,
    "slope": 0.4,
    "cap": 0.7,
    "cap_score": 0.7,   # 历史行为：score 超过此值直接封顶 cap（分段，非计算值 min cap）
}


def _validate_pm(cfg_pm: dict) -> dict:
    """校验并规范化 position_map 配置（type 白名单 + 数值范围检查）

    非法值回退默认；返回完整映射 dict。load_position_map / _resolve_pm 共用。
    """
    pm = dict(DEFAULT_POSITION_MAP)
    typ = cfg_pm.get("type", "linear")
    if typ in ("linear", "square", "flat"):
        pm["type"] = typ
    for k in ("base", "slope", "cap", "cap_score"):
        v = cfg_pm.get(k)
        if isinstance(v, (int, float)):
            v = float(v)
            if k == "cap" and not (0.0 < v <= 1.0):
                continue
            if k in ("base", "slope", "cap_score") and v < 0:
                continue
            pm[k] = v
    return pm


def load_position_map() -> dict:
    """读 config['position_map']，带默认值兜底 + 校验

    - config 缺失/损坏 → 默认
    - 非法值（base<0 / slope<0 / cap 越界 [0,1]）→ 默认
    """
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg_pm = (json.load(f).get("position_map") or {})
        return _validate_pm(cfg_pm)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_POSITION_MAP)


def _resolve_pm(config: dict = None) -> dict:
    """解析映射参数：优先用传入 config['position_map']（回测/搜索覆盖），
    否则读 config.json。传入的 config 缺 position_map 字段时回退文件默认。"""
    if config is not None:
        cfg_pm = config.get("position_map")
        if isinstance(cfg_pm, dict) and cfg_pm:
            return _validate_pm(cfg_pm)
    return load_position_map()


def calc_position(state_val: str, score: float, config: dict = None) -> float:
    """信号 → 仓位（0.0 ~ cap）

    state_val: '空仓' → 0；其他（持仓/观望）→ 按映射计算
    score: total_score（-1~+1），负分钳到 0
    config: 可选；传入时用 config['position_map']（回测/搜索用），None 时读 config.json
    """
    if state_val == '空仓':
        return 0.0
    pm = _resolve_pm(config)
    base = pm["base"]
    slope = pm["slope"]
    cap = pm["cap"]

    pos_score = max(0.0, min(score, 1.0))
    if pm["type"] == "square":
        pos = pm["base"] + pos_score * pos_score * pm["slope"]
    elif pm["type"] == "flat":
        pos = pm["base"]
    else:  # linear
        # 历史行为（分段）：score 超过 cap_score 直接封顶 cap；
        # 否则 base + score×slope。注意这不是「计算结果 min cap」——
        # 两者在 score∈(cap_score,1) 区间行为不同（基线同构约束）
        if pos_score > pm.get("cap_score", 0.7):
            pos = cap
        else:
            pos = pm["base"] + pos_score * pm["slope"]
    return round(min(pos, cap), 2)


if __name__ == "__main__":
    import sys
    # 快速验证：score 从 0 到 1 的仓位输出
    pm = load_position_map()
    print(f"映射: {pm}")
    for s in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        print(f"  score={s:+.2f} → 仓位 {calc_position('持仓', s)}")
