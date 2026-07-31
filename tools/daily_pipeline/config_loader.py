"""config_loader.py — 策略配置加载器

从 YAML 策略文件读取配置，转成 Python dict。
支持 --strategy 参数切换。
"""

import os
from pathlib import Path
from typing import Optional

import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STRATEGIES_DIR = PROJECT_ROOT / "strategies"


def load_strategy(name: str = "default") -> dict:
    """加载策略配置

    Parameters
    ----------
    name : str
        策略文件名（不含 .yaml），如 "588000"、"588000_aggressive"

    Returns
    -------
    dict — 完整策略配置字典
    """
    path = STRATEGIES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"策略文件不存在: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_strategy_name"] = name
    config["_strategy_path"] = str(path)
    return config


def list_strategies() -> list[dict]:
    """列出可用策略"""
    results = []
    if not STRATEGIES_DIR.exists():
        return results
    for f in sorted(STRATEGIES_DIR.glob("*.yaml")):
        name = f.stem
        try:
            cfg = load_strategy(name)
            results.append({
                "name": name,
                "description": cfg.get("description", ""),
                "path": str(f),
            })
        except Exception:
            pass
    return results
