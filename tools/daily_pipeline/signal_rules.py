#!/usr/bin/env python3
"""
signal_rules.py — 三级决策状态机

功能：读特征 → 按状态机逻辑 → 出决策 + 仓位建议
状态持久化到 data/state.json，重启不丢失

用法：
  python signal_rules.py                          # 跑一次决策
  python signal_rules.py --reset                  # 重置状态机为空仓
  python signal_rules.py --verbose                # 输出完整决策链
"""

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
STATE_PATH = PROJECT_ROOT / "data" / "588000" / "state.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_state() -> dict:
    """读取持久化状态"""
    default = {
        'state': '空仓',         # 持仓 / 观望 / 空仓
        'waiting_days': 0,       # 确认计数
        'last_decision_date': None,  # 上次决策日期
    }
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            saved = json.load(f)
            default.update(saved)
    return default


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def factor_emoji(score: float) -> str:
    """分数 → 可视化"""
    if score > 0.5: return '🟢 强'
    if score > 0.2: return '🟢'
    if score > -0.2: return '⚪ 中性'
    if score > -0.5: return '🔴'
    return '🔴 弱'


def factor_arrow(score: float) -> str:
    if score > 0.3: return '↑'
    if score > 0: return '↗'
    if score > -0.3: return '→'
    if score > -0.6: return '↘'
    return '↓'


def get_adaptive_thresholds(config: dict, latest_features: dict) -> dict:
    """根据 MA60 斜率自适应选择阈值"""
    at = config.get('adaptive_thresholds', {})
    if not at.get('enabled', False):
        return config['thresholds']

    ma60_sl = latest_features.get('ma60_slope', 0)
    if ma60_sl is None or ma60_sl != ma60_sl:  # 仅捕获 None 和 NaN，0 是合法值
        return config['thresholds']

    base = config['thresholds']
    if ma60_sl > at.get('ma60_slope_uptrend', 0.005):
        return {
            'buy': at.get('uptrend_buy', 0.2),
            'sell': at.get('uptrend_sell', -0.1),
            'confirm_days': base.get('confirm_days', 2),
            'weekly_filter_percentile': base.get('weekly_filter_percentile', 0.2),
        }
    elif ma60_sl < at.get('ma60_slope_downtrend', -0.005):
        return {
            'buy': at.get('downtrend_buy', 0.4),
            'sell': at.get('downtrend_sell', -0.3),
            'confirm_days': base.get('confirm_days', 2),
            'weekly_filter_percentile': base.get('weekly_filter_percentile', 0.2),
        }
    else:
        return {
            'buy': at.get('sideways_buy', 0.3),
            'sell': at.get('sideways_sell', -0.2),
            'confirm_days': base.get('confirm_days', 2),
            'weekly_filter_percentile': base.get('weekly_filter_percentile', 0.2),
        }


def decide(latest_features: dict, _historical_features=None, state_override=None, persist=True, config_override=None) -> dict:
    """状态机决策主逻辑

    Parameters
    ----------
    state_override : dict or None
        传入状态字典时回测模式，不读文件。None=正常模式读 state.json
    persist : bool
        是否将状态持久化到文件。回测模式=False
    config_override : dict or None
        传入配置字典时使用该配置，否则读 config.json
    """
    config = config_override if config_override is not None else load_config()
    state = load_state() if state_override is None else state_override

    # 自适应阈值
    thresholds = get_adaptive_thresholds(config, latest_features)
    buy_th = thresholds['buy']
    sell_th = thresholds['sell']
    confirm_days = thresholds.get('confirm_days', 2)

    # 连续周线调节分（替代二进制过滤）
    total_score = latest_features.get('total_score', 0)
    weekly_mod = latest_features.get('weekly_modifier', 0.0)
    adjusted_score = total_score + weekly_mod

    if adjusted_score != adjusted_score:  # NaN check
        adjusted_score = total_score

    current_state = state['state']
    waiting_days = state.get('waiting_days', 0)
    raw_date = latest_features.get('date', date.today())
    if hasattr(raw_date, 'strftime'):
        today_str = raw_date.strftime('%Y-%m-%d')
    else:
        today_str = str(raw_date)[:10]
    last_date = state.get('last_decision_date')
    if last_date is not None and not isinstance(last_date, str):
        last_date = str(last_date)[:10]

    # 不是新的交易日的决策不更新状态
    if last_date == today_str:
        print(f"  [INFO] 今日已决策过，跳过")
        result = _build_result(
            state['state'], total_score,
            '持有' if state['state'] == '持仓' else '等待',
            '今日已决策，沿用昨日',
            latest_features, config
        )
        if not persist:
            result['_new_state'] = state
        return result

    new_state = current_state
    action = '不动'
    explanation = ''

    # --- 状态机逻辑（使用 adjusted_score，已含周线调节分） ---

    # 空仓 → 持仓：调整后总分 > 买入阈值，即时切换
    if current_state == '空仓':
        if adjusted_score > buy_th:
            new_state = '持仓'
            action = '买入（试仓）'
            explanation = f'空仓转持仓：调整分 {adjusted_score:.2f} > {buy_th}'
        elif adjusted_score > sell_th:
            action = '等待'
            explanation = f'空仓观望：调整分 {adjusted_score:.2f} 未达买入阈值'
        else:
            action = '等待'
            explanation = f'空仓观望：市场偏空'

    # 持仓 → 空仓：需要连续确认
    elif current_state == '持仓':
        if adjusted_score < sell_th:
            waiting_days += 1
            if waiting_days >= confirm_days:
                new_state = '空仓'
                action = '卖出'
                explanation = f'持仓转空仓：连续 {confirm_days} 天信号为空'
                waiting_days = 0
            else:
                new_state = '持仓'  # 仍保持持仓
                action = '持有观察'
                explanation = f'信号偏空（第 {waiting_days}/{confirm_days} 天确认），暂持'
        elif adjusted_score < buy_th:
            # 持仓 → 观望
            new_state = '观望'
            action = '减仓观望'
            explanation = f'调整分 {adjusted_score:.2f} 回落至模糊区，减仓观察'
            waiting_days = 0
        else:
            new_state = '持仓'
            action = '持有'
            explanation = f'趋势良好，继续持有'
            waiting_days = 0

    # 观望 → 持仓 或 观望 → 空仓
    elif current_state == '观望':
        if adjusted_score > buy_th:
            new_state = '持仓'
            action = '买入加仓'
            explanation = f'观望转持仓：调整分 {adjusted_score:.2f} 回升'
            waiting_days = 0
        elif adjusted_score < sell_th:
            new_state = '空仓'
            action = '卖出'
            explanation = f'观望转空仓：调整分 {adjusted_score:.2f} 偏空'
            waiting_days = 0
        else:
            action = '继续观望'
            explanation = f'模糊区间，继续等待确认'

    # --- 仓位计算 ---
    def calc_position(state_val, score):
        if state_val == '空仓':
            return 0.0
        pos_score = max(0.0, min(score, 1.0))
        if pos_score > 0.7:
            return 0.7
        return round(0.3 + pos_score * 0.4, 2)

    position = calc_position(new_state, total_score)

    # --- 保存状态 ---
    state_update = {
        'state': new_state,
        'waiting_days': waiting_days,
        'last_decision_date': today_str,
    }
    if persist:
        save_state(state_update)

    result = _build_result(new_state, total_score, action, explanation,
                           latest_features, config, position)
    if not persist:
        result['_new_state'] = state_update
    return result


def _build_result(state: str, score: float, action: str, explanation: str,
                  features: dict, config: dict, position: float = 0) -> dict:
    return {
        'date': features.get('date', str(date.today())),
        'decision': state,
        'total_score': score,
        'action': action,
        'explanation': explanation,
        'position': f"{position * 100:.0f}%",
        'weekly_modifier': features.get('weekly_modifier', 0.0),
        'factors': {
            'momentum': features.get('momentum', 0),
            'trend': features.get('trend', 0),
            'volume_price': features.get('volume_price', 0),
            'rsrs': features.get('rsrs', 0),
            'relative_strength': features.get('relative_strength', 0),
        },
    }


def print_panel(result: dict, verbose: bool = True):
    """打印决策面板"""
    print(f"\n{'=' * 50}")
    date_str = str(result['date'])[:10]
    print(f"  588000 日线信号面板 — {date_str}")
    print(f"{'=' * 50}")

    # 周线调节分
    wm = result.get('weekly_modifier', 0.0)
    wm_str = f'{wm:+.2f}' if wm != 0 else '0.00'
    print(f"  周线调节: {wm_str} （修正总分 {result['total_score']:+.2f} → {result['total_score']+wm:+.2f}）")

    # 各因子
    f = result['factors']
    print(f"  ─────────────────────────────")
    print(f"  短期动量:     {f['momentum']:+.2f}  {factor_arrow(f['momentum'])} {factor_emoji(f['momentum'])}")
    print(f"  中期趋势:     {f['trend']:+.2f}  {factor_arrow(f['trend'])} {factor_emoji(f['trend'])}")
    print(f"  量价关系:     {f['volume_price']:+.2f}  {factor_arrow(f['volume_price'])} {factor_emoji(f['volume_price'])}")
    print(f"  RSRS:         {f['rsrs']:+.2f}  {factor_arrow(f['rsrs'])} {factor_emoji(f['rsrs'])}")

    print(f"  ─────────────────────────────")
    state = load_state()
    print(f"  综合得分:     {result['total_score']:+.2f}")
    print(f"  状态机:       {state.get('state', '?')} → {result['decision']}")
    print(f"  操作建议:     {result['action']} ({result['position']} 仓位)")
    print(f"  理由:         {result['explanation']}")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description='信号规则状态机决策')
    parser.add_argument('--reset', action='store_true', help='重置状态为空仓')
    parser.add_argument('--verbose', action='store_true', help='输出完整决策链')
    args = parser.parse_args()

    # 导入 feature 模块
    sys.path.insert(0, str(SCRIPT_DIR))
    from compute_features import load_features_cache, load_data, compute_all_features, load_config as feat_config

    config = load_config()
    symbol = config['symbol']

    if args.reset:
        save_state({'state': '空仓', 'waiting_days': 0, 'last_decision_date': None})
        print("  状态已重置为空仓")
        return

    # 确保数据是最新的
    df_sym = load_data(symbol)
    df_bench = load_data(config['benchmark'])
    features_df = compute_all_features(df_sym, df_bench, config)

    # 取最新特征
    latest = load_features_cache(symbol)
    if len(latest) == 0:
        print("  [ERR] 无特征数据，请先运行 compute_features.py")
        sys.exit(1)

    latest_row = latest.iloc[-1].to_dict()
    result = decide(latest_row, features_df)

    print_panel(result, args.verbose)

    return result


if __name__ == '__main__':
    main()
