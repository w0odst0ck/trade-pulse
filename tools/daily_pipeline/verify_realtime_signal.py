#!/usr/bin/env python3
"""
verify_realtime_signal.py — 方案 B 信号稳定性验证（阶段 3）

目标：量化「14:45 盘中实时信号」 vs 「收盘信号」的状态机决策差异。
这决定方案 B 的实盘可信度：盘中预览/尾盘确认的信号，和收盘后正式信号，
翻转率越低越可信。

方法（复用 execution_timing 的盘中特征构造，不改生产代码）：
  1. 构造两条特征序列：
     - close 口径：生产 features_cache（收盘特征）
     - intraday 口径：min15 截至 14:45 聚合 bar 替换当日行（14:50 尾盘确认
       时最后可见完整 bar 是 14:45 结束那根；cutoff=1450 排除 14:45 本身）
  2. 两条序列分别跑状态机（decide 逐日，各自独立演进状态）
  3. 对比覆盖区间内每日决策：
     - 决策翻转：盘中决策 != 收盘决策（持仓/观望/空仓 状态或操作建议不同）
     - 方向性翻转：买卖方向相反（盘中说买/加，收盘说卖/清，反之亦然）——最危险
     - 输出翻转率 + 明细 CSV

用法：
  python3 verify_realtime_signal.py [--cutoff 1450] [--json]
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from execution_timing import (
    build_intraday_feature_series,
    load_benchmark,
    load_min15,
    load_prod_features,
    load_csv,
)
from signal_rules import decide
from backtest import load_config

# 与生产一致
START = '2026-01-05'  # min15 覆盖起点
CUTOFF_DEFAULT = 1450  # 14:50 尾盘确认时可见 14:45 结束的 bar（hhmm < 1450）

# 方向性操作（买卖方向相反 = 最危险翻转）
BULL_ACTIONS = {'买入（试仓）', '买入加仓', '加仓'}
BEAR_ACTIONS = {'卖出', '清仓', '减仓观望'}


def run_state_machine(features_df: pd.DataFrame, config: dict,
                      start: str) -> list:
    """逐日跑状态机，返回 [(date, decision, action, score), ...]"""
    df = features_df[(features_df['date'] >= pd.Timestamp(start))]
    df = df.sort_values('date').reset_index(drop=True)
    state = {'state': '空仓', 'waiting_days': 0, 'last_decision_date': None}
    rows = []
    for _, row in df.iterrows():
        feat = row.to_dict()
        result = decide(
            feat, features_df,
            state_override=state, persist=False, config_override=config,
        )
        state = result['_new_state']
        rows.append({
            'date': str(row['date'])[:10],
            'decision': result['decision'],
            'action': result['action'],
            'score': round(float(result['total_score']), 4),
        })
    return rows


def action_direction(action: str) -> str:
    """操作方向：bull / bear / hold（用于方向性翻转判定）"""
    if action in BULL_ACTIONS:
        return 'bull'
    if action in BEAR_ACTIONS:
        return 'bear'
    return 'hold'


def compare(close_rows: list, intraday_rows: list) -> dict:
    """对比两条决策序列 → 翻转统计"""
    by_date = {r['date']: r for r in close_rows}
    stats = {'total': 0, 'state_flip': 0, 'action_flip': 0, 'direction_flip': 0}
    details = []
    for r in intraday_rows:
        c = by_date.get(r['date'])
        if c is None:
            continue
        stats['total'] += 1
        d = {
            'date': r['date'],
            'close_decision': c['decision'], 'rt_decision': r['decision'],
            'close_action': c['action'], 'rt_action': r['action'],
            'close_score': c['score'], 'rt_score': r['score'],
        }
        state_flip = c['decision'] != r['decision']
        action_flip = c['action'] != r['action']
        dir_flip = (action_direction(c['action']) == 'bull'
                    and action_direction(r['action']) == 'bear') or \
                   (action_direction(c['action']) == 'bear'
                    and action_direction(r['action']) == 'bull')
        if state_flip:
            stats['state_flip'] += 1
        if action_flip:
            stats['action_flip'] += 1
        if dir_flip:
            stats['direction_flip'] += 1
            d['DIR_FLIP'] = True
        if state_flip or action_flip:
            details.append(d)
    stats['state_flip_rate'] = round(stats['state_flip'] / stats['total'], 4) if stats['total'] else 0
    stats['action_flip_rate'] = round(stats['action_flip'] / stats['total'], 4) if stats['total'] else 0
    stats['direction_flip_rate'] = round(stats['direction_flip'] / stats['total'], 4) if stats['total'] else 0
    stats['details'] = details
    return stats


def main():
    parser = argparse.ArgumentParser(description='方案 B 信号稳定性验证')
    parser.add_argument('--cutoff', type=int, default=CUTOFF_DEFAULT,
                        help='盘中截止 HHMM（默认 1450 = 14:45 bar 可见）')
    parser.add_argument('--json', action='store_true', help='JSON 输出')
    args = parser.parse_args()

    config = load_config()
    symbol = config['symbol']

    print(f"\n🎯 方案 B 信号稳定性验证（cutoff={args.cutoff}，区间 {START} ~ 数据末日）")
    print("=" * 60)

    daily = load_csv(PROJECT_ROOT / config['data_dir'] / symbol / 'daily.csv')
    min15 = load_min15(config)
    bench = load_benchmark(config)

    # 收盘口径特征（生产缓存）
    close_feat = load_prod_features(config)
    # 盘中口径特征（cutoff 时点 bar 替换当日）
    intraday_feat = build_intraday_feature_series(
        daily, min15, bench, config, cutoff_hhmm=args.cutoff
    )

    print(f"  收盘特征: {len(close_feat)} 行 (至 {close_feat['date'].max()})")
    print(f"  盘中特征: {len(intraday_feat)} 行 (至 {intraday_feat['date'].max()})")

    # 两条独立状态机演进
    close_rows = run_state_machine(close_feat, config, START)
    intraday_rows = run_state_machine(intraday_feat, config, START)

    stats = compare(close_rows, intraday_rows)

    print(f"\n  ── 覆盖交易日 {stats['total']} 天 ──")
    print(f"  状态翻转（持仓/观望/空仓 不同）: {stats['state_flip']} 天 "
          f"({stats['state_flip_rate'] * 100:.1f}%)")
    print(f"  操作翻转（action 不同）:        {stats['action_flip']} 天 "
          f"({stats['action_flip_rate'] * 100:.1f}%)")
    print(f"  方向翻转（买卖相反，最危险）:   {stats['direction_flip']} 天 "
          f"({stats['direction_flip_rate'] * 100:.1f}%)")

    if stats['direction_flip'] > 0:
        print(f"\n  ⚠️ 方向翻转明细（盘中 vs 收盘 买卖相反）：")
        for d in stats['details']:
            if d.get('DIR_FLIP'):
                print(f"    {d['date']}: 盘中 {d['rt_action']}({d['rt_score']:+.2f}) "
                      f"vs 收盘 {d['close_action']}({d['close_score']:+.2f})")

    if stats['action_flip'] > 0 and stats['direction_flip'] == 0:
        print(f"\n  非方向性操作差异明细（状态/仓位微调，非买卖反转）：")
        for d in stats['details'][:15]:
            print(f"    {d['date']}: 盘中 {d['rt_decision']}/{d['rt_action']} "
                  f"vs 收盘 {d['close_decision']}/{d['close_action']}")

    # 保存明细
    out_dir = PROJECT_ROOT / 'data' / symbol / 'backtest'
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / 'realtime_signal_verify.csv'
    if stats['details']:
        pd.DataFrame(stats['details']).to_csv(detail_path, index=False)
        print(f"\n  差异明细: {detail_path}")

    summary = {k: v for k, v in stats.items() if k != 'details'}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
