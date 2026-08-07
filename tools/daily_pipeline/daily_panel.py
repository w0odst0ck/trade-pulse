#!/usr/bin/env python3
"""
daily_panel.py — 每日信号面板（统一入口）

每天收盘后跑这个：
  python daily_panel.py

全流程：数据拉取 → 特征计算 → 信号决策 → 面板输出
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

# 把 SCRIPT_DIR 加入路径，方便 import 同目录模块
sys.path.insert(0, str(SCRIPT_DIR))


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='588000 每日信号面板')
    parser.add_argument('--skip-fetch', action='store_true', help='跳过数据拉取（用缓存）')
    parser.add_argument('--force-fetch', action='store_true', help='强制全量拉取数据')
    parser.add_argument('--force-features', action='store_true', help='全量重算特征')
    parser.add_argument('--json', action='store_true', help='以 JSON 格式输出（供程序消费）')
    parser.add_argument('--reset', action='store_true', help='重置状态机为空仓')
    parser.add_argument('--push', action='store_true', help='推送结果到飞书')
    args = parser.parse_args()

    config = load_config()
    symbol = config['symbol']

    print(f"\n🔧 588000 每日信号面板 — {config['provider']}")
    print("=" * 45)

    # --- 1. 数据 ---
    if not args.skip_fetch:
        from fetch_data import fetch_data
        df_sym = fetch_data(symbol, "2023-01-01", args.force_fetch)
        df_bench = fetch_data(config['benchmark'], "2023-01-01", args.force_fetch)
    else:
        from compute_features import load_data
        df_sym = load_data(symbol)
        df_bench = load_data(config['benchmark'])
        print("  [SKIP] 跳过数据拉取")

    # --- 2. 特征 ---
    from compute_features import compute_all_features, load_features_cache
    features_df = compute_all_features(df_sym, df_bench, config, args.force_features)

    # --- 3. 信号 ---
    from signal_rules import decide, print_panel, save_state

    if args.reset:
        save_state({'state': '空仓', 'waiting_days': 0, 'last_decision_date': None})
        print("  [RESET] 状态已重置为空仓\n")
        return

    latest_features = load_features_cache(symbol)
    if len(latest_features) == 0:
        print("  [ERR] 特征数据为空")
        sys.exit(1)

    latest_row = latest_features.iloc[-1].to_dict()
    result = decide(latest_row, features_df)

    # --- 4. 输出 ---
    if args.json:
        # result 的 date 来自 features 可能是 Timestamp，需先转 str（否则 json.dumps 崩）
        out = dict(result)
        out['date'] = str(out.get('date', ''))
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_panel(result, verbose=True)

    # 推送
    if args.push:
        from datetime import date
        from trading_calendar import is_trading_day
        from feishu_push import push_signal_card
        today = date.today()
        if not is_trading_day(today):
            print(f"  [SKIP] {today} 不是交易日，跳过推送")
        else:
            push_signal_card(result)


if __name__ == '__main__':
    main()
