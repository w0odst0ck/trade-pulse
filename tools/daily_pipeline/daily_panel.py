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
    parser.add_argument('--realtime', action='store_true',
                        help='盘中实时模式：拉实时 bar 拼今日特征（默认 preview 不写 state）')
    parser.add_argument('--realtime-confirm', action='store_true',
                        help='盘中实时 + 确认模式：拉实时 bar 拼今日特征并写 state（14:50 尾盘执行信号）')
    args = parser.parse_args()

    config = load_config()
    symbol = config['symbol']

    print(f"\n🔧 588000 每日信号面板 — {config['provider']}")
    print("=" * 45)

    # --- 0. 实时模式入口（盘中预览 / 尾盘确认） ---
    if args.realtime or args.realtime_confirm:
        run_realtime(args, config, symbol)
        return

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

    # --- 3.5 链路可信度 → 建议仓位（实盘风控） ---
    _attach_link_confidence(result, symbol)

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


def _parse_position(result: dict) -> float:
    """解析 result['position']（形如 "60%"）为小数，失败返回 0.0"""
    try:
        pos_str = str(result.get('position', '0%')).replace('%', '').strip()
        return float(pos_str) / 100.0
    except (ValueError, TypeError):
        return 0.0


def _attach_link_confidence(result: dict, symbol: str) -> dict:
    """附加链路可信度 → 建议仓位（实盘风控：信号仓位 × 链路乘数）

    探测降级/数据陈旧时自动打折，避免基于不可靠数据的重仓决策。
    只影响建议，不写 state（状态机照常，执行权在用户）。
    """
    try:
        from link_health import get_link_confidence, apply_multiplier
        confidence = get_link_confidence(symbol)
        signal_pos = _parse_position(result)
        adj = apply_multiplier(signal_pos, confidence)
        result['link_confidence'] = {
            'level': adj['level'],
            'emoji': adj['emoji'],
            'multiplier': adj['multiplier'],
            'reason': adj['reason'],
            'stale_days': adj['stale_days'],
        }
        result['advised_position'] = f"{adj['advised_position'] * 100:.0f}%"
    except Exception as e:
        # 附加失败不阻塞信号，但**保守降级**（满信任违背风控目标）：
        # 链路模块异常 = 无法确认数据质量 → 按 degraded 0.75 打折并标明原因
        result['link_confidence'] = {'level': 'degraded', 'emoji': '🟡', 'multiplier': 0.75,
                                     'reason': f'链路评估异常(保守降级): {type(e).__name__}', 'stale_days': None}
        sig = _parse_position(result)
        result['advised_position'] = f"{round(sig * 0.75, 2) * 100:.0f}%"
    return result


def run_realtime(args, config: dict, symbol: str) -> dict:
    """盘中实时模式（方案 B）：历史日线 + 实时 bar 拼接 → 实时特征 → 状态机决策

    - preview（--realtime）：算信号不写 state，推送「🟡 盘中预览」
    - confirm（--realtime-confirm）：算信号写 state，推送「🟢 尾盘执行信号」
    - 实时源失败 → 回退收盘口径（用昨日收盘特征，注明兜底）

    返回 result dict（含 signal_mode / signal_data_date 字段）。
    """
    import json as _json
    from datetime import date

    from compute_features import (
        compute_realtime_features, load_data,
    )
    from realtime_quote import fetch_realtime_bar
    from signal_rules import decide, print_panel
    from trading_calendar import is_trading_day

    is_confirm = bool(args.realtime_confirm)
    print(f"  [RT] 实时模式：{'confirm（写 state）' if is_confirm else 'preview（不写 state）'}")

    today = date.today()
    if not is_trading_day(today):
        print(f"  [SKIP] {today} 不是交易日，跳过实时信号")
        return {'decision': '非交易日', 'action': '跳过', 'signal_mode': 'realtime'}

    # 昨收（sanity check 基准）：取日线最后一行 close
    try:
        df_hist = load_data(symbol)
        prev_close = float(df_hist['close'].iloc[-1]) if len(df_hist) else None
    except Exception as e:
        print(f"  [WARN] 读取历史日线失败: {e}")
        prev_close = None

    # 拉实时 bar（588000 + 基准 000688，双源互备 + sanity + 时段守卫）
    bar_sym = fetch_realtime_bar(symbol, prev_close=prev_close)
    bar_bench = fetch_realtime_bar(config['benchmark'], prev_close=None)

    # --- 兜底：实时源不可用 → 回退收盘口径 ---
    if bar_sym is None:
        print("  ⚠️ 实时行情不可用，回退收盘口径（昨日收盘特征 + 不写 state）")
        return _fallback_close(args, config, symbol, reason='实时行情不可用')

    # 实时特征（历史 + 实时 bar 拼接，不落盘）；计算异常也走兜底（不崩溃、不写 state）
    try:
        features_df = compute_realtime_features(
            symbol, config['benchmark'], bar_sym, bar_bench, config
        )
    except Exception as e:
        print(f"  [WARN] 实时特征计算异常: {type(e).__name__}: {e}")
        return _fallback_close(args, config, symbol, reason=f'实时特征计算失败: {type(e).__name__}')
    if len(features_df) == 0:
        print("  [ERR] 实时特征计算为空")
        return _fallback_close(args, config, symbol, reason='实时特征为空')

    latest_row = features_df.iloc[-1].to_dict()
    data_date = str(latest_row.get('date', ''))[:10]

    # 状态机决策（preview 不写 state；confirm 写 state + signal_mode=realtime）；
    # 决策异常也走兜底（实时链路任何一步失败都不能带病写 state）
    try:
        result = decide(
            latest_row, features_df,
            persist=is_confirm,
            signal_mode='realtime',
            signal_data_date=data_date,
        )
    except Exception as e:
        print(f"  [WARN] 实时决策异常: {type(e).__name__}: {e}")
        return _fallback_close(args, config, symbol, reason=f'实时决策失败: {type(e).__name__}')
    result['signal_mode'] = 'realtime'
    result['signal_data_date'] = data_date
    result['is_preview'] = not is_confirm

    # 链路可信度 → 建议仓位（实盘风控：实时链路降级/数据陈旧时打折）
    _attach_link_confidence(result, symbol)

    # 输出
    if args.json:
        out = dict(result)
        out['date'] = str(out.get('date', ''))
        print(_json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_panel(result, verbose=True)

    # confirm 时把实时信号持久化到 realtime_signal.json（供 UI dashboard 展示实时口径
    # 的 score/factors；收盘后 features 更新自动回落收盘口径）
    if is_confirm:
        _persist_realtime_signal(result, config, symbol)

    # 推送（交易日才推）
    if args.push:
        from feishu_push import push_signal_card
        if is_confirm:
            push_signal_card(result, preview=False)
        else:
            push_signal_card(result, preview=True)

    return result


def _persist_realtime_signal(result: dict, config: dict, symbol: str) -> None:
    """把实时确认信号写入 data/{symbol}/realtime_signal.json（UI 展示用）"""
    import os
    from pathlib import Path as P
    data_dir = PROJECT_ROOT / config['data_dir'] / symbol
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / 'realtime_signal.json'
    payload = {
        'date': str(result.get('date', ''))[:10],
        'signal_data_date': str(result.get('signal_data_date', ''))[:10],
        'state': result.get('decision', '?'),
        'action': result.get('action', '?'),
        'total_score': result.get('total_score', 0),
        'weekly_modifier': result.get('weekly_modifier', 0.0),
        'signal_mode': result.get('signal_mode', 'realtime'),
        'factors': {k: result.get('factors', {}).get(k, 0)
                    for k in ['momentum', 'trend', 'volume_price', 'rsrs']},
    }
    try:
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(f"  [SAVE] 实时信号已写入 {path}")
    except OSError as e:
        print(f"  [WARN] 写入 realtime_signal.json 失败: {e}")


def _fallback_close(args, config: dict, symbol: str, reason: str) -> dict:
    """实时源失败兜底：用昨日收盘特征跑收盘口径决策，不写 state

    实时信号不可信时绝不能写 state（避免用旧数据污染状态机），
    无论 preview/confirm 都 persist=False，只推送「实时源不可用」告警。
    """
    import json as _json

    from compute_features import load_data, load_features_cache, compute_all_features
    from signal_rules import decide, print_panel

    print(f"  ⚠️ [FALLBACK] {reason} → 收盘口径决策（只读，不写 state）")

    try:
        df_sym = load_data(symbol)
        df_bench = load_data(config['benchmark'])
        features_df = compute_all_features(df_sym, df_bench, config, persist=False)
        latest = load_features_cache(symbol)
        if len(latest) == 0:
            latest = features_df
        latest_row = latest.iloc[-1].to_dict()
        result = decide(
            latest_row, features_df,
            persist=False,          # 兜底不写 state（实时失败当天不更新状态机）
            signal_mode='close',
            signal_data_date=str(latest_row.get('date', ''))[:10],
        )
        result['fallback'] = reason
        result['signal_mode'] = 'close'
    except Exception as e:
        print(f"  [ERR] 兜底决策失败: {e}")
        result = {'decision': '未知', 'action': '无法决策', 'total_score': 0,
                  'signal_mode': 'close', 'fallback': f'{reason}; 兜底也失败: {e}'}

    if args.json:
        out = dict(result)
        out['date'] = str(out.get('date', ''))
        print(_json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_panel(result, verbose=True)

    if args.push:
        try:
            from feishu_push import push_text
            push_text(
                f"⚠️ [trade-pulse] 实时信号不可用（{reason}），"
                f"今日推送为收盘口径（{result.get('decision', '?')} / {result.get('action', '?')}）"
            )
        except Exception as e:
            print(f"  [WARN] 兜底告警推送失败: {e}")

    return result


if __name__ == '__main__':
    main()
