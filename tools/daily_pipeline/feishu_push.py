#!/usr/bin/env python3
"""
feishu_push.py — 飞书推送模块

用法：
  python feishu_push.py <signal.json>        # 从 JSON 文件推送
  python feishu_push.py --stdin               # 从 stdin 读 JSON 推送
  python feishu_push.py --test                # 发测试消息

依赖：无（纯 requests）
"""

import json
import sys
import os
import time
from pathlib import Path

import requests

# WyrmGate 凭证
# APP_ID 从环境变量读取，或 fallback 到 TOOLS.md 中的值
# APP_SECRET 必须走环境变量 FEISHU_APP_SECRET（已在 ~/.bashrc 中设置）
APP_ID = os.environ.get("FEISHU_APP_ID", "cli_aac181b732781bb6")
APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
if not APP_SECRET:
    raise RuntimeError("FEISHU_APP_SECRET 环境变量未设置")
API_BASE = "https://open.feishu.cn/open-apis"
CHAT_ID = "oc_0c5546a611fd44d8d0930cd5ea0bacd1"  # study-vault 同步群

TOKEN_CACHE_PATH = Path(__file__).parent / ".feishu_token.json"


def get_tenant_token() -> str:
    """获取 tenant_access_token，带缓存"""
    # 尝试读缓存
    if TOKEN_CACHE_PATH.exists():
        try:
            with open(TOKEN_CACHE_PATH) as f:
                cached = json.load(f)
            if cached.get('expire_at', 0) > time.time():
                return cached['token']
        except Exception:
            pass

    resp = requests.post(
        f"{API_BASE}/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"获取 token 失败: {data}")

    token = data["tenant_access_token"]
    expire_in = data.get("expire", 7200)

    # 缓存（提前 5 分钟过期）
    with open(TOKEN_CACHE_PATH, "w") as f:
        json.dump({
            "token": token,
            "expire_at": time.time() + expire_in - 300,
        }, f)

    return token


def push_signal_card(result: dict, to_chat_id: str = CHAT_ID):
    """推送信号卡片到飞书群"""
    token = get_tenant_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    # 构建富文本卡片
    date_str = str(result.get('date', ''))[:10]
    decision = result.get('decision', '?')
    action = result.get('action', '?')
    position = result.get('position', '?')
    score = result.get('total_score', 0)
    explanation = result.get('explanation', '')
    weekly_mod = result.get('weekly_modifier', 0.0)
    factors = result.get('factors') or {}  # None 保护（default 只在 key 缺失时生效）

    # 因子表情映射
    def factor_indicator(val):
        if val > 0.3: return '🟢'
        if val > 0: return '🟢'
        if val > -0.3: return '⚪'
        return '🔴'
    def factor_bar(val):
        bars = 10
        filled = max(0, min(bars, int((val + 1) * bars / 2)))
        return '█' * filled + '░' * (bars - filled)

    factor_lines = []
    factor_names = {
        'momentum': '短期动量', 'trend': '中期趋势',
        'volume_price': '量价关系', 'rsrs': 'RSRS',
    }
    # 只显示 result 中实际存在的因子（config 删掉的因子不展示）
    for key, name in factor_names.items():
        if key not in factors:
            continue
        val = factors.get(key, 0)
        if val is None:
            val = 0
        factor_lines.append(
            f"  {name}  {factor_indicator(val)} {factor_bar(val)}  {val:+.2f}"
        )

    factor_text = "\n".join(factor_lines)

    # 决策表情
    decision_emoji = {
        '空仓': '🈳',
        '持仓': '📈',
        '观望': '👀',
    }.get(decision, '📊')

    card_content = (
        f"📊 trade-pulse | {date_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"标的：588000 科创50ETF\n"
        f"决策：{decision_emoji} {decision}\n"
        f"操作：{action}\n"
        f"仓位建议：{position}\n"
        f"综合分：{score:+.2f}（周线调节 {weekly_mod:+.2f}）\n"
        f"\n"
        f"因子状态：\n"
        f"{factor_text}\n"
        f"\n"
        f"📝 {explanation}"
    )

    # 飞书消息体（富文本）
    message = {
        "receive_id": to_chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": card_content}),
    }

    resp = requests.post(
        f"{API_BASE}/im/v1/messages?receive_id_type=chat_id",
        headers=headers,
        json=message,
        timeout=15,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"飞书推送失败: {data}")

    print(f"  [PUSH] 信号已推送到 study-vault 群 ✅")
    return data


def push_multi_panel(results: list, to_chat_id: str = CHAT_ID):
    """推送多标的扫描面板
    results: [各标的的 result dict]
    """
    token = get_tenant_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    lines = ["📊 trade-pulse 多标的扫描", "━━━━━━━━━━━━━━━━━━━━"]

    for i, r in enumerate(results):
        symbol = r.get('symbol', f'标的{i+1}')
        decision = r.get('decision', '?')
        action = r.get('action', '?')
        position = r.get('position', '?')
        score = r.get('total_score', 0)
        emoji = {'空仓': '🈳', '持仓': '📈', '观望': '👀'}.get(decision, '📊')

        lines.append(f"\n{symbol}")
        lines.append(f"  {emoji} {decision} | 分数 {score:+.2f} | 仓位 {position}")
        lines.append(f"  操作: {action}")
        lines.append(f"  ─────────────────────")

    # 找最优
    if results:
        best = max(results, key=lambda r: r.get('total_score', -999))
        lines.append(f"\n🏆 今日最优: {best.get('symbol', '?')} ({best.get('total_score', 0):+.2f})")

    card_text = "\n".join(lines)

    message = {
        "receive_id": to_chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": card_text}),
    }

    resp = requests.post(
        f"{API_BASE}/im/v1/messages?receive_id_type=chat_id",
        headers=headers,
        json=message,
        timeout=15,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"推送失败: {data}")

    print(f"  [PUSH] 多标的扫描已推送 ✅")
    return data


def test_push():
    """发送测试消息"""
    result = {
        'date': '2026-07-31',
        'decision': '空仓',
        'action': '等待',
        'position': '0%',
        'total_score': -0.78,
        'explanation': '各因子偏空，等待',
        'weekly_modifier': 0.0,
        'factors': {
            'momentum': 0.12,
            'trend': 0.18,
            'volume_price': -0.30,
            'rsrs': -0.25,
            'relative_strength': -0.35,
        },
    }
    print("  [TEST] 发送测试信号...")
    push_signal_card(result)
    print("  [TEST] 发送完毕 ✅")


if __name__ == '__main__':
    if '--test' in sys.argv:
        test_push()
    elif '--stdin' in sys.argv:
        data = json.load(sys.stdin)
        push_signal_card(data)
    elif len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1]) as f:
            data = json.load(f)
        push_signal_card(data)
    else:
        print("用法:")
        print("  python feishu_push.py <signal.json>    # 从文件推送")
        print("  python feishu_push.py --stdin          # 从 stdin 读 JSON")
        print("  python feishu_push.py --test           # 发测试消息")
