#!/usr/bin/env bash
# trade-pulse 健康检查（14:30）— 纯 shell 实现，零 LLM 调用
# 行为：
#   - 全部健康 → 输出 NO_REPLY（cron 静默，不推送）
#   - 有异常（含冷却中的源过滤）→ 输出告警文本（推送飞书）+ exit 2
# 说明：akshare/eastmoney 处于冷却期是预期状态，不算故障（读 source_health.json 判断）
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJ" || { echo "⚠️ trade-pulse 健康检查：项目目录不存在"; exit 1; }

python3 - <<'PYEOF'
import json, subprocess, sys, time

# 1. 运行健康检查（捕获 TimeoutExpired / OSError，区分「脚本执行失败」与「真实故障」）
try:
    r = subprocess.run(
        ["python3", "tools/daily_pipeline/health_check.py", "--json"],
        capture_output=True, text=True, timeout=120,
    )
except subprocess.TimeoutExpired:
    print("⚠️ trade-pulse 健康检查脚本执行超时（>120s）")
    sys.exit(3)
except OSError as e:
    print("⚠️ trade-pulse 健康检查脚本无法启动: %s" % e)
    sys.exit(3)

# 2. 解析 JSON：逐个尝试 { 位置直到解析成功（health_check.py 输出前可能有 login/logout 噪音行）
text = r.stdout
d = None
for i, ch in enumerate(text):
    if ch != "{":
        continue
    try:
        d = json.loads(text[i:])
        break
    except Exception:
        continue
if d is None:
    print("⚠️ trade-pulse 健康检查输出解析失败（exit=%d）" % r.returncode)
    tail = (r.stderr or text).strip()[-2000:]
    if tail:
        print(tail)
    sys.exit(3)

# 3. 读取数据源冷却状态（文件缺失/损坏 → 警告但不静默，宁可如实报告）
now = time.time()
cooldown_sources = set()
try:
    sh = json.load(open("tools/daily_pipeline/source_health.json"))
    for symbol, sources in sh.items():
        for src, st in sources.items():
            cu = st.get("cooldown_until", 0) or 0
            if cu > now:
                cooldown_sources.add(src)
except Exception as e:
    print("ℹ️ 警告：source_health.json 读取失败（%s），冷却过滤未启用，可能误报冷却中的源" % e)

# 4. 收集失败项
problems = []
for name, item in d.items():
    if not isinstance(item, dict) or "ok" not in item:
        continue
    if item.get("ok"):
        continue
    # providers 子项：只报告非冷却中的源（防御：源条目必须是 dict）
    if name == "providers":
        for src, st in (item.get("sources") or {}).items():
            if not isinstance(st, dict):
                continue
            if not st.get("ok") and src not in cooldown_sources:
                problems.append("❌ %s: %s" % (src, st.get("msg", "")))
    else:
        problems.append("❌ %s: %s" % (name, item.get("msg", "")))

# 5. 输出
if problems:
    print("⚠️ trade-pulse 健康检查异常")
    for p in problems:
        print(p)
    # 附加冷却中的源备注（信息性，不算故障）
    if cooldown_sources:
        print("ℹ️ 冷却中（预期跳过）: %s" % ", ".join(sorted(cooldown_sources)))
    sys.exit(2)
else:
    print("NO_REPLY")
    sys.exit(0)
PYEOF
exit $?
