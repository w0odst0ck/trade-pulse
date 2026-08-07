#!/usr/bin/env bash
# 信号接近翻多预警（14:25 信号任务内调用）— 纯 shell，零 LLM
# 行为：解析 daily_panel --json 的综合分，若 0.05 <= score < 0.1 且空仓/等待
#       → 推飞书文本预警「接近翻多」；其余情况静默（无输出）
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || exit 1

OUT=$(python3 tools/daily_pipeline/daily_panel.py --skip-fetch --json 2>/dev/null)
RC=$?
[ $RC -ne 0 ] && exit 0  # 面板失败不预警（主任务会报错）

# 提取综合分 / 决策：取第一个 { 到末尾完整解析（--json 是多行 indent 输出）
# 防御：若噪音行含 { 导致解析失败，尝试后续 { 位置直到成功
PARSE=$(echo "$OUT" | python3 -c "
import sys, json
text = sys.stdin.read()
for i, ch in enumerate(text):
    if ch != '{':
        continue
    try:
        d = json.loads(text[i:])
    except Exception:
        continue
    print(d.get('total_score', 'nan'))
    print(d.get('decision', ''))
    break
" 2>/dev/null)
SCORE=$(echo "$PARSE" | sed -n '1p')
DECISION=$(echo "$PARSE" | sed -n '2p')

# 数值比较：score >= 0.05 且 < 0.1，且处于空仓/等待（持仓时不需要翻多预警）
WARN=$(python3 - "$SCORE" "$DECISION" <<'PYEOF'
import sys
try:
    score = float(sys.argv[1])
except (ValueError, TypeError):
    sys.exit(0)
decision = sys.argv[2] if len(sys.argv) > 2 else ""
if 0.05 <= score < 0.10 and decision in ("空仓", "等待"):
    print("1")
PYEOF
)

if [ "$WARN" = "1" ]; then
  # 推预警文本（FEISHU_SECRET 由 python 从 ~/.bashrc 提取，绕开 shell $() 脱敏问题）
  python3 - "$SCORE" "$DECISION" <<'PYEOF'
import sys, os, re, json, subprocess
from pathlib import Path

# 提取 FEISHU_APP_SECRET（同 daily_signal.sh 逻辑）
secret = ""
bashrc = Path.home() / ".bashrc"
if bashrc.exists():
    m = re.search(r'^export FEISHU_APP_SECRET="([^"]*)"', bashrc.read_text(encoding="utf-8"), re.M)
    if m:
        secret = m.group(1)
if not secret:
    print("  [WARN-FAIL] FEISHU_APP_SECRET 提取为空，跳过预警")
    sys.exit(0)

sys.path.insert(0, 'tools/daily_pipeline')
os.environ['FEISHU_APP_SECRET'] = secret
from feishu_push import push_text

score = sys.argv[1]
decision = sys.argv[2]
msg = (
    f"🟡 信号接近翻多预警\n"
    f"综合得分 {score}（翻多阈值 0.1），当前状态 {decision}\n"
    f"还差一点点就触发建仓信号，未来几天留意 14:25 信号卡"
)
try:
    push_text(msg)
    print("  [WARN-PUSH] 接近翻多预警已推送")
except Exception as e:
    print(f"  [WARN-FAIL] 预警推送失败: {e}")
PYEOF
fi
exit 0
