#!/usr/bin/env bash
# catchup_signal.sh — 14:50 confirm 失败后的收盘补决策（方案 B 稳定性闭环 C）
#
# 场景：当天 14:50 尾盘确认失败（实时源挂/任务异常）→ state 未更新 →
#       用户当天没有信号。16:30 收盘数据到位后，用收盘口径补算决策并推送，
#       保证每个交易日都有信号兜底，不空窗。
#
# 触发：close_catchup.sh（16:30）在收盘数据到位后调用。
# 行为：
#   - state.last_decision_date == 今天 → 14:50 confirm 已成功，跳过（正常）
#   - 否则 → 收盘口径补决策（写 state，signal_mode=close）+ 推送「⚠️ 补决策」
# 幂等：补决策后 state 更新为今天，重复调用自动跳过。
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ trade-pulse 补决策：项目目录不存在"; exit 1; }

FEISHU_APP_SECRET=$(sed -n 's/^export FEISHU_APP_SECRET="\(.*\)"/\1/p' ~/.bashrc)
if [ -z "$FEISHU_APP_SECRET" ]; then
  echo "⚠️ trade-pulse 补决策：FEISHU_APP_SECRET 提取为空，中止"
  exit 1
fi
export FEISHU_APP_SECRET

TODAY=$(date +%F)

# 检查 state 是否今日已决策
STATE=data/588000/state.json
if [ -f "$STATE" ]; then
  LAST_DEC=$(python3 -c "import json; print(json.load(open('$STATE')).get('last_decision_date',''))" 2>/dev/null)
  if [ "$LAST_DEC" == "$TODAY" ]; then
    echo "NO_REPLY"  # 14:50 confirm 已成功，无需补
    exit 0
  fi
else
  echo "⚠️ trade-pulse 补决策：state.json 不存在，跳过（首次运行？）"
  exit 0
fi

echo "⚠️ trade-pulse 补决策：今日 14:50 确认未执行（state 最后决策 $LAST_DEC），用收盘数据补决策..."

# 收盘口径补决策（daily_panel 默认路径：fetch→features→decide 写 state→推送）
# 用 --skip-fetch：16:30 刚补拉过数据，避免重复拉取
python3 tools/daily_pipeline/daily_panel.py --skip-fetch --push
RC=$?
if [ $RC -ne 0 ]; then
  echo "⚠️ trade-pulse 补决策失败（exit=$RC）"
  exit $RC
fi

# 校验补决策是否真正写入 state（非交易日/逻辑异常时 last_decision_date 可能未更新）
NEW_DEC=$(python3 -c "import json; print(json.load(open('$STATE')).get('last_decision_date',''))" 2>/dev/null)
if [ "$NEW_DEC" != "$TODAY" ]; then
  echo "⚠️ trade-pulse 补决策未生效：state 最后决策仍是 $NEW_DEC（今日 $TODAY）"
  # 推失败告警（区别于成功通知，不误导用户以为有信号）
  # 密钥已由 shell 导出（FEISHU_APP_SECRET），python 直接读环境变量，不再重复解析
  python3 - "$TODAY" "$NEW_DEC" <<'PYEOF'
import sys, os

sys.path.insert(0, "tools/daily_pipeline")
if not os.environ.get("FEISHU_APP_SECRET"):
    print("  [WARN] FEISHU_APP_SECRET 未导出，跳过失败告警推送")
    sys.exit(0)
from feishu_push import push_text
try:
    push_text(
        f"⚠️ [trade-pulse] {sys.argv[1]} 收盘补决策未生效（state 最后决策 {sys.argv[2]}），"
        f"请人工检查状态机与日志。"
    )
    print("  [PUSH] 补决策失败告警已推送")
except Exception as e:
    print(f"  [WARN] 失败告警推送异常: {e}")
PYEOF
  exit 1
fi

# 补决策成功后推送说明（与正常信号卡区分）
python3 - "$TODAY" <<'PYEOF'
import sys, os

sys.path.insert(0, "tools/daily_pipeline")
if not os.environ.get("FEISHU_APP_SECRET"):
    print("  [WARN] FEISHU_APP_SECRET 未导出，跳过说明推送")
    sys.exit(0)
from feishu_push import push_text
try:
    push_text(
        f"⚠️ [trade-pulse] {sys.argv[1]} 尾盘实时确认未执行（链路异常），"
        f"已用收盘数据补决策（见上方信号卡）。今日信号为收盘口径。"
    )
    print("  [PUSH] 补决策说明已推送")
except Exception as e:
    print(f"  [WARN] 补决策说明推送失败: {e}")
PYEOF

echo "✅ trade-pulse 收盘补决策完成"
exit 0
