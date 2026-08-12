#!/usr/bin/env bash
# trade-pulse 实盘数据源探测（关键时点前置）— 纯 shell 实现，零 LLM 调用
# 时点：09:05(盘前) / 14:20(预览前) / 14:45(确认前) / 15:25(增量前) / 16:25(补齐前)
# 行为：
#   - 全绿 → NO_REPLY（cron 静默）
#   - 源挂 → 按 link_health.evaluate_alert 决策推送飞书告警
#     （severity=emergency 首次即推；normal 连续 2 次；blind 连续 3 次；
#      持续挂不重复轰炸，恢复后重置——滞回统一在 link_health 状态机）
#   - probe_runner 自身崩溃 → exit 1（failureAlert 兜底，不误报数据源）
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJ" || { echo "⚠️ trade-pulse 探测：项目目录不存在"; exit 1; }

MODE="${1:-intraday}"   # morning / intraday / after_close
SYMBOL="${2:-588000}"

LOCK_FILE=/tmp/trade-pulse-probe.lock
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "NO_REPLY"; exit 0; }   # 已有探测在跑则跳过

OUT=$(python3 tools/daily_pipeline/probe_runner.py --symbol "$SYMBOL" --mode "$MODE" 2>&1)
RC=$?

# 非交易日 → 静默
if echo "$OUT" | grep -q "NO_REPLY"; then
  echo "NO_REPLY"
  exit 0
fi

# 全绿 → 静默（probe_runner 已写 probe_health.json；link_health 状态机在下次
# evaluate_alert 时自动把 ok_streak 累计，连续 2 次恢复才升回 full）
if echo "$OUT" | grep -q "PROBE_OK"; then
  echo "NO_REPLY"
  exit 0
fi

# ⚠️ probe_runner 自身崩溃（traceback/import 错误/OOM）：
# RC≠0 且无 [FAIL] 行 = 脚本故障（非数据源故障）→ 走 failureAlert 兜底；
# 真源故障（RC=1 + 有 [FAIL]）→ 落到下方 evaluate_alert 决策。
if [ "$RC" -ne 0 ] && ! echo "$OUT" | grep -q "\[FAIL\]"; then
  echo "⚠️ trade-pulse 探测脚本执行异常（exit=$RC），走 failureAlert 兜底"
  echo "$OUT" | tail -5
  exit 1
fi

# 有源失败 → 由 link_health 统一状态机决策是否告警（唯一 writer，防双 writer 冲突）
FAILED=$(echo "$OUT" | grep "^  \[FAIL\]" | sed 's/^  \[FAIL\] //' | cut -d: -f1 | tr '\n' ' ')
DETAIL=$(echo "$OUT" | grep "^  \[FAIL\]" | head -3 | sed 's/^  \[FAIL\] //')
export PROBE_MODE="$MODE" PROBE_DIR="$SCRIPT_DIR" PROBE_FAILED="$FAILED" PROBE_DETAIL="$DETAIL"

python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["PROBE_DIR"])
from link_health import evaluate_alert
from feishu_push import push_text

alert = evaluate_alert(os.environ.get("PROBE_SYMBOL", "588000"))
if not alert.get("should_alert"):
    sys.exit(0)  # 滞回未到阈值或已告警过（持续挂不重复轰炸）

severity = alert.get("severity", "normal")
level = alert.get("level", "degraded")
reason = alert.get("reason", "")
if severity == "emergency":
    push_text("🚨 trade-pulse 数据源全挂（紧急）\n"
              f"探测时点: {os.environ['PROBE_MODE']}\n"
              f"失败源: {os.environ['PROBE_FAILED']}\n"
              f"详情:\n{os.environ['PROBE_DETAIL']}\n"
              f"链路等级: {alert.get('emoji', '🔴')} {level}（{reason}）\n"
              "→ 今日实时信号将降级收盘口径，决策滞后一天\n"
              "→ 已持仓按上次有效信号持有，不恐慌操作\n"
              "→ 建议仓位将按链路可信度打折（见信号卡片）")
else:
    push_text("⚠️ trade-pulse 数据源降级\n"
              f"探测时点: {os.environ['PROBE_MODE']}\n"
              f"失败源: {os.environ['PROBE_FAILED']}\n"
              f"详情:\n{os.environ['PROBE_DETAIL']}\n"
              f"链路等级: {alert.get('emoji', '🟡')} {level}（{reason}）\n"
              "→ 系统将自动走新浪/腾讯兜底，信号仍可用\n"
              "→ 建议仓位将按链路可信度打折（见信号卡片）")
PYEOF
PUSH_RC=$?
if [ $PUSH_RC -eq 0 ]; then
  echo "⚠️ 数据源告警已推送"
else
  echo "❌ 数据源告警推送失败（python exit=$PUSH_RC），已记录探测状态"
  exit $PUSH_RC
fi
exit 0
