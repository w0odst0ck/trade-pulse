#!/usr/bin/env bash
# trade-pulse 分钟数据增量（15:10）— 纯 shell 实现，零 LLM 调用
# 行为：baostock 免费源增量拉取，为 Kronos 分钟级验证积累数据
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SCRIPT_DIR/../.." && pwd)
LOCK_FILE=/tmp/trade-pulse-minute-fetch.lock
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR"

cd "$PROJ" || { echo "⚠️ trade-pulse 分钟数据：项目目录不存在"; exit 1; }

# 并发保护：已有实例在跑则直接失败（fetch_minute.py 非原子写 CSV，防并发写坏）
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "⚠️ trade-pulse 分钟数据：已有实例在运行，本次跳过"
  exit 1
fi

LOG_FILE="$LOG_DIR/minute_$(date +%F).log"
python3 tools/kronos/fetch_minute.py --all >>"$LOG_FILE" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
  bash tools/daily_pipeline/chain_mark.sh minute_fetch ok "分钟线增量完成"
  echo "✅ trade-pulse 分钟数据已增量更新（日志 $LOG_FILE）"
else
  bash tools/daily_pipeline/chain_mark.sh minute_fetch fail "exit=$RC"
  echo "⚠️ trade-pulse 分钟数据拉取失败（exit=$RC，日志 $LOG_FILE）"
  tail -5 "$LOG_FILE"
fi
exit $RC
