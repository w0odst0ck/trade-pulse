#!/usr/bin/env bash
# chain_mark.sh — 链路环节状态标记（方案 B 稳定性：链路健康日报用）
#
# 用法：bash chain_mark.sh <task_name> <status> [detail]
#   status: ok | fail | skip
#   写入 logs/chain/YYYY-MM-DD.json（append 合并，原子写）
#
# 各 cron 脚本在关键环节调用，17:00 daily_health_report.sh 汇总推送。
set -uo pipefail

TASK="${1:-unknown}"
STATUS="${2:-ok}"
DETAIL="${3:-}"

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || exit 1

DIR=logs/chain
mkdir -p "$DIR"
DATE=$(date +%F)
TIME=$(date +%H:%M)
FILE="$DIR/$DATE.json"

# 读现有（容忍损坏/不存在）
EXIST="{}"
if [ -s "$FILE" ]; then
  EXIST=$(cat "$FILE" 2>/dev/null || echo "{}")
fi

# python 合并写入（原子）
python3 - "$FILE" "$TASK" "$STATUS" "$TIME" "$DETAIL" <<'PYEOF'
import json, os, sys
path, task, status, ts, detail = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    data = {}
data[task] = {"status": status, "time": ts, "detail": detail}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
os.replace(tmp, path)
PYEOF
exit 0
