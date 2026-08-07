#!/usr/bin/env bash
# trade-pulse 早间补拉（09:15 昨日数据兜底）— 纯 shell 实现，零 LLM 调用
# 行为：
#   - 昨日数据已到位 → NO_REPLY（静默）
#   - 补拉成功 → ✅（exit 0，announce 推送）
#   - 补拉失败（数据未发布，预期）→ ⚠️（exit 0，announce 推送；command 非零退出会吞 delivery）
#   - 脚本自身故障（目录不存在等）→ exit 1
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ trade-pulse 早间补拉：项目目录不存在"; exit 1; }

CSV=data/588000/daily.csv
YESTERDAY=$(date -d yesterday +%F)

if [ ! -s "$CSV" ]; then
  echo "⚠️ trade-pulse 早间补拉：日报文件不存在或为空，开始补拉..."
  python3 tools/daily_pipeline/fetch_data.py --require-date "$YESTERDAY"
  RC=$?
  if [ $RC -eq 0 ]; then
    NEW=$(tail -1 "$CSV" | cut -d, -f1)
    echo "✅ trade-pulse 早间补拉成功：日线已更新到 $NEW"
  else
    echo "⚠️ trade-pulse 早间补拉仍失败：所有数据源连续两天未发布，建议人工检查"
  fi
  exit 0
fi

LAST=$(tail -1 "$CSV" | cut -d, -f1)

# 字符串比较 YYYY-MM-DD 即时间序；last >= yesterday 视为已到位
if [[ "$LAST" == "$YESTERDAY" || "$LAST" > "$YESTERDAY" ]]; then
  echo "NO_REPLY"
  exit 0
fi

echo "⚠️ trade-pulse 早间补拉：昨日数据未到位（最新 $LAST，目标 $YESTERDAY），开始补拉..."
python3 tools/daily_pipeline/fetch_data.py --require-date "$YESTERDAY"
RC=$?
if [ $RC -eq 0 ]; then
  NEW=$(tail -1 "$CSV" | cut -d, -f1)
  echo "✅ trade-pulse 早间补拉成功：日线已更新到 $NEW"
else
  echo "⚠️ trade-pulse 早间补拉仍失败：最新数据 $LAST，所有数据源未发布，建议人工检查"
fi
exit 0
