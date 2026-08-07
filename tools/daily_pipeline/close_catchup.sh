#!/usr/bin/env bash
# trade-pulse 收盘数据补齐（16:30 兜底）— 纯 shell 实现，零 LLM 调用
# 行为：数据已到位 → 静默；未到位 → 补拉，成功推 ✅ / 失败推 ⚠️ + exit 1
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ trade-pulse 收盘补齐：项目目录不存在"; exit 1; }

CSV=data/588000/daily.csv
if [ ! -s "$CSV" ]; then
  echo "⚠️ trade-pulse 收盘补齐：日报文件不存在或为空，开始补拉..."
  python3 tools/daily_pipeline/fetch_data.py --require-date "$TODAY"
  RC=$?
  if [ $RC -eq 0 ]; then
    NEW=$(tail -1 "$CSV" | cut -d, -f1)
    echo "✅ trade-pulse 补齐成功：日线已更新到 $NEW"
  else
    echo "⚠️ trade-pulse 补齐失败：所有数据源当天数据仍未发布，明日 09:15 早间兜底任务会自动补拉，无需人工操作"
  fi
  exit $RC
fi

LAST=$(tail -1 "$CSV" | cut -d, -f1)
TODAY=$(date +%F)

# 字符串比较 YYYY-MM-DD 即时间序；last >= today 视为已到位
if [[ "$LAST" == "$TODAY" || "$LAST" > "$TODAY" ]]; then
  echo "NO_REPLY"
  exit 0
fi

echo "⚠️ trade-pulse 收盘补齐：今日数据未到位（最新 $LAST），开始补拉..."
python3 tools/daily_pipeline/fetch_data.py --require-date "$TODAY"
RC=$?
if [ $RC -eq 0 ]; then
  NEW=$(tail -1 data/588000/daily.csv | cut -d, -f1)
  echo "✅ trade-pulse 补齐成功：日线已更新到 $NEW"
else
  echo "⚠️ trade-pulse 补齐失败：所有数据源当天数据仍未发布，明日 09:15 早间兜底任务会自动补拉，无需人工操作"
fi
exit $RC
