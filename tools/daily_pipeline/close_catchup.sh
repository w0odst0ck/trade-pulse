#!/usr/bin/env bash
# trade-pulse 收盘数据补齐（16:30 兜底）— 纯 shell 实现，零 LLM 调用
# 行为：
#   - 数据已到位 → NO_REPLY（静默）
#   - 补拉成功 → ✅（exit 0，announce 推送）
#   - 补拉失败（数据未发布，常态）→ NO_REPLY（静默；09:15 早间补拉失败才是真告警）
#   - 脚本自身故障（目录不存在等）→ exit 1（failureAlert 兜底）
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ trade-pulse 收盘补齐：项目目录不存在"; exit 1; }

CSV=data/588000/daily.csv
TODAY=$(date +%F)

if [ ! -s "$CSV" ]; then
  echo "⚠️ trade-pulse 收盘补齐：日报文件不存在或为空，开始补拉..."
  python3 tools/daily_pipeline/fetch_data.py --require-date "$TODAY"
  RC=$?
  if [ $RC -eq 0 ]; then
    NEW=$(tail -1 "$CSV" | cut -d, -f1)
    echo "✅ trade-pulse 补齐成功：日线已更新到 $NEW"
  else
    # 文件缺失 + 补拉失败：这是异常（正常文件应存在），推告警
    echo "⚠️ trade-pulse 补齐失败：日报文件缺失且补拉未成功，明日 09:15 兜底任务会自动补拉"
  fi
  exit 0
fi

LAST=$(tail -1 "$CSV" | cut -d, -f1)

# 字符串比较 YYYY-MM-DD 即时间序；last >= today 视为已到位
if [[ "$LAST" == "$TODAY" || "$LAST" > "$TODAY" ]]; then
  echo "NO_REPLY"
  exit 0
fi

# 今日未到位 → 补拉；未发布是常态（腾讯定型晚），静默；真实故障（traceback）需透传
echo "⚠️ trade-pulse 收盘补齐：今日数据未到位（最新 $LAST），开始补拉..."
OUT=$(python3 tools/daily_pipeline/fetch_data.py --require-date "$TODAY" 2>&1)
RC=$?
if [ $RC -eq 0 ]; then
  # 收盘数据到位 → 回填实时快照 final 真值列（兜底链路同样处理）
  python3 tools/daily_pipeline/realtime_daily.py --backfill-final
  NEW=$(tail -1 "$CSV" | cut -d, -f1)
  echo "✅ trade-pulse 补齐成功：日线已更新到 $NEW"
  exit 0
fi

# 区分：完整性校验失败 = 未发布（常态，静默）；其他错误 = 真实故障（推送）
if echo "$OUT" | grep -q "数据完整性校验失败"; then
  echo "NO_REPLY"
  exit 0
fi
echo "⚠️ trade-pulse 补齐异常（exit=$RC），非「数据未发布」场景，建议人工检查："
echo "$OUT" | tail -10
exit 0
