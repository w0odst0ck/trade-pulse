#!/usr/bin/env bash
# trade-pulse 收盘后日线增量（15:30）— 纯 shell 实现，零 LLM 调用
# 行为：
#   - 成功 → ✅（exit 0，announce 推送）
#   - 数据未发布（预期）→ ⚠️（exit 0，announce 推送；command 非零退出会吞 delivery）
#   - 真实错误（网络/解析）→ ⚠️ 透传报错（exit 0 推送，避免被吞；16:30 兜底重试）
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ trade-pulse 收盘日线：项目目录不存在"; exit 1; }

TODAY=$(date +%F)
OUT=$(python3 tools/daily_pipeline/fetch_data.py --require-date "$TODAY" 2>&1)
RC=$?
if [ $RC -eq 0 ]; then
  # 收盘数据到位 → 回填实时快照的 final 真值列（双线并行数据层；失败不阻塞）
  python3 tools/daily_pipeline/realtime_daily.py --backfill-final
  echo "✅ trade-pulse 日线已更新到 $TODAY"
  exit 0
fi

# 区分失败类型：完整性校验失败 = 数据未发布（常态：腾讯盘后定型晚于 16:30）
# → 静默 NO_REPLY（不打扰，09:15 早间补拉失败才是真正需要人工的场景）
if echo "$OUT" | grep -q "数据完整性校验失败"; then
  echo "NO_REPLY"
  exit 0
fi

# 其他错误（网络/解析/配置）：透传具体报错，需人工排查
# （exit 0 推送文本，command 非零退出会吞 delivery）
echo "⚠️ trade-pulse 日线拉取异常（exit=$RC），非「数据未发布」场景，建议人工检查："
echo "$OUT" | tail -10
exit 0
