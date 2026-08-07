#!/usr/bin/env bash
# trade-pulse 工具库索引上游同步（每月 1 号）— 纯 shell 实现，零 LLM 调用
# 行为：跑 sync_awesome.py（代理）；无新增 → 静默；有新增 → 推摘要
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SCRIPT_DIR/../.." && pwd)
LOCK_FILE=/tmp/trade-pulse-awesome-sync.lock

# 代理可配置：优先用已有环境变量，否则默认本地代理
export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890}"

cd "$PROJ" || { echo "⚠️ trade-pulse 工具库同步：项目目录不存在"; exit 1; }

# 并发保护：已有实例在跑则直接失败（避免索引文件竞争写坏）
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "⚠️ trade-pulse 工具库同步：已有实例在运行，本次跳过"
  exit 1
fi

OUT=$(python3 tools/sync_awesome.py 2>&1)
RC=$?
if [ $RC -ne 0 ]; then
  echo "⚠️ trade-pulse 工具库索引同步失败（exit=$RC）"
  echo "$OUT" | tail -5
  exit $RC
fi

# 无新增检测：匹配「过滤后 0 个」或「无值得评估的新增」任一（双保险防文案变动）
if echo "$OUT" | grep -qE "新增\(过滤后\): 0 个|无值得评估的新增"; then
  echo "NO_REPLY"
  exit 0
fi

echo "✅ trade-pulse 工具库索引已同步："
echo "$OUT" | grep -E "新增|候选池" | head -10
exit 0
