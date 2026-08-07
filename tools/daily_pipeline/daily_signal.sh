#!/usr/bin/env bash
# trade-pulse 每日信号推送 + UI 部署（14:25）— 纯 shell 实现，零 LLM 调用
# 行为：信号推送 → 纸面盘更新 → UI 构建 → git 提交部署；失败输出告警 + 透传退出码
set -uo pipefail

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJ" || { echo "⚠️ trade-pulse 每日信号：项目目录不存在"; exit 1; }

# 飞书密钥（不要 source ~/.bashrc，非交互 shell 会提前 return）
FEISHU_APP_SECRET=$(sed -n 's/^export FEISHU_APP_SECRET="\(.*\)"/\1/p' ~/.bashrc)
if [ -z "$FEISHU_APP_SECRET" ]; then
  echo "⚠️ trade-pulse 每日信号：FEISHU_APP_SECRET 提取为空（检查 ~/.bashrc 格式），中止"
  exit 1
fi
export FEISHU_APP_SECRET

TODAY=$(date +%F)

# 1. 信号推送（内部 fetch_data 多源，失败自动降级）
python3 tools/daily_pipeline/daily_panel.py --push
RC=$?
if [ $RC -ne 0 ]; then
  echo "⚠️ trade-pulse 每日信号推送失败（exit=$RC）"
  exit $RC
fi

# 1.5 接近翻多预警（0.05 <= 综合分 < 0.1 时推飞书文本；其余静默）
bash tools/daily_pipeline/signal_watch.sh

# 2. 纸面盘增量（与回测同口径，无状态文件自动全量）
python3 tools/daily_pipeline/paper_trade.py --update
PAPER_RC=$?
if [ $PAPER_RC -ne 0 ]; then
  echo "⚠️ trade-pulse 纸面盘更新失败（exit=$PAPER_RC）"
  exit $PAPER_RC
fi

# 3. UI 构建（失败时跳过 git 提交）
python3 tools/ui/build_ui.py
UI_RC=$?
if [ $UI_RC -ne 0 ]; then
  echo "⚠️ trade-pulse UI 构建失败（exit=$UI_RC），跳过部署，信号推送已完成"
  exit $UI_RC
fi

# 4. git 提交部署（空 diff 不算失败）
git add docs/
if git diff --cached --quiet -- docs/; then
  echo "✅ trade-pulse 今日信号已推送（UI 无变化，跳过部署）"
  exit 0
fi
git commit -m "ui: update dashboard $TODAY" && git push origin main
GIT_RC=$?
if [ $GIT_RC -ne 0 ]; then
  echo "⚠️ trade-pulse git 部署失败（exit=$GIT_RC），UI 已构建未推送"
  exit $GIT_RC
fi

echo "✅ trade-pulse 今日信号已推送 + 面板已更新 https://w0odst0ck.github.io/trade-pulse/（部署约1-2分钟）"
exit 0
