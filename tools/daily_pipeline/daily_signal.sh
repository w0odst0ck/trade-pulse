#!/usr/bin/env bash
# trade-pulse 每日信号盘中预览 + 纸面盘 + UI 部署（14:25）— 纯 shell 实现，零 LLM 调用
# 行为（方案 B）：实时模式预览信号推送（不写 state）→ 纸面盘更新 → UI 构建 → git 提交部署；
# 失败输出告警 + 透传退出码
# 注：14:25 为「盘中预览」（🟡，不写 state）；14:50 由 daily_confirm.sh 做「尾盘确认」（🟢，写 state）
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

# 1. 盘中预览推送（实时模式：拉实时 bar 拼今日特征，不写 state；失败自动回退收盘口径）
python3 tools/daily_pipeline/daily_panel.py --realtime --push
RC=$?
if [ $RC -ne 0 ]; then
  echo "⚠️ trade-pulse 盘中预览推送失败（exit=$RC）"
  exit $RC
fi

# 1.2 14:25 实时快照积累（双线并行数据层：close_1425 列；失败不影响主流程）
python3 tools/daily_pipeline/realtime_daily.py --snapshot-1425
SNAP_RC=$?
if [ $SNAP_RC -ne 0 ]; then
  echo "⚠️ trade-pulse 14:25 快照写入失败（exit=$SNAP_RC，不阻塞主流程）"
fi

# 1.5 接近翻多预警（实时口径，0.05 <= 综合分 < 0.1 时推飞书文本；其余静默）
bash tools/daily_pipeline/signal_watch.sh

# 2. 纸面盘增量（与回测同口径，无状态文件自动全量；收盘口径特征，不受实时影响）
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

# 4. UI 部署（公共脚本：add/commit/fetch/rebase/push + 并发安全；失败时跳过部署）
bash tools/daily_pipeline/deploy_ui.sh "ui: update dashboard $TODAY"
GIT_RC=$?
if [ $GIT_RC -ne 0 ]; then
  echo "⚠️ trade-pulse UI 部署失败（exit=$GIT_RC），信号推送已完成"
  exit $GIT_RC
fi

bash tools/daily_pipeline/chain_mark.sh daily_signal ok "预览+UI部署完成"
echo "✅ trade-pulse 今日盘中预览已推送 + 面板已更新 https://w0odst0ck.github.io/trade-pulse/（部署约1-2分钟）"
exit 0
