#!/usr/bin/env bash
# trade-pulse 尾盘确认（14:50）— 纯 shell 实现，零 LLM 调用（方案 B）
# 行为：实时模式 confirm → 写 state.json（signal_mode=realtime）→ 推送「🟢 尾盘确认」
#       → UI 构建部署（dashboard 反映当日实时确认）
# 与 14:25 daily_signal.sh 分工：14:25 盘中预览（不写 state）→ 14:50 尾盘确认（写 state）
# 兜底：实时源不可用 → daily_panel 自动回退收盘口径且不写 state（避免脏数据污染状态机）
set -uo pipefail

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJ" || { echo "⚠️ trade-pulse 尾盘确认：项目目录不存在"; exit 1; }

# 飞书密钥
FEISHU_APP_SECRET=$(sed -n 's/^export FEISHU_APP_SECRET="\(.*\)"/\1/p' ~/.bashrc)
if [ -z "$FEISHU_APP_SECRET" ]; then
  echo "⚠️ trade-pulse 尾盘确认：FEISHU_APP_SECRET 提取为空，中止"
  exit 1
fi
export FEISHU_APP_SECRET

TODAY=$(date +%F)

# 1. 尾盘确认（实时模式 + confirm：写 state + 推送「🟢 尾盘确认」）
python3 tools/daily_pipeline/daily_panel.py --realtime-confirm --push
RC=$?
if [ $RC -ne 0 ]; then
  echo "⚠️ trade-pulse 尾盘确认失败（exit=$RC）"
  exit $RC
fi

# 1.2 14:50 实时快照积累（双线并行数据层：close_1450 列；失败不影响主流程）
python3 tools/daily_pipeline/realtime_daily.py --snapshot-1450
SNAP_RC=$?
if [ $SNAP_RC -ne 0 ]; then
  echo "⚠️ trade-pulse 14:50 快照写入失败（exit=$SNAP_RC，不阻塞主流程）"
fi

# 2. UI 构建部署（dashboard 反映当日实时确认信号；公共部署脚本含并发安全）
python3 tools/ui/build_ui.py
UI_RC=$?
if [ $UI_RC -ne 0 ]; then
  echo "⚠️ trade-pulse UI 构建失败（exit=$UI_RC），尾盘确认已推送，跳过部署"
  exit $UI_RC
fi

bash tools/daily_pipeline/deploy_ui.sh "ui: update dashboard $TODAY (realtime confirm)"
GIT_RC=$?
if [ $GIT_RC -ne 0 ]; then
  echo "⚠️ trade-pulse UI 部署失败（exit=$GIT_RC），尾盘确认已推送"
  exit $GIT_RC
fi

bash tools/daily_pipeline/chain_mark.sh daily_confirm ok "确认+UI部署完成"
echo "✅ trade-pulse 尾盘确认已推送 + 面板已更新 https://w0odst0ck.github.io/trade-pulse/"
exit 0
