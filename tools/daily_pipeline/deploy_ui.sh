#!/usr/bin/env bash
# trade-pulse UI 部署公共脚本（daily_signal.sh 14:25 / daily_confirm.sh 14:50 共用）
#
# 职责：git add docs/ → 有变更则 commit → fetch → rebase → push。
# 并发安全设计（14:25 与 14:50 相隔 25 分钟，可能重叠部署）：
#   - flock 互斥锁：两任务共享同一 git 工作区，同时执行会互相破坏
#     （A 的 rebase 进行中，B 的 abort 会中止 A 的操作）→ 后到者等待
#   - fetch 失败 → 退出（本次部署失败，下次自动重试），绝不盲目 push
#   - rebase 冲突 → abort 恢复（本地 commit 保留，下次重试带上）
#   - 无变更且无未推送 commit → 跳过部署（成功退出）
set -uo pipefail

COMMIT_MSG="${1:-ui: update dashboard $(date +%F)}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || { echo "⚠️ trade-pulse 部署：项目目录不存在"; exit 1; }

# 互斥锁（同一 git 工作区串行部署；等待最多 120s，超时放弃本次部署）
LOCK_FILE="$(pwd)/.git/deploy_ui.lock"
exec 9>"$LOCK_FILE" || { echo "⚠️ trade-pulse 部署：无法创建锁文件"; exit 1; }
flock -w 120 9 || { echo "⚠️ trade-pulse 部署：等待锁超时（另一部署进行中），放弃本次"; exit 1; }

git add docs/ || { echo "⚠️ trade-pulse git add 失败"; exit 1; }

# 无变更且无未推送 commit → 跳过
if git diff --cached --quiet -- docs/ && [ -z "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
  echo "✅ trade-pulse UI 无变化，跳过部署"
  exit 0
fi

# 有 staged docs 变更 → commit（未推送的遗留 commit 自动带上）
if ! git diff --cached --quiet -- docs/; then
  git commit -m "$COMMIT_MSG" || { echo "⚠️ trade-pulse git commit 失败"; exit 1; }
fi

# fetch 失败 → 退出（下次重试），绝不盲目 push 覆盖远端
if ! git fetch origin main; then
  echo "⚠️ trade-pulse git fetch 失败，退出（本次部署失败，下次自动重试）"
  exit 1
fi

# rebase 冲突 → abort 恢复（commit 保留），本次部署失败
if ! git rebase origin/main; then
  echo "⚠️ trade-pulse rebase 冲突，abort 恢复（commit 已保留，下次重试；本次部署失败）"
  git rebase --abort 2>/dev/null || true
  exit 1
fi

git push origin main
PUSH_RC=$?
if [ $PUSH_RC -ne 0 ]; then
  echo "⚠️ trade-pulse git push 失败（exit=$PUSH_RC）"
  exit $PUSH_RC
fi
echo "✅ trade-pulse UI 已部署"
exit 0
