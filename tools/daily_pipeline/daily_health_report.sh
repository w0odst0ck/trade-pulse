#!/usr/bin/env bash
# daily_health_report.sh — 链路健康日报（17:00，方案 B 稳定性闭环）
#
# 汇总当日各 cron 环节执行状态（logs/chain/YYYY-MM-DD.json）：
#   09:15 morning_catchup / 14:25 daily_signal / 14:30 health_check /
#   14:50 daily_confirm / 15:10 minute_fetch / 15:30 daily_fetch / 16:30 close_catchup
#
# 行为：
#   - 全环节 ok/skip → ✅ 日报（简短推送）
#   - 有 fail → 🔴 日报 + 失败明细（隔夜就能发现链路问题）
#   - 环节缺失（json 无该 task）→ ⚠️ 标记缺失（该环节可能没跑）
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ trade-pulse 健康日报：项目目录不存在"; exit 1; }

# 飞书密钥
FEISHU_APP_SECRET=$(sed -n 's/^export FEISHU_APP_SECRET="\(.*\)"/\1/p' ~/.bashrc)
if [ -z "$FEISHU_APP_SECRET" ]; then
  echo "⚠️ trade-pulse 健康日报：FEISHU_APP_SECRET 提取为空，中止"
  exit 1
fi
export FEISHU_APP_SECRET

DATE=$(date +%F)
FILE="logs/chain/$DATE.json"

if [ ! -s "$FILE" ]; then
  echo "⚠️ trade-pulse 健康日报：今日链路标记文件不存在（所有环节均未运行？链路可能全挂）"
  # 17:00 时所有环节都没记录本身就是异常 → exit 1 触发 failureAlert 告警
  exit 1
fi

# python 生成+推送日报；失败必须透传非零退出码（否则监控脚本自身失败会被静默，
# 造成假「全绿」——监控的监控不能失效）
python3 - "$FILE" "$DATE" <<'PYEOF'
import json, os, sys

path = sys.argv[1]
today = sys.argv[2]
try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"⚠️ trade-pulse 健康日报：标记文件解析失败: {e}")
    sys.exit(1)

# 期望环节（按时间序）+ 中文标签
EXPECTED = [
    ("morning_catchup", "09:15 早间补拉"),
    ("daily_signal",    "14:25 盘中预览"),
    ("health_check",    "14:30 健康检查"),
    ("daily_confirm",   "14:50 尾盘确认"),
    ("minute_fetch",    "15:10 分钟数据"),
    ("daily_fetch",     "15:30 收盘日线"),
    ("close_catchup",   "16:30 收盘兜底"),
]

lines = []
fails = []
missing = []
for task, label in EXPECTED:
    entry = data.get(task)
    if not entry:
        missing.append(label)
        lines.append(f"  ⚠️ {label}: 未记录（环节可能未运行）")
        continue
    status = entry.get("status", "?")
    detail = entry.get("detail", "")
    time = entry.get("time", "")
    if status == "ok":
        lines.append(f"  ✅ {label} ({time})")
    elif status == "skip":
        lines.append(f"  ➖ {label} ({time}) 跳过: {detail}")
    else:
        fails.append(f"{label} ({time}): {detail}")
        lines.append(f"  ❌ {label} ({time}): {detail}")

# 汇总标题
if fails:
    title = f"🔴 trade-pulse 链路日报 {today}：{len(fails)} 个环节异常"
elif missing:
    title = f"🟡 trade-pulse 链路日报 {today}：{len(missing)} 个环节未记录"
else:
    title = f"✅ trade-pulse 链路日报 {today}：全部环节正常"

text = title + "\n" + "━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(lines)
print(text)

# 推送
sys.path.insert(0, "tools/daily_pipeline")
from feishu_push import push_text
try:
    push_text(text)
    print("  [PUSH] 链路日报已推送")
except Exception as e:
    print(f"  [WARN] 日报推送失败: {e}")
PYEOF
RC=$?
if [ $RC -ne 0 ]; then
  echo "⚠️ trade-pulse 健康日报生成失败（exit=$RC），推送中止（防止假全绿）"
  exit $RC
fi
exit 0
