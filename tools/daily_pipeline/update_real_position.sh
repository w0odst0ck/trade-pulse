#!/usr/bin/env bash
# update_real_position.sh — 收盘后更新实盘持仓现价/市值（纯 shell + python，零 LLM）
# 行为：读 daily.csv 最新收盘价 → 更新 real_position.json 的 current_price /
#       market_value / unrealized_pnl / position_pct（总资产保持用户提供值）
# 触发：15:40 cron（15:30 收盘数据到位后）；手动可随时跑
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ trade-pulse 持仓更新：项目目录不存在"; exit 1; }

python3 - <<'PYEOF'
import json
from pathlib import Path

pos_path = Path("data/588000/real_position.json")
daily_path = Path("data/588000/daily.csv")

if not pos_path.exists():
    print("  [SKIP] real_position.json 不存在（无实盘持仓记录）")
    exit(0)
if not daily_path.exists():
    print("  [WARN] daily.csv 不存在，无法更新收盘价")
    exit(1)

pos = json.loads(pos_path.read_text(encoding="utf-8"))

import pandas as pd
df = pd.read_csv(daily_path)
if len(df) == 0:
    print("  [WARN] daily.csv 为空")
    exit(1)

close = float(df["close"].iloc[-1])
close_date = str(df["date"].iloc[-1])[:10]
shares = pos.get("shares", 0)
cost = pos.get("cost_basis", pos.get("entry_price", 0))
total = pos.get("total_assets", 0)

pos["current_price"] = close
pos["price_date"] = close_date
pos["market_value"] = round(shares * close, 2)
if cost:
    pos["unrealized_pnl"] = round((close - cost) * shares, 2)
    pos["unrealized_pnl_pct"] = round((close - cost) / cost * 100, 2)
if total > 0:
    pos["position_pct"] = round(pos["market_value"] / total * 100, 2)
pos["updated_at"] = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")

tmp = pos_path.with_name(pos_path.name + ".tmp")
tmp.write_text(json.dumps(pos, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(pos_path)
print(f"  [OK] 持仓已更新：收盘 {close}（{close_date}）市值 {pos['market_value']} 元 浮动 {pos.get('unrealized_pnl', 0):+.2f} 元")
PYEOF
exit $?
