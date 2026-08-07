#!/usr/bin/env bash
# 数据源盘后定型时点验证（15:30 一次性）— 纯 shell 实现，零 LLM 调用
# 行为：各调一次新浪/腾讯，输出两个源的最新日期 + 判断
set -uo pipefail

PROJ=/home/l/.openclaw/workspace/projects/trade-pulse
cd "$PROJ" || { echo "⚠️ 数据源定型验证：项目目录不存在"; exit 1; }

SINA_OUT=$(timeout 60 python3 -c "
import akshare as ak
df = ak.fund_etf_hist_sina(symbol='sh588000')
print(df.iloc[-1, 0])
" 2>&1)
SINA_RC=$?

TENCENT_OUT=$(timeout 60 python3 -c "
import sys; sys.path.insert(0, 'tools')
from data_provider.tencent import TencentProvider
t = TencentProvider()
df = t.fetch_daily('588000', '2026-08-06', '2026-08-08')
print(df['date'].iloc[-1])
" 2>&1)
TENCENT_RC=$?

echo "新浪: 最新 ${SINA_OUT:-（失败 exit=$SINA_RC）}"
echo "腾讯: 最新 ${TENCENT_OUT:-（失败 exit=$TENCENT_RC）}"

# 判断（注意：这是盘中数据，收盘后可能还会变）
if [ "$SINA_RC" -eq 0 ] && [ "$TENCENT_RC" -eq 0 ]; then
  if [ "${SINA_OUT:0:10}" = "${TENCENT_OUT:0:10}" ]; then
    echo "结论: 两源一致（${SINA_OUT:0:10}），盘后定型时点相同"
  elif [ "${SINA_OUT:0:10}" > "${TENCENT_OUT:0:10}" ]; then
    echo "结论: 新浪盘后定型快于腾讯（新浪 ${SINA_OUT:0:10} > 腾讯 ${TENCENT_OUT:0:10}）"
  else
    echo "结论: 腾讯盘后定型快于新浪（腾讯 ${TENCENT_OUT:0:10} > 新浪 ${SINA_OUT:0:10}）"
  fi
fi
exit 0
