#!/usr/bin/env python3
"""
trading_calendar.py — A 股交易日历

功能：
  - is_trading_day(): 判断当天是否为交易日
  - next_trading_day(): 下一个交易日
  - prev_trading_day(): 上一个交易日

多标的预留：不绑定标的，只用日期判断
"""

from datetime import date, timedelta
from pathlib import Path
import json


# 手动维护的节假日（上海证券交易所）
# 格式：YYYY-MM-DD，只记非周末的假期
A_HOLIDAYS = {
    # 2026
    "2026-01-01",  # 元旦
    "2026-01-02",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",  # 春节
    "2026-04-06",  # 清明
    "2026-05-01", "2026-05-04", "2026-05-05",  # 劳动节
    "2026-06-19",  # 端午
    "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07",  # 国庆+中秋
    # 2027
    "2027-01-01",
    "2027-02-08", "2027-02-09", "2027-02-10", "2027-02-11", "2027-02-12",  # 春节
    "2027-04-05",  # 清明
    "2027-05-03", "2027-05-04", "2027-05-05",  # 劳动节
}

# 调休上班日（周末补班）
A_WORKDAYS = {
    # 2026
    "2026-02-14", "2026-02-15",  # 春节补班
    "2026-04-05",  # 清明补班
    "2026-04-25",  # 劳动节补班
    "2026-06-21",  # 端午补班
    "2026-09-27", "2026-10-10",  # 国庆补班
    # 2027
    "2027-02-06", "2027-02-07",  # 春节补班
    "2027-04-03",  # 清明补班
    "2027-05-02",  # 劳动节补班
}


def is_trading_day(d: date = None) -> bool:
    """判断是否为 A 股交易日"""
    if d is None:
        d = date.today()

    d_str = d.strftime("%Y-%m-%d")

    # 调休上班日
    if d_str in A_WORKDAYS:
        return True

    # 周末
    if d.weekday() >= 5:  # 5=周六, 6=周日
        return False

    # 法定假日
    if d_str in A_HOLIDAYS:
        return False

    return True


def next_trading_day(d: date = None) -> date:
    """下一个交易日"""
    if d is None:
        d = date.today()
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d: date = None) -> date:
    """上一个交易日"""
    if d is None:
        d = date.today()
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_between(start: date, end: date) -> list:
    """列举两个日期之间的所有交易日"""
    days = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


# ---- 命令行用法 ----
if __name__ == "__main__":
    from datetime import date
    today = date.today()
    print(f"今天: {today}  {'✅ 交易日' if is_trading_day(today) else '❌ 非交易日'}")
    print(f"上一个交易日: {prev_trading_day(today)}")
    print(f"下一个交易日: {next_trading_day(today)}")
