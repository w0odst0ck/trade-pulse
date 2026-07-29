#!/usr/bin/env python3
"""
fetch_data.py — 588000 日线数据拉取

功能：从 AkShare 拉取 588000（科创50ETF）及基准 000688（科创50指数）日线
      自动增量更新，带重试和缓存。
备用源：新浪 / 东方财富网页接口（AkShare 挂了自动降级）

用法：
  python fetch_data.py                    # 增量更新
  python fetch_data.py --force            # 强制全量拉取
  python fetch_data.py --start 2024-01-01 # 指定起始日期
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_data_path(symbol: str) -> Path:
    """数据缓存路径：data/{symbol}/daily.csv"""
    config = load_config()
    data_dir = PROJECT_ROOT / config["data_dir"]
    return data_dir / symbol / "daily.csv"


def normalize_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """AkShare 中文列名 → 统一英文列名"""
    col_map = {
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '振幅': 'amplitude',
        '涨跌幅': 'change_pct',
        '涨跌额': 'change',
        '换手率': 'turnover',
    }
    df = df.rename(columns=col_map)
    df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount',
            'amplitude', 'change_pct', 'change', 'turnover']]
    df['date'] = pd.to_datetime(df['date'])
    df['symbol'] = symbol
    return df.sort_values('date').reset_index(drop=True)


def fetch_akshare(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    """从 AkShare 拉取 ETF/指数日线"""
    import akshare as ak

    try:
        if symbol.startswith('5') or symbol.startswith('1'):
            # ETF（588000 等用 fund_etf_hist_em）
            df = ak.fund_etf_hist_em(
                symbol=symbol, period='daily',
                start_date=start_date.replace('-', ''),
                end_date='20500101', adjust='qfq'
            )
        else:
            # 指数（000688 等用 stock_zh_index_daily_em，需加 sh 前缀）
            from akshare import stock_zh_index_daily_em
            sh_symbol = f'sh{symbol}' if not symbol.startswith('sh') else symbol
            df_all = stock_zh_index_daily_em(symbol=sh_symbol)
            # 指数数据列名可能是中文或英文，统一处理
            ren = {}
            for col in df_all.columns:
                if '日' in str(col) or 'date' in str(col).lower():
                    ren[col] = 'date'
                elif '开' in str(col):
                    ren[col] = 'open'
                elif '收' in str(col):
                    ren[col] = 'close'
                elif '最' in str(col) and '低' in str(col):
                    ren[col] = 'low'
                elif '最' in str(col) and ('高' in str(col) or 'high' in str(col).lower()):
                    ren[col] = 'high'
                elif '成交' in str(col) or 'volume' in str(col).lower() or 'vol' in str(col).lower():
                    ren[col] = 'volume'
                elif '金额' in str(col) or 'amount' in str(col).lower() or 'amt' in str(col).lower():
                    ren[col] = 'amount'
                elif '振' in str(col) or 'amplitude' in str(col).lower():
                    ren[col] = 'amplitude'
                elif '涨跌' in str(col) and '幅' in str(col):
                    ren[col] = 'change_pct'
                elif '涨跌' in str(col) and ('额' in str(col) or '值' in str(col)):
                    ren[col] = 'change'
                elif '换手' in str(col) or 'turnover' in str(col).lower():
                    ren[col] = 'turnover'
            df_all = df_all.rename(columns=ren)
            # 补缺失列（指数数据可能没有某些字段）
            for req_col in ['date', 'open', 'close', 'high', 'low', 'volume']:
                if req_col not in df_all.columns:
                    raise KeyError(f"指数数据缺少必需列: {req_col}, 实际列: {list(df_all.columns)}")
            for opt_col in ['amount', 'amplitude', 'change_pct', 'change', 'turnover']:
                if opt_col not in df_all.columns:
                    df_all[opt_col] = 0
            if 'date' in df_all.columns:
                # date 可能是 '2026-07-29' 或 '20260729' 格式，统一去横线比较
                df = df_all[df_all['date'].str.replace('-', '', regex=False) >= start_date.replace('-', '')].copy()
            else:
                df = df_all.copy()

        if df is None or len(df) == 0:
            return None
        return df
    except Exception as e:
        print(f"  [WARN] AkShare 请求失败: {e}", file=sys.stderr)
        return None


def fetch_eastmoney_fallback(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    """备用源：东财网页接口直拉（不经过 AkShare 封装）"""
    import requests

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://quote.eastmoney.com/'
        }

        if symbol.startswith('5'):
            # ETF
            secid = f"1.{symbol}"
            url = (
                "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
                "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
                "&klt=101&fqt=1"
                f"&beg={start_date.replace('-', '')}&end=20500101"
            )
        else:
            # 指数
            secid = f"1.{symbol}"
            url = (
                "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
                "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
                "&klt=101&fqt=1"
                f"&beg={start_date.replace('-', '')}&end=20500101"
            )

        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()

        if data.get('data') is None or data['data'].get('klines') is None:
            return None

        rows = []
        for line in data['data']['klines']:
            parts = line.split(',')
            rows.append({
                'date': parts[0],
                'open': float(parts[1]),
                'close': float(parts[2]),
                'high': float(parts[3]),
                'low': float(parts[4]),
                'volume': float(parts[5]),
                'amount': float(parts[6]),
                'amplitude': float(parts[7]) if len(parts) > 7 else 0,
                'change_pct': float(parts[8]) if len(parts) > 8 else 0,
                'change': float(parts[9]) if len(parts) > 9 else 0,
                'turnover': float(parts[10]) if len(parts) > 10 else 0,
            })

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        return df.sort_values('date').reset_index(drop=True)

    except Exception as e:
        print(f"  [WARN] 备用源也挂了: {e}", file=sys.stderr)
        return None


def load_local(path: Path) -> pd.DataFrame:
    """读取本地缓存数据"""
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        return df.sort_values('date').reset_index(drop=True)
    return pd.DataFrame()


def fetch_data(symbol: str, start_date: str, force: bool = False) -> pd.DataFrame:
    """拉取数据（主入口），自动增量更新"""
    config = load_config()
    data_path = get_data_path(symbol)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 读本地缓存
    local_df = load_local(data_path)

    if force or len(local_df) == 0:
        # 全量拉取
        print(f"  [INFO] 全量拉取 {symbol} 从 {start_date}")
        start = start_date
    else:
        # 增量：从最新日期往后拉
        last_date = local_df['date'].max().strftime('%Y-%m-%d')
        start = last_date
        print(f"  [INFO] 增量更新 {symbol} 从 {start}（已有 {len(local_df)} 条到 {last_date}）")

    # 2. 尝试 AkShare
    df_new = None
    retries = config.get("retry_count", 2)
    for attempt in range(retries + 1):
        df_new = fetch_akshare(symbol, start)
        if df_new is not None and len(df_new) > 0:
            break
        if attempt < retries:
            print(f"  [RETRY] 第 {attempt + 1} 次重试...")
            time.sleep(config.get("retry_delay_sec", 3))

    # 3. AkShare 失败 → 备用源（带重试）
    if df_new is None or len(df_new) == 0:
        print(f"  [FALLBACK] 切东方财富网页接口...")
        for attempt in range(retries + 1):
            df_new = fetch_eastmoney_fallback(symbol, start)
            if df_new is not None and len(df_new) > 0:
                break
            if attempt < retries:
                print(f"  [RETRY] 备用源第 {attempt + 1} 次重试...")
                time.sleep(config.get("retry_delay_sec", 3))

    # 4. 全挂了 → 用本地缓存（打标记）
    if df_new is None or len(df_new) == 0:
        if len(local_df) > 0:
            print(f"  ⚠️ 数据源均不可用，使用本地缓存（{len(local_df)} 条）")
            local_df.attrs['stale'] = True
            return local_df
        else:
            raise RuntimeError(f"无法获取 {symbol} 数据，且无本地缓存")

    # 5. 标准化列名
    df_new = normalize_columns(df_new, symbol)

    # 6. 合并增量
    if len(local_df) > 0 and not force:
        combined = pd.concat([local_df, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    else:
        combined = df_new

    # 7. 写缓存
    combined.to_csv(data_path, index=False)
    print(f"  [OK] {symbol}: {len(combined)} 条 ({combined['date'].min().date()} ~ {combined['date'].max().date()})")
    return combined


def main():
    parser = argparse.ArgumentParser(description="拉取 588000 + 000688 日线数据")
    parser.add_argument('--force', action='store_true', help='强制全量重拉')
    parser.add_argument('--start', default='2023-01-01', help='起始日期 (YYYY-MM-DD)')
    args = parser.parse_args()

    config = load_config()
    print("\n📥 数据拉取")
    symbol = config['symbol']
    benchmark = config['benchmark']

    df_sym = fetch_data(symbol, args.start, args.force)
    df_bench = fetch_data(benchmark, args.start, args.force)

    return df_sym, df_bench


if __name__ == '__main__':
    main()
