#!/usr/bin/env python3
"""
compute_features.py — 技术指标特征计算

功能：读 daily.csv → 算 6 个因子分数 + 周线过滤
      支持增量计算（已算过的行不动，只算最新数据）

用法：
  python compute_features.py                              # 增量计算
  python compute_features.py --force                      # 全量重算
  python compute_features.py --output panel               # 只输出最新因子面板数据
  python compute_features.py --output history             # 输出全量历史特征 CSV
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_data(symbol: str) -> pd.DataFrame:
    """从本地缓存读数据"""
    config = load_config()
    data_dir = PROJECT_ROOT / config["data_dir"]
    path = data_dir / symbol / "daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在，先跑 fetch_data.py: {path}")
    return pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)


def load_features_cache(symbol: str) -> pd.DataFrame:
    """读全量特征缓存"""
    config = load_config()
    data_dir = PROJECT_ROOT / config["data_dir"]
    path = data_dir / symbol / "features_cache.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()


def save_features_cache(df: pd.DataFrame, symbol: str):
    """写全量特征缓存"""
    config = load_config()
    data_dir = PROJECT_ROOT / config["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / symbol / "features_cache.csv"
    df.to_csv(path, index=False)


def momentum_score(df: pd.DataFrame, window: int = 5, hist: int = 60) -> pd.Series:
    """短期动量因子：ROC(window) → 自适应分位映射到 [-1, +1]"""
    roc = df['close'].pct_change(window)
    # 滚动分位映射
    result = pd.Series(np.nan, index=df.index)
    for i in range(hist, len(roc)):
        hist_vals = roc.iloc[i - hist: i].dropna()
        if len(hist_vals) < 10:
            continue
        lower = hist_vals.quantile(0.1)
        upper = hist_vals.quantile(0.9)
        val = roc.iloc[i]
        if upper > lower:
            score = (val - lower) / (upper - lower) * 2 - 1
            result.iloc[i] = np.clip(score, -1.0, 1.0)
    return result


def trend_score(df: pd.DataFrame, window: int = 20, slope_window: int = 5, hist: int = 60) -> pd.Series:
    """中期趋势因子：MA20 斜率 → [-1, +1]"""
    ma = df['close'].rolling(window).mean()
    slope = ma.diff(slope_window) / ma.shift(slope_window)

    result = pd.Series(np.nan, index=df.index)
    for i in range(hist, len(slope)):
        hist_vals = slope.iloc[i - hist: i].dropna()
        if len(hist_vals) < 10:
            continue
        val = slope.iloc[i]
        lower = hist_vals.quantile(0.1)
        upper = hist_vals.quantile(0.9)

        if val > 0:
            score = val / upper if upper > 0 else 0.5
        else:
            score = val / abs(lower) if lower < 0 else -0.5
        result.iloc[i] = np.clip(score, -1.0, 1.0)
    return result


def volatility_score(df: pd.DataFrame, window: int = 14, hist: int = 90) -> pd.Series:
    """波动率因子：ATR → 低波=+1，高波=-1"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    result = pd.Series(np.nan, index=df.index)
    for i in range(hist, len(atr)):
        hist_vals = atr.iloc[i - hist: i].dropna()
        if len(hist_vals) < 10:
            continue
        vmin, vmax = hist_vals.min(), hist_vals.max()
        if vmax > vmin:
            pct = (atr.iloc[i] - vmin) / (vmax - vmin)  # 0~1
            score = 1 - pct * 2  # 高波=-1，低波=+1
            result.iloc[i] = np.clip(score, -1.0, 1.0)
    return result


def volume_price_score(df: pd.DataFrame) -> pd.Series:
    """量价配合因子 → [-1, +1]，基于规则"""
    price_chg5 = df['close'].pct_change(5)
    vol_chg5 = df['volume'].pct_change(5)

    scores = pd.Series(0.0, index=df.index)

    # 放量上涨
    mask_bull = (price_chg5 > 0.02) & (vol_chg5 > 0.2)
    scores[mask_bull] = 1.0

    # 放量下跌
    mask_bear = (price_chg5 < -0.02) & (vol_chg5 > 0.2)
    scores[mask_bear] = -1.0

    # 缩量调整（小跌+量缩）
    mask_adj = (price_chg5.between(-0.02, 0)) & (vol_chg5 < -0.1)
    scores[mask_adj] = 0.5

    # 缩量上涨（力度存疑）
    mask_weak = (price_chg5 > 0.01) & (vol_chg5 < -0.1)
    scores[mask_weak] = 0.2

    return scores


def rsrs_score(df: pd.DataFrame, window: int = 18) -> pd.Series:
    """RSRS（阻力支撑相对强弱）→ [-1, +1]

    用 rolling corr 近似 OLS beta：high 与 low 的相关性 * r²
    """
    from scipy import stats

    result = pd.Series(np.nan, index=df.index)
    rsrs_series = pd.Series(np.nan, index=df.index)

    for i in range(window, len(df)):
        seg_high = df['high'].iloc[i - window: i]
        seg_low = df['low'].iloc[i - window: i]
        corr, _ = stats.pearsonr(seg_low, seg_high)
        beta = corr * (seg_high.std() / seg_low.std())
        r2 = corr ** 2
        rsrs_series.iloc[i] = beta * r2

    # 滚动 z-score 归一化
    hist = 60
    for i in range(hist, len(rsrs_series)):
        hist_vals = rsrs_series.iloc[i - hist: i].dropna()
        if len(hist_vals) < 10:
            continue
        mean, std = hist_vals.mean(), hist_vals.std()
        if std > 0:
            z = (rsrs_series.iloc[i] - mean) / std
            result.iloc[i] = np.clip(z / 2, -1.0, 1.0)

    return result


def relative_strength_score(df: pd.DataFrame, df_bench: pd.DataFrame, window: int = 20, hist: int = 60) -> pd.Series:
    """比价优势因子：symbol vs benchmark → [-1, +1]"""
    # 对齐日期
    merged = pd.merge(
        df[['date', 'close']].rename(columns={'close': 'close_sym'}),
        df_bench[['date', 'close']].rename(columns={'close': 'close_bench'}),
        on='date', how='inner'
    ).sort_values('date').reset_index(drop=True)

    ret_sym = merged['close_sym'].pct_change(window)
    ret_bench = merged['close_bench'].pct_change(window)
    diff = ret_sym - ret_bench

    result = pd.Series(np.nan, index=merged.index)
    for i in range(hist, len(diff)):
        hist_vals = diff.iloc[i - hist: i].dropna()
        if len(hist_vals) < 10:
            continue
        lower = hist_vals.quantile(0.1)
        upper = hist_vals.quantile(0.9)
        val = diff.iloc[i]
        if upper > lower:
            score = (val - lower) / (upper - lower) * 2 - 1
            result.iloc[i] = np.clip(score, -1.0, 1.0)

    # 映射回原 df 的时间轴
    merged['score'] = result
    final = df[['date']].merge(merged[['date', 'score']], on='date', how='left')
    return final['score']


def weekly_filter(df: pd.DataFrame, percentile: float = 0.2) -> pd.Series:
    """大周期过滤：周线 MA20 斜率是否处于最低 20% → True=可交易"""
    df_weekly = df.set_index('date').resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum', 'amount': 'sum'
    }).dropna().reset_index()

    df_weekly['ma20'] = df_weekly['close'].rolling(20).mean()
    df_weekly['ma20_slope'] = df_weekly['ma20'].pct_change(5)

    result = pd.Series(True, index=df_weekly.index)
    for i in range(25, len(df_weekly)):
        hist = df_weekly['ma20_slope'].iloc[i - 20: i].dropna()
        if len(hist) < 5:
            continue
        threshold = hist.quantile(percentile)
        val = df_weekly['ma20_slope'].iloc[i]
        if val < threshold:
            result.iloc[i] = False

    df_weekly['can_trade'] = result

    # 映射回去：周线结果给每天的最后一根用
    daily_result = pd.Series(True, index=df.index)
    df_with_week = df[['date']].copy()
    df_with_week['week_label'] = pd.to_datetime(df['date']).dt.to_period('W-FRI').dt.to_timestamp()
    weekly_map = df_weekly.set_index('date')['can_trade'].to_dict()
    mask = df_with_week['week_label'].map(weekly_map).fillna(True)
    daily_result = mask.values

    return daily_result


def compute_all_features(df_sym: pd.DataFrame, df_bench: pd.DataFrame, config: dict,
                         force: bool = False) -> pd.DataFrame:
    """计算全量历史特征"""
    symbol = config['symbol']

    # 读取已有缓存
    cached = load_features_cache(symbol) if not force else pd.DataFrame()

    if not force and len(cached) > 0:
        # 只算新数据
        last_cached_date = cached['date'].max()
        new_data = df_sym[df_sym['date'] > last_cached_date].copy()
        if len(new_data) == 0:
            print(f"  [SKIP] 特征已是最新（{last_cached_date.date()}）")
            return cached

        print(f"  [INFO] 增量计算特征：{len(new_data)} 条新数据（从 {new_data['date'].min().date()}）")
        # 取缓存 + 新数据合并作为完整 df
        df_full = pd.concat([cached[['date']], df_sym], ignore_index=False).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        # 但新计算只针对新行
        compute_df = df_sym
    else:
        print(f"  [INFO] 全量计算特征：{len(df_sym)} 条")
        compute_df = df_sym
        cached = pd.DataFrame()

    w = config['weights']
    windows = {
        'momentum': config.get('momentum_window', 5),
        'trend': config.get('trend_window', 20),
        'atr': config.get('atr_window', 14),
        'rsrs': config.get('rsrs_window', 18),
        'rel_strength': config.get('rel_strength_window', 20),
    }

    # 算因子
    features = compute_df[['date', 'close', 'volume']].copy()
    features['momentum'] = momentum_score(compute_df, windows['momentum'])
    features['trend'] = trend_score(compute_df, windows['trend'])
    features['volatility'] = volatility_score(compute_df, windows['atr'])
    features['volume_price'] = volume_price_score(compute_df)
    features['rsrs'] = rsrs_score(compute_df, windows['rsrs'])
    features['relative_strength'] = relative_strength_score(compute_df, df_bench, windows['rel_strength'])
    features['weekly_can_trade'] = weekly_filter(compute_df, config['thresholds']['weekly_filter_percentile'])

    # 加权总分
    features['total_score'] = (
        features['momentum'].fillna(0) * w['momentum'] +
        features['trend'].fillna(0) * w['trend'] +
        features['volatility'].fillna(0) * w['volatility'] +
        features['volume_price'].fillna(0) * w['volume_price'] +
        features['rsrs'].fillna(0) * w['rsrs'] +
        features['relative_strength'].fillna(0) * w['relative_strength']
    )

    # 合并缓存
    if not force and len(cached) > 0:
        combined = pd.concat([cached, features], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    else:
        combined = features

    save_features_cache(combined, symbol)
    print(f"  [OK] 特征缓存: {len(combined)} 条 ({combined['date'].min().date()} ~ {combined['date'].max().date()})")
    return combined


def get_latest_features(symbol: str) -> dict:
    """取出最新一天的因子值（用于信号面板）"""
    features = load_features_cache(symbol)
    if len(features) == 0:
        return {}

    latest = features.iloc[-1].to_dict()
    return {
        'date': str(latest['date'].date()) if hasattr(latest['date'], 'date') else str(latest['date'])[:10],
        'momentum': round(latest.get('momentum', 0), 3),
        'trend': round(latest.get('trend', 0), 3),
        'volatility': round(latest.get('volatility', 0), 3),
        'volume_price': round(latest.get('volume_price', 0), 3),
        'rsrs': round(latest.get('rsrs', 0), 3),
        'relative_strength': round(latest.get('relative_strength', 0), 3),
        'total_score': round(latest.get('total_score', 0), 3),
        'weekly_can_trade': bool(latest.get('weekly_can_trade', True)),
    }


def main():
    parser = argparse.ArgumentParser(description='计算 588000 技术指标特征')
    parser.add_argument('--force', action='store_true', help='全量重算')
    parser.add_argument('--output', choices=['panel', 'history', 'all'], default='panel',
                       help='输出内容：panel=最新因子 | history=全量历史 | all=两者')
    args = parser.parse_args()

    config = load_config()
    print("\n📊 特征计算")
    df_sym = load_data(config['symbol'])
    df_bench = load_data(config['benchmark'])

    features = compute_all_features(df_sym, df_bench, config, args.force)

    if args.output in ('panel', 'all'):
        print("\n--- 最新因子 ---")
        latest = get_latest_features(config['symbol'])
        for k, v in latest.items():
            print(f"  {k}: {v}")

    if args.output in ('history', 'all'):
        history_path = Path(PROJECT_ROOT) / config['data_dir'] / config['symbol'] / 'features_history.csv'
        features.to_csv(history_path, index=False)
        print(f"\n  [OK] 历史特征已写入: {history_path}")

    return features


if __name__ == '__main__':
    main()
