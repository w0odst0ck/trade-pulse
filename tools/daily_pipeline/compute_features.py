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
    path = data_dir / symbol / "features_cache.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def weekly_modifier(df: pd.DataFrame, config: dict) -> pd.Series:
    """周线调节分：MA20 斜率分位 → [-0.3, +0.3]，替代二进制过滤"""
    wm = config.get('weekly_modifier', {})
    min_mod = wm.get('min_modifier', -0.3)
    max_mod = wm.get('max_modifier', 0.3)

    df_weekly = df.set_index('date').resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum', 'amount': 'sum'
    }).dropna().reset_index()

    df_weekly['ma20'] = df_weekly['close'].rolling(20).mean()
    df_weekly['ma20_slope'] = df_weekly['ma20'].pct_change(5)

    result = pd.Series(0.0, index=df_weekly.index)
    for i in range(25, len(df_weekly)):
        hist = df_weekly['ma20_slope'].iloc[i - 20: i].dropna()
        if len(hist) < 5:
            continue
        val = df_weekly['ma20_slope'].iloc[i]
        # 分位 → 调节分
        rank = (hist < val).sum()
        pct = rank / len(hist)
        modifier = min_mod + (max_mod - min_mod) * pct
        result.iloc[i] = round(modifier, 3)

    # 映射回日线（用 period 字符串对齐，避免 resample 周五 vs to_period 周一不匹配）
    # 防前视：调节分 shift(1) 滞后一周——本周五收盘才确定的 ma20_slope，
    # 下周一才开始生效，避免周一~周四提前使用未来信息
    df_weekly['modifier'] = result.shift(1).values
    df_weekly['week_period'] = pd.to_datetime(df_weekly['date']).dt.to_period('W-FRI').astype(str)
    weekly_map = df_weekly.set_index('week_period')['modifier'].to_dict()
    df_daily = df[['date']].copy()
    df_daily['week_period'] = pd.to_datetime(df_daily['date']).dt.to_period('W-FRI').astype(str)
    filled = df_daily['week_period'].map(weekly_map).fillna(0.0)

    return filled.values


def ma60_slope(df: pd.DataFrame) -> pd.Series:
    """MA60 斜率，用于自适应阈值判断"""
    ma60 = df['close'].rolling(60).mean()
    slope = ma60.pct_change(5)
    return slope


def compute_total_score(features: pd.DataFrame, config: dict) -> pd.Series:
    """合成加权总分 total_score（等权 或 市场状态自适应权重）。

    - enabled=false（默认）：逐位复现历史逻辑
      total_score = Σ features[f].fillna(0) * config['weights'][f]，与旧版完全一致。
    - enabled=true：按每行 ma60_slope 的市场状态选权重——
      > ma60_slope_uptrend → uptrend 权重；< ma60_slope_downtrend → downtrend 权重；
      其余（含 ma60_slope 为 NaN 的行）→ sideways 权重。
      向量化：三组 mask 各算一次加权和，再用 np.where 组合。
      键集约定：权重键与 config['weights'] 一致（w.keys() 遍历），
      regime 权重中缺失的键按 0 计（total_score 可能因此小于 1 倍因子和，
      配置时须保证每个 regime 的权重和 = 1）。
    """
    w = config['weights']

    def weighted(weights: dict) -> pd.Series:
        # 与历史合成式一致：仅累加 config['weights'] 中存在的因子，缺失键按 0 计
        return sum(
            features[f].fillna(0) * weights.get(f, 0)
            for f in w.keys()
            if f in features.columns
        )

    rb = config.get('weights_by_regime', {})
    if rb.get('enabled', False):
        slope = features.get('ma60_slope')
        if slope is None:
            # 无 ma60_slope 列（旧缓存）：无法判定状态，全部按 sideways 权重
            return weighted(rb.get('sideways', {}))
        mask_up = slope > rb.get('ma60_slope_uptrend', 0.005)
        mask_dn = slope < rb.get('ma60_slope_downtrend', -0.005)
        score_up = weighted(rb.get('uptrend', {}))
        score_dn = weighted(rb.get('downtrend', {}))
        score_side = weighted(rb.get('sideways', {}))
        return pd.Series(
            np.where(mask_up, score_up, np.where(mask_dn, score_dn, score_side)),
            index=features.index,
        )
    return weighted(w)


def _compute_factor_df(df_sym: pd.DataFrame, df_bench: pd.DataFrame, config: dict) -> pd.DataFrame:
    """纯因子计算（不读写缓存）：对完整 df_sym/df_bench 算全部因子 + total_score。

    供 compute_all_features（增量/全量）与 compute_realtime_features（盘中实时）复用，
    保证两路径因子口径完全一致。
    """
    windows = {
        'momentum': config.get('momentum_window', 5),
        'trend': config.get('trend_window', 20),
        'atr': config.get('atr_window', 14),
        'rsrs': config.get('rsrs_window', 18),
        'rel_strength': config.get('rel_strength_window', 20),
    }

    features = df_sym[['date', 'close', 'volume']].copy()
    features['momentum'] = momentum_score(df_sym, windows['momentum'])
    features['trend'] = trend_score(df_sym, windows['trend'])
    features['volume_price'] = volume_price_score(df_sym)
    features['rsrs'] = rsrs_score(df_sym, windows['rsrs'])
    features['relative_strength'] = relative_strength_score(df_sym, df_bench, windows['rel_strength'])
    features['weekly_modifier'] = weekly_modifier(df_sym, config)
    features['ma60_slope'] = ma60_slope(df_sym)
    features['total_score'] = compute_total_score(features, config)
    return features


def compute_all_features(df_sym: pd.DataFrame, df_bench: pd.DataFrame, config: dict,
                         force: bool = False, symbol: str = None,
                         persist: bool = True) -> pd.DataFrame:
    """计算全量历史特征

    symbol 为空时取 config['symbol']（默认标的）；传入时按 data/{symbol} 读写缓存，
    因子计算本身与标的无关（相对强度用传入的 df_bench，默认基准 config['benchmark']）。
    persist=False 时不写 features_cache（盘中实时路径用，避免污染收盘口径缓存）。
    """
    symbol = symbol or config['symbol']

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

    # 算因子（纯计算，与实时路径共用同一实现）
    features = _compute_factor_df(compute_df, df_bench, config)

    # 合并缓存
    if not force and len(cached) > 0:
        combined = pd.concat([cached, features], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        # 全行重算 total_score：确保旧缓存行与新行同口径
        # （enabled=false 时与旧值逐位一致；enabled=true 时避免新旧口径混用）
        combined['total_score'] = compute_total_score(combined, config)
    else:
        combined = features

    if persist:
        save_features_cache(combined, symbol)
        print(f"  [OK] 特征缓存: {len(combined)} 条 ({combined['date'].min().date()} ~ {combined['date'].max().date()})")
    else:
        print(f"  [OK] 特征计算完成（不落盘）: {len(combined)} 条")
    return combined


def append_realtime_bar(df: pd.DataFrame, bar: dict) -> pd.DataFrame:
    """把盘中实时 bar 追加到日线 df 末尾（内存态，不落盘）

    bar 字段：date/open/high/low/close/volume/amount（与 daily.csv 列对齐）。
    若 bar.date 已存在于 df（同日重复调用），替换该行（实时刷新语义）；
    否则追加为最后一行。返回新 df（不改原对象）。
    """
    out = df.copy()
    cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount',
            'amplitude', 'change_pct', 'change', 'turnover', 'symbol']
    # 缺失价格/量字段置 NaN 而非 0：0 价会被因子函数当成真实数据产生看似合理的错误
    # total_score（0 价 bar 驱动实盘买卖 = 灾难）；NaN 则被因子函数自然排除/填充
    row = {
        'date': pd.Timestamp(bar['date']),
        'open': bar.get('open', float('nan')),
        'close': bar.get('close', float('nan')),
        'high': bar.get('high', float('nan')),
        'low': bar.get('low', float('nan')),
        'volume': bar.get('volume', float('nan')),
        'amount': bar.get('amount', float('nan')),
    }
    # symbol 列单独处理（须先于 fill 循环：否则 fill 先加了 symbol 列，
    # 后续判断恒 True 会把 symbol 填成 NaN）
    if 'symbol' not in out.columns:
        out['symbol'] = ''
    row['symbol'] = out['symbol'].iloc[0] if len(out) else ''
    for c in cols:
        if c not in out.columns:
            out[c] = float('nan') if c != 'date' else pd.NaT

    existing = out[out['date'] == row['date']]
    if len(existing) > 0:
        idx = existing.index[0]
        for c in cols:
            if c in row and c != 'date':
                out.at[idx, c] = row[c]
        return out.sort_values('date').reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out.sort_values('date').reset_index(drop=True)


def compute_realtime_features(symbol: str, bench_symbol: str, bar_sym: dict,
                              bar_bench: dict, config: dict) -> pd.DataFrame:
    """盘中实时特征：历史日线 + 实时 bar 拼接后全量重算（不落盘）

    返回完整特征 df（含实时行），由调用方取最后一行做决策。
    注意：
      - 不写 features_cache（收盘口径缓存保持纯净，次日增量重算时自动覆盖实时行）
      - 因子口径与 compute_all_features 完全一致（共用 _compute_factor_df）
      - 实时 volume 为当日累计量（腾讯=手，与 daily.csv 一致），
        volume_price 因子会用「半天累计量 vs 5 日均量」——盘中 14:25 量能天然偏低，
        属预期偏差（execution_timing 实验证明 total_score 相关 0.98，影响可控）
    """
    df_sym = load_data(symbol)
    df_bench = load_data(bench_symbol)
    if bar_sym:
        df_sym = append_realtime_bar(df_sym, bar_sym)
    else:
        # 防御：正常路径由 run_realtime 保证 bar_sym 非 None（None 直接兜底不进来），
        # 但本函数应可独立调用——缺失时显式告警而非静默用纯历史数据
        print("  [WARN] 标的实时 bar 缺失，特征仅含历史收盘数据（非实时口径）")
    if bar_bench:
        df_bench = append_realtime_bar(df_bench, bar_bench)
    else:
        # 基准实时 bar 缺失：relative_strength 因子（展示用，权重 0）将因 inner join
        # 丢弃当日行 → 该因子为 NaN。不影响 total_score（config.weights 无此项），
        # 但显式告警避免静默降级被误解为完整信号。
        print("  [WARN] 基准(000688)实时 bar 缺失，relative_strength 因子当日为 NaN"
              "（该因子权重 0，不影响 total_score）")
    return _compute_factor_df(df_sym, df_bench, config)


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
        'volume_price': round(latest.get('volume_price', 0), 3),
        'rsrs': round(latest.get('rsrs', 0), 3),
        'relative_strength': round(latest.get('relative_strength', 0), 3),
        'total_score': round(latest.get('total_score', 0), 3),
        'weekly_modifier': round(latest.get('weekly_modifier', 0), 3),
    }


def main():
    parser = argparse.ArgumentParser(description='计算 ETF 技术指标特征（默认 588000，可 --symbol 指定任意标的）')
    parser.add_argument('--symbol', default=None, help='标的代码（默认 config.json 的 symbol）')
    parser.add_argument('--benchmark', default=None, help='基准指数代码（默认 config.json 的 benchmark）')
    parser.add_argument('--force', action='store_true', help='全量重算')
    parser.add_argument('--output', choices=['panel', 'history', 'all'], default='panel',
                       help='输出内容：panel=最新因子 | history=全量历史 | all=两者')
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config['symbol']
    benchmark = args.benchmark or config['benchmark']
    print("\n📊 特征计算")
    df_sym = load_data(symbol)
    df_bench = load_data(benchmark)

    features = compute_all_features(df_sym, df_bench, config, args.force, symbol=symbol)

    if args.output in ('panel', 'all'):
        print("\n--- 最新因子 ---")
        latest = get_latest_features(symbol)
        for k, v in latest.items():
            print(f"  {k}: {v}")

    if args.output in ('history', 'all'):
        history_path = Path(PROJECT_ROOT) / config['data_dir'] / symbol / 'features_history.csv'
        features.to_csv(history_path, index=False)
        print(f"\n  [OK] 历史特征已写入: {history_path}")

    return features


if __name__ == '__main__':
    main()
