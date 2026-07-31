"""AkShare 数据源实现（主数据源）"""

import time
from typing import Optional

import pandas as pd

from .base import DataProvider, DataProviderError


class AkShareProvider(DataProvider):
    """通过 AkShare 获取 A 股 ETF / 指数日线数据"""

    @property
    def name(self) -> str:
        return "akshare"

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "20500101",
    ) -> Optional[pd.DataFrame]:
        import akshare as ak

        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        try:
            if symbol.startswith("5") or symbol.startswith("1"):
                # ETF
                df = ak.fund_etf_hist_em(
                    symbol=symbol, period="daily",
                    start_date=start, end_date=end, adjust="qfq"
                )
            else:
                # 指数
                sh_symbol = f"sh{symbol}" if not symbol.startswith("sh") else symbol
                df = ak.stock_zh_index_daily_em(symbol=sh_symbol)
                df = df[df["date"].str.replace("-", "", regex=False) >= start].copy()

            if df is None or len(df) == 0:
                return None
            return df

        except Exception as e:
            raise DataProviderError(f"AkShare 请求失败 ({symbol}): {e}") from e

    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude",
            "涨跌幅": "change_pct", "涨跌额": "change",
            "换手率": "turnover",
        }

        # 尝试中文映射，若列名已是英文则直接通过
        rename = {}
        for col in df.columns:
            if col in col_map:
                rename[col] = col_map[col]

        if rename:
            df = df.rename(columns=rename)

        # 确保必需列存在
        required = ["date", "open", "close", "high", "low", "volume"]
        for col in required:
            if col not in df.columns:
                if col == "date":
                    # 指数数据可能用 "日期" 已被映射
                    pass
                raise KeyError(f"AkShare 标准化失败：缺少 {col}")

        # 补齐可选列
        for opt in ["amount", "amplitude", "change_pct", "change", "turnover"]:
            if opt not in df.columns:
                df[opt] = 0

        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df = df[
            ["date", "open", "close", "high", "low", "volume", "amount",
             "amplitude", "change_pct", "change", "turnover", "symbol"]
        ]
        return df.sort_values("date").reset_index(drop=True)
