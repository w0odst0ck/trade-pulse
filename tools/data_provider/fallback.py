"""东方财富网页接口（备用数据源，AkShare 不可用时自动降级）"""

from typing import Optional

import pandas as pd
import requests

from .base import DataProvider, DataProviderError


class EastMoneyProvider(DataProvider):
    """通过东方财富网页接口直拉数据"""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
    }

    @property
    def name(self) -> str:
        return "eastmoney"

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "20500101",
    ) -> Optional[pd.DataFrame]:
        secid = f"1.{symbol}"
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        url = (
            "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
            f"&klt=101&fqt=1&beg={start}&end={end}"
        )

        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            data = resp.json()
        except Exception as e:
            raise DataProviderError(f"东方财富请求失败 ({symbol}): {e}") from e

        if data.get("data") is None or data["data"].get("klines") is None:
            return None

        rows = []
        for line in data["data"]["klines"]:
            parts = line.split(",")
            rows.append({
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
                "amplitude": float(parts[7]) if len(parts) > 7 else 0,
                "change_pct": float(parts[8]) if len(parts) > 8 else 0,
                "change": float(parts[9]) if len(parts) > 9 else 0,
                "turnover": float(parts[10]) if len(parts) > 10 else 0,
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        return df.sort_values("date").reset_index(drop=True)

    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # EastMoney 返回格式已标准化，无需额外映射
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        return df.sort_values("date").reset_index(drop=True)
