"""baostock 数据源（独立备用源，东财系挂掉时可用）"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import DataProvider, DataProviderError


class BaoStockProvider(DataProvider):
    """通过 baostock 免费接口拉取日线数据

    特点：
    - 与东财系（AkShare/EastMoney）完全独立，互为灾备
    - 免费，无频率限制（登录式 API）
    - 日线历史深度大（可到 1990s），分钟线约 240 天
    - 注意：baostock 日线当日数据有延迟（收盘结算后才有当天）
    """

    @property
    def name(self) -> str:
        return "baostock"

    def _login(self):
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise DataProviderError(f"baostock login 失败: {lg.error_msg}")
        return bs

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "20500101",
    ) -> Optional[pd.DataFrame]:
        try:
            bs = self._login()
            try:
                # baostock 日期格式要求 YYYY-MM-DD（不接收 YYYYMMDD）
                start = start_date.replace("-", "")
                end = end_date.replace("-", "")
                if len(start) == 8:
                    start = f"{start[:4]}-{start[4:6]}-{start[6:]}"
                if len(end) == 8:
                    end = f"{end[:4]}-{end[4:6]}-{end[6:]}"
                rs = bs.query_history_k_data_plus(
                    f"sh.{symbol}",
                    "date,code,open,high,low,close,volume,amount",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="2",  # 前复权
                )
                if rs.error_code != "0":
                    raise DataProviderError(f"baostock 查询失败: {rs.error_code} {rs.error_msg}")

                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())

                if not rows:
                    return None

                df = pd.DataFrame(rows, columns=["date", "code", "open", "high", "low", "close", "volume", "amount"])
                df = df.drop(columns=["code"])
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df["date"] = pd.to_datetime(df["date"])
                df["symbol"] = symbol
                return df.sort_values("date").reset_index(drop=True)
            finally:
                bs.logout()
        except DataProviderError:
            raise
        except Exception as e:
            raise DataProviderError(f"baostock 请求失败 ({symbol}): {e}") from e

    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """baostock 列已标准化；补齐与 AkShare 一致的扩展列"""
        df = df.copy()
        # 计算派生列（与 AkShare 输出对齐）
        if "change_pct" not in df.columns:
            df["change_pct"] = df["close"].pct_change() * 100
        if "change" not in df.columns:
            df["change"] = df["close"].diff()
        if "amplitude" not in df.columns:
            prev_close = df["close"].shift(1).replace(0, np.nan)
            df["amplitude"] = (df["high"] - df["low"]) / prev_close * 100
        if "turnover" not in df.columns:
            df["turnover"] = 0.0
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        # 首行派生列必然 NaN → 填 0，保证 CSV 无 NaN/inf
        for col in ["change_pct", "change", "amplitude"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([float("inf"), float("-inf")], 0.0)
        return df.sort_values("date").reset_index(drop=True)
