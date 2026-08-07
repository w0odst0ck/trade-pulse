"""新浪数据源（独立域名，与腾讯/东财系互不依赖）"""

from typing import Optional

import pandas as pd

from .base import DataProvider, DataProviderError


class SinaProvider(DataProvider):
    """通过 akshare 封装的新浪接口拉取日线数据

    特点：
    - 独立域名（finance.sina.com.cn），与腾讯（gtimg）和东财系完全独立
    - 历史更深（约 2020 起），当天可得性比腾讯好
    - 接口对连续请求有限流（TypeError），实现内单次调用即可，重试交给上层多源调度
    - ETF（5/1 开头）走 fund_etf_hist_sina，volume 单位=股（腾讯=手，差 100 倍，
      由 normalize 统一 ÷100 对齐）；指数走 stock_zh_index_daily
    - 000688 是科创50指数（sh000688），与深市个股 sz000688 靠 config['markets']
      或显式前缀区分；市场前缀映射复用 fetch_data.resolve_tencent_symbol 逻辑
      （上层调用时传入带前缀 symbol），本类内按代码首字符推断兜底
    """

    @property
    def name(self) -> str:
        return "sina"

    @staticmethod
    def _market_prefix(symbol: str) -> str:
        """按证券代码首字符映射市场前缀：6/5/9 -> sh，0/1/3 -> sz，8/4 -> bj"""
        if symbol[:2].lower() in ("sh", "sz", "bj"):
            return symbol[:2].lower()
        first = symbol[0]
        if first in "659":
            return "sh"
        if first in "013":
            return "sz"
        if first in "84":
            return "bj"
        raise DataProviderError(f"无法识别市场前缀: {symbol}")

    @staticmethod
    def _is_etf(symbol: str) -> bool:
        """ETF 判定：代码首字符为 5（沪 ETF）或 1（深 ETF/LOF）"""
        code = symbol[2:] if symbol[:2].lower() in ("sh", "sz", "bj") else symbol
        return code[:1] in ("5", "1")

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "20500101",
    ) -> Optional[pd.DataFrame]:
        import akshare as ak

        try:
            if symbol[:2].lower() in ("sh", "sz", "bj"):
                prefixed = symbol
            else:
                prefixed = f"{self._market_prefix(symbol)}{symbol}"

            if self._is_etf(prefixed):
                df = ak.fund_etf_hist_sina(symbol=prefixed)
            else:
                df = ak.stock_zh_index_daily(symbol=prefixed)

            if df is None or len(df) == 0:
                return None

            # 新浪接口无 start/end 参数（全量返回），本地按日期过滤
            df = df.copy()
            dstr = df["date"].astype(str).str.replace("-", "", regex=False)
            start = start_date.replace("-", "")
            end = end_date.replace("-", "")
            df = df[(dstr >= start) & (dstr <= end)].reset_index(drop=True)
            if len(df) == 0:
                return None
            return df
        except DataProviderError:
            raise
        except Exception as e:
            raise DataProviderError(f"新浪请求失败 ({symbol}): {e}") from e

    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """标准化列名与扩展列（与 tencent.normalize 行为一致）

        新浪原始列：date/open/high/low/close/volume/amount/postVol/postAmt
        - volume 股→手（÷100）对齐腾讯口径（实测 ETF 与指数的新浪接口均返回股）
        - 补齐 amount/amplitude/change_pct/change/turnover 空列
        """
        df = df.copy()
        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"新浪标准化失败：缺少 {col}")

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 新浪 volume 单位=股（ETF 与指数接口均如此），腾讯=手（100 股 = 1 手）
        df["volume"] = df["volume"] / 100.0

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
