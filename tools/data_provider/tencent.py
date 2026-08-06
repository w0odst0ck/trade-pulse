"""腾讯数据源（独立备用源，与东财系完全独立）"""

from typing import List, Optional

import pandas as pd
import requests

from .base import DataProvider, DataProviderError


class TencentProvider(DataProvider):
    """通过腾讯网页接口（web.ifzq.gtimg.cn）直拉日线数据

    特点：
    - 与东财系（AkShare/EastMoney）完全独立，互为灾备
    - 免费，无登录，盘后即时更新（当天可得）
    - 返回前复权日K（qfqday），历史覆盖深
    - 单次请求有根数上限（实测 count=1000 只回 640 根），
      必须按自然年分段循环拉取后拼接去重，保证区间全覆盖
    """

    BASE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://gu.qq.com/",
    }

    @property
    def name(self) -> str:
        return "tencent"

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

    def _fetch_segment(self, tencent_symbol: str, seg_start: str, seg_end: str) -> List[str]:
        """拉取单段（单自然年）K线，返回原始行列表"""
        params = f"{tencent_symbol},day,{seg_start},{seg_end},1000,qfq"
        url = f"{self.BASE_URL}?param={params}"
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            data = resp.json()
        except Exception as e:
            raise DataProviderError(f"腾讯请求失败 ({tencent_symbol} {seg_start}~{seg_end}): {e}") from e

        # API 级错误（非 200 / code != 0）视为故障：抛异常让上层走 fallback，
        # 而不是静默当"标的无数据"处理；仅请求成功且无 klines 才返回空。
        # （getattr 兼容单测中的简化 mock 响应对象）
        if getattr(resp, "status_code", 200) != 200:
            raise DataProviderError(
                f"腾讯接口 HTTP {resp.status_code} ({tencent_symbol} {seg_start}~{seg_end})"
            )
        if isinstance(data, dict) and data.get("code") not in (0, None):
            raise DataProviderError(
                f"腾讯接口返回错误 code={data.get('code')} ({tencent_symbol} {seg_start}~{seg_end})"
            )

        node = (data.get("data") or {}).get(tencent_symbol) or {}
        return node.get("qfqday") or node.get("day") or []

    @staticmethod
    def _parse_line(line) -> Optional[List[str]]:
        """单行解析：接口每行返回 list（如 ['2023-01-03', '1.001', ...]），
        兼容 CSV 字符串形式；取前 6 字段 [日期, 开盘, 收盘, 最高, 最低, 成交量]"""
        if isinstance(line, (list, tuple)):
            parts = [str(x) for x in line]
        else:
            parts = str(line).split(",")
        if len(parts) < 6:
            return None
        return parts[:6]

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "20500101",
    ) -> Optional[pd.DataFrame]:
        """直连腾讯 fqkline 接口拉取日线

        按自然年分段循环（start_date ~ end_date 每年一段），拼接后去重，
        保证区间全覆盖（单次请求有根数上限，不能一次拉完）。
        每行按逗号分割取前 6 字段：[日期, 开盘, 收盘, 最高, 最低, 成交量]。

        symbol 可带市场前缀（如 "sh588000"、"sh000688"），此时直接使用；
        纯数字代码（如 "000688"）由 _market_prefix 兜底补前缀。
        """
        try:
            start = start_date.replace("-", "")
            end = end_date.replace("-", "")
            if symbol[:2].lower() in ("sh", "sz", "bj"):
                tencent_symbol = symbol  # 已带市场前缀，直接使用，不再重复拼
            else:
                tencent_symbol = f"{self._market_prefix(symbol)}{symbol}"
            start_year = int(start[:4])
            end_year = int(end[:4])

            raw_rows: List[List[str]] = []
            for year in range(start_year, end_year + 1):
                seg_start = f"{year}-01-01"
                seg_end = f"{year}-12-31"
                if year == start_year:
                    seg_start = f"{start[:4]}-{start[4:6]}-{start[6:]}"
                if year == end_year:
                    seg_end = f"{end[:4]}-{end[4:6]}-{end[6:]}"
                for line in self._fetch_segment(tencent_symbol, seg_start, seg_end):
                    parts = self._parse_line(line)
                    if parts is not None:
                        raw_rows.append(parts)

            if not raw_rows:
                return None

            df = pd.DataFrame(
                raw_rows, columns=["date", "open", "close", "high", "low", "volume"]
            )
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["date"])
            # 跨年分段可能重叠/重复，按 date 去重后升序
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            return df
        except DataProviderError:
            raise
        except Exception as e:
            raise DataProviderError(f"腾讯请求失败 ({symbol}): {e}") from e

    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """标准化列名与扩展列（与 akshare.normalize 行为一致）"""
        df = df.copy()
        required = ["date", "open", "close", "high", "low", "volume"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"腾讯标准化失败：缺少 {col}")

        # 补齐可选列（腾讯接口不返回这些指标，与 akshare.normalize 一致填 0）
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
