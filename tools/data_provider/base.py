"""数据提供者抽象接口"""
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataProviderError(Exception):
    """数据获取异常"""
    pass


class DataProvider(ABC):
    """数据源接口，所有 Provider 必须实现"""

    @abstractmethod
    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "20500101",
    ) -> Optional[pd.DataFrame]:
        """获取日线数据
        
        Parameters
        ----------
        symbol : str
            标的代码，如 "588000"、"000688"
        start_date : str
            起始日期，格式 "YYYY-MM-DD" 或 "YYYYMMDD"
        end_date : str
            结束日期，默认到 2050 年

        Returns
        -------
        pd.DataFrame | None
            包含列: date, open, close, high, low, volume, amount, ...
            失败时返回 None
        """
        ...

    @abstractmethod
    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """标准化列名为统一英文格式
        
        输入：源数据的列名（可能是中文）
        输出：统一英文列名 + 排序 + symbol 列
        
        要求的输出列：
          date, open, close, high, low, volume, amount,
          amplitude, change_pct, change, turnover, symbol
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称，用于日志"""
        ...
