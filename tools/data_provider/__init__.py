"""
data_provider — 数据源抽象层

InstOrderTrace 的数据获取统一接口。当前实现：
- AkShareProvider：主数据源（ETF 日线）
- EastMoneyProvider：备用数据源（AkShare 不可用时自动降级）
"""

from .base import DataProvider, DataProviderError
from .akshare import AkShareProvider
from .fallback import EastMoneyProvider

__all__ = ["DataProvider", "DataProviderError", "AkShareProvider", "EastMoneyProvider"]
