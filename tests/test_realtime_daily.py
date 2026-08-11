#!/usr/bin/env python3
"""realtime_daily.py 单元测试（双线并行：快照积累 + 回填）"""

import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "daily_pipeline"))

import realtime_daily as rd


def _bar(close=1.835, prefix_key=None, **kw):
    b = {
        'date': date(2026, 8, 12), 'open': 1.84, 'high': 1.851, 'low': 1.799,
        'close': close, 'volume': 34364399.0, 'amount': 6255060479.0,
        'prev_close': 1.84, 'source': 'tencent',
    }
    b.update(kw)
    return b


class TestSnapshotPath(unittest.TestCase):
    """路径解析（mock config）"""

    def test_path_under_data_dir(self):
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp)
            cfg_path = proj / "tools" / "daily_pipeline" / "config.json"
            cfg_path.parent.mkdir(parents=True)
            cfg_path.write_text(_json.dumps({"data_dir": "data", "symbol": "588000"}),
                                encoding="utf-8")
            with mock.patch.object(rd, 'PROJECT_ROOT', proj):
                p = rd.snapshot_path('588000')
        self.assertIn('588000', str(p))
        self.assertTrue(str(p).endswith('realtime_daily.csv'))


class TestWriteSnapshot(unittest.TestCase):
    """列级幂等 upsert"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proj = Path(self.tmp.name)
        # mock 数据目录到临时目录
        cfg_path = self.proj / "tools" / "daily_pipeline" / "config.json"
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text(json.dumps({"data_dir": "data", "symbol": "588000"}),
                            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_1425_creates_row(self):
        with mock.patch.object(rd, 'PROJECT_ROOT', self.proj):
            ok = rd.write_snapshot('588000', '1425', bar=_bar(close=1.83))
        self.assertTrue(ok)
        # 直接读文件
        p = self.proj / "data" / "588000" / "realtime_daily.csv"
        self.assertTrue(p.exists())
        import pandas as pd
        df = pd.read_csv(p)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['close_1425'], 1.83)
        self.assertTrue(pd.isna(df.iloc[0]['close_1450']))  # 1450 列未写

    def test_second_prefix_upserts_same_row(self):
        """14:25 写后 14:50 写 → 同一行两列都有，不新增行"""
        with mock.patch.object(rd, 'PROJECT_ROOT', self.proj):
            rd.write_snapshot('588000', '1425', bar=_bar(close=1.83))
            rd.write_snapshot('588000', '1450', bar=_bar(close=1.835))
        import pandas as pd
        p = self.proj / "data" / "588000" / "realtime_daily.csv"
        df = pd.read_csv(p)
        self.assertEqual(len(df), 1)  # 同一行
        self.assertEqual(df.iloc[0]['close_1425'], 1.83)
        self.assertEqual(df.iloc[0]['close_1450'], 1.835)

    def test_dirty_bar_not_written(self):
        """脏数据（close<=0）→ 不写入"""
        with mock.patch.object(rd, 'PROJECT_ROOT', self.proj):
            ok = rd.write_snapshot('588000', '1425', bar=_bar(close=0.0))
        self.assertFalse(ok)
        p = self.proj / "data" / "588000" / "realtime_daily.csv"
        self.assertFalse(p.exists())  # 文件都没创建

    def test_bar_none_skips(self):
        """实时不可用 → 留空不写"""
        with mock.patch.object(rd, 'PROJECT_ROOT', self.proj), \
             mock.patch.object(rd, 'fetch_realtime_bar', return_value=None):
            ok = rd.write_snapshot('588000', '1425')
        self.assertFalse(ok)
        p = self.proj / "data" / "588000" / "realtime_daily.csv"
        self.assertFalse(p.exists())


class TestBackfillFinal(unittest.TestCase):
    """收盘回填真值"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proj = Path(self.tmp.name)
        cfg_path = self.proj / "tools" / "daily_pipeline" / "config.json"
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text(json.dumps({"data_dir": "data", "symbol": "588000"}),
                            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _make_daily(self, today_close=1.85):
        import pandas as pd
        daily_dir = self.proj / "data" / "588000"
        daily_dir.mkdir(parents=True)
        df = pd.DataFrame({
            'date': [pd.Timestamp('2026-08-11'), pd.Timestamp('2026-08-12')],
            'open': [1.84, 1.84], 'high': [1.86, 1.86], 'low': [1.80, 1.80],
            'close': [1.83, today_close], 'volume': [4e7, 4.5e7],
        })
        df.to_csv(daily_dir / "daily.csv", index=False)

    def test_backfill_updates_existing_row(self):
        import pandas as pd
        self._make_daily()
        with mock.patch.object(rd, 'PROJECT_ROOT', self.proj), \
             mock.patch.object(rd, 'date') as mock_date:
            mock_date.today.return_value = date(2026, 8, 12)
            # 先写 14:50 快照
            rd.write_snapshot('588000', '1450', bar=_bar(close=1.835))
            ok = rd.backfill_final('588000')
        self.assertTrue(ok)
        p = self.proj / "data" / "588000" / "realtime_daily.csv"
        df = pd.read_csv(p)
        self.assertEqual(df.iloc[0]['close_1450'], 1.835)
        self.assertEqual(df.iloc[0]['close_final'], 1.85)  # 回填真值
        self.assertEqual(len(df), 1)

    def test_backfill_no_daily_row_returns_false(self):
        """daily.csv 无当日行（盘后未发布）→ 返回 False，final 留空"""
        self._make_daily()
        with mock.patch.object(rd, 'PROJECT_ROOT', self.proj), \
             mock.patch.object(rd, 'date') as mock_date:
            mock_date.today.return_value = date(2026, 8, 13)  # daily 没有 08-13
            ok = rd.backfill_final('588000')
        self.assertFalse(ok)


class TestAnalysis(unittest.TestCase):
    """偏差统计"""

    def test_price_deviation(self):
        import pandas as pd
        df = pd.DataFrame({
            'date': ['2026-08-01', '2026-08-02'],
            'close_1425': [1.80, 1.90], 'close_1450': [1.82, 1.92],
            'close_final': [1.83, 1.90],
            'volume_1425': [3e7, 3.5e7], 'volume_1450': [3.5e7, 4e7],
            'volume_final': [4e7, 4.2e7],
        })
        from realtime_vs_close_analysis import analyze_price_deviation, analyze_volume_ratio
        res = analyze_price_deviation(df)
        self.assertEqual(res['n'], 2)
        # 1.82 vs 1.83 → -0.55%; 1.92 vs 1.90 → +1.05%
        self.assertAlmostEqual(res['mean_pct'], 0.25, delta=0.01)
        vol = analyze_volume_ratio(df)
        self.assertEqual(vol['n'], 2)
        self.assertAlmostEqual(vol['mean_ratio'], (3.5/4 + 4/4.2) / 2, delta=0.01)


if __name__ == '__main__':
    unittest.main()
