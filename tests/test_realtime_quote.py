#!/usr/bin/env python3
"""realtime_quote.py 单元测试（方案 B：盘中实时行情 + 稳定性/兜底）"""

import json
import sys
import unittest
from datetime import datetime, date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "daily_pipeline"))

import realtime_quote as rq


# 腾讯实时接口样例（实测 08-10 返回，GBK 解码后）
TENCENT_SAMPLE = (
    'v_sh588000="1~科创50ETF华夏~588000~1.835~1.840~1.840~34364399~16043013~18321386~'
    '1.834~108532~1.833~33517~1.832~12928~1.831~15351~1.830~12741~1.835~59477~'
    '1.836~21993~1.837~9856~1.838~32634~1.839~29320~~20260810161457~-0.005~-0.27~'
    '1.851~1.799~1.835/34364399/6255060479~34364399~625506~6.93~~~1.851~1.799~2.83~'
    '909.70~909.70~0.00~2.208~1.472~0.62~29789~1.820~~~~~~625506.0479~378.6339~'
    '20634~   A~ETF~29.50~12.16~~~~2.390~1.096~-3.98~-12.62~2.69~49574668200~'
    '49574668200~8.86~26.64~49574668200~-0.04~1.8358~67.43~0.11~1.8345~CNY~0~'
    '___D__F__N~1.840~-76073~";'
)

# 新浪实时接口样例（真实格式：32+ 字段，逗号分割）
SINA_SAMPLE = (
    'var hq_str_sh588000="科创50ETF华夏,1.840,1.840,1.835,1.851,1.799,'
    '1.834,1.835,3436439900,6255060479,16043013,18321386,'
    '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
    '2026-08-10,16:14:57,00,";'
)


def _fake_get(text: str, encoding: str = "gbk"):
    """构造 requests.get mock 返回对象"""
    resp = mock.MagicMock()
    resp.text = text
    resp.status_code = 200
    return resp


class TestIsMarketOpen(unittest.TestCase):
    """时段守卫：交易日 + 连续竞价时段"""

    def test_market_open_afternoon(self):
        # 14:50 周三（2026-08-12 是周三，交易日）
        with mock.patch.object(rq, 'is_trading_day', return_value=True):
            self.assertTrue(rq.is_market_open(datetime(2026, 8, 12, 14, 50)))

    def test_market_open_morning(self):
        with mock.patch.object(rq, 'is_trading_day', return_value=True):
            self.assertTrue(rq.is_market_open(datetime(2026, 8, 12, 10, 0)))

    def test_market_closed_after_15(self):
        with mock.patch.object(rq, 'is_trading_day', return_value=True):
            self.assertFalse(rq.is_market_open(datetime(2026, 8, 12, 15, 30)))

    def test_market_closed_before_930(self):
        with mock.patch.object(rq, 'is_trading_day', return_value=True):
            self.assertFalse(rq.is_market_open(datetime(2026, 8, 12, 9, 0)))

    def test_non_trading_day(self):
        with mock.patch.object(rq, 'is_trading_day', return_value=False):
            self.assertFalse(rq.is_market_open(datetime(2026, 8, 12, 14, 50)))

    def test_lunch_break(self):
        with mock.patch.object(rq, 'is_trading_day', return_value=True):
            self.assertFalse(rq.is_market_open(datetime(2026, 8, 12, 12, 0)))


class TestParseTencent(unittest.TestCase):
    """腾讯实时解析"""

    def test_parse_valid(self):
        with mock.patch('requests.get', return_value=_fake_get(TENCENT_SAMPLE)) as m:
            bar = rq._fetch_tencent('sh588000')
        self.assertIsNotNone(bar)
        self.assertEqual(bar['close'], 1.835)
        self.assertEqual(bar['open'], 1.840)
        self.assertEqual(bar['high'], 1.851)
        self.assertEqual(bar['low'], 1.799)
        self.assertEqual(bar['volume'], 34364399)  # 手，与 daily.csv 一致
        self.assertEqual(bar['amount'], 6255060479)
        self.assertEqual(bar['source'], 'tencent')
        m.assert_called_once()

    def test_parse_bare_symbol_prefix(self):
        """纯数字代码自动补 sh 前缀"""
        with mock.patch('requests.get', return_value=_fake_get(TENCENT_SAMPLE)) as m:
            rq._fetch_tencent('588000')
        url = m.call_args[0][0]
        self.assertIn('sh588000', url)

    def test_parse_000688_prefix_via_config(self):
        """000688 必须解析为 sh（config['markets'] 显式映射：科创50指数），
        不能按首字符推断成 sz（深市股票）——否则拉错标的"""
        self.assertEqual(rq._resolve_prefix('000688'), 'sh')
        with mock.patch('requests.get', return_value=_fake_get(TENCENT_SAMPLE)) as m:
            rq._fetch_tencent('000688')
        url = m.call_args[0][0]
        self.assertIn('sh000688', url)
        self.assertNotIn('sz000688', url)

    def test_prefix_fallback_first_char(self):
        """config 无映射时按首字符兜底（0/1/3→sz）"""
        with mock.patch('realtime_quote._load_markets', return_value={}):
            self.assertEqual(rq._resolve_prefix('159915'), 'sz')
            self.assertEqual(rq._resolve_prefix('588000'), 'sh')

    def test_parse_request_failure(self):
        with mock.patch('requests.get', side_effect=Exception('conn refused')):
            self.assertIsNone(rq._fetch_tencent('sh588000'))

    def test_parse_empty_response(self):
        with mock.patch('requests.get', return_value=_fake_get('')):
            self.assertIsNone(rq._fetch_tencent('sh588000'))

    def test_parse_garbage(self):
        with mock.patch('requests.get', return_value=_fake_get('not a quote')):
            self.assertIsNone(rq._fetch_tencent('sh588000'))


class TestParseSina(unittest.TestCase):
    """新浪实时解析"""

    def test_parse_valid(self):
        with mock.patch('requests.get', return_value=_fake_get(SINA_SAMPLE)) as m:
            bar = rq._fetch_sina('sh588000')
        self.assertIsNotNone(bar)
        self.assertEqual(bar['close'], 1.835)
        self.assertEqual(bar['open'], 1.840)
        self.assertEqual(bar['high'], 1.851)
        self.assertEqual(bar['low'], 1.799)
        # 新浪 volume 单位=股 → ÷100 对齐手
        self.assertEqual(bar['volume'], 34364399.0)
        self.assertEqual(bar['amount'], 6255060479)
        self.assertEqual(bar['source'], 'sina')
        # 必须带 Referer（新浪反爬要求）
        headers = m.call_args[1]['headers']
        self.assertIn('Referer', headers)

    def test_parse_failure(self):
        with mock.patch('requests.get', side_effect=Exception('timeout')):
            self.assertIsNone(rq._fetch_sina('sh588000'))

    def test_index_volume_no_divide(self):
        """指数（000688）新浪 volume=手，不 ÷100（实测 2026-08-11 验证）"""
        idx_sample = (
            'var hq_str_sh000688="科创50,1713.2581,1737.7737,1729.4453,1733.1611,1699.9151,'
            '1729.4300,1729.5100,4664714,43223276261,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
            '2026-08-11,10:15:00,00,";'
        )
        with mock.patch('requests.get', return_value=_fake_get(idx_sample)) as m:
            bar = rq._fetch_sina('sh000688')
        self.assertIsNotNone(bar)
        # 指数 volume=手，不换算（4664714 手 ≈ 腾讯 4669284 手，同日同源量级一致）
        self.assertEqual(bar['volume'], 4664714.0)
        # 与腾讯源量级一致（不差 100 倍）
        self.assertGreater(bar['volume'], 4e6)


class TestSanityCheck(unittest.TestCase):
    """数据合理性校验"""

    def setUp(self):
        self.good = {
            'open': 1.84, 'high': 1.851, 'low': 1.799,
            'close': 1.835, 'volume': 34364399,
        }

    def test_good_bar(self):
        ok, issues = rq.sanity_check(self.good, prev_close=1.840)
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_high_less_than_close(self):
        bar = dict(self.good, high=1.80)  # high < close 1.835
        ok, issues = rq.sanity_check(bar)
        self.assertFalse(ok)
        self.assertTrue(any('high' in i for i in issues))

    def test_low_greater_than_open(self):
        bar = dict(self.good, low=1.85)  # low > open 1.84
        ok, issues = rq.sanity_check(bar)
        self.assertFalse(ok)

    def test_negative_price(self):
        bar = dict(self.good, close=-1.0)
        ok, _ = rq.sanity_check(bar)
        self.assertFalse(ok)

    def test_nan_price(self):
        bar = dict(self.good, close=float('nan'))
        ok, _ = rq.sanity_check(bar)
        self.assertFalse(ok)

    def test_zero_volume(self):
        bar = dict(self.good, volume=0)
        ok, _ = rq.sanity_check(bar)
        self.assertFalse(ok)

    def test_deviation_over_20pct(self):
        """close 与昨收偏差 > 20% → 脏数据拦截"""
        bar = dict(self.good, close=2.3)  # 昨收 1.84，偏差 25%
        ok, issues = rq.sanity_check(bar, prev_close=1.840)
        self.assertFalse(ok)
        self.assertTrue(any('偏差' in i for i in issues))

    def test_deviation_within_20pct_ok(self):
        # 整体抬高：close 偏差 8.7%，high/low 同步抬高避免 high<close 矛盾
        bar = dict(self.good, close=2.0, high=2.05, low=1.95, open=2.0)
        ok, _ = rq.sanity_check(bar, prev_close=1.840)
        self.assertTrue(ok)


class TestFetchRealtimeBar(unittest.TestCase):
    """双源互备 + 交叉验证 + 兜底"""

    NOW = datetime(2026, 8, 12, 14, 50)  # 交易日盘中

    def _run(self, tx=None, sina=None, prev_close=1.84):
        """注入假拉取函数（mock 今日=2026-08-12，与 _bar 日期一致）"""
        fns = (
            (lambda s: tx) if tx is not None else (lambda s: None),
            (lambda s: sina) if sina is not None else (lambda s: None),
        )
        fake_today = type('FakeDate', (), {'today': staticmethod(lambda: date(2026, 8, 12))})()
        with mock.patch.object(rq, 'is_market_open', return_value=True), \
             mock.patch.object(rq, 'date_cls', fake_today):
            return rq.fetch_realtime_bar('sh588000', prev_close=prev_close, _fetch_fns=fns)

    def _bar(self, close=1.835, **kw):
        b = {
            'date': date(2026, 8, 12), 'open': 1.84, 'high': 1.851, 'low': 1.799,
            'close': close, 'volume': 34364399.0, 'amount': 6255060479.0,
            'prev_close': 1.84, 'source': 'tencent',
        }
        b.update(kw)
        return b

    def test_tencent_only(self):
        bar = self._run(tx=self._bar())
        self.assertIsNotNone(bar)
        self.assertEqual(bar['source'], 'tencent')

    def test_sina_only(self):
        bar = self._run(sina=self._bar(source='sina'))
        self.assertIsNotNone(bar)
        self.assertEqual(bar['source'], 'sina')

    def test_both_consistent_use_tencent(self):
        bar = self._run(tx=self._bar(), sina=self._bar(source='sina'))
        self.assertEqual(bar['source'], 'tencent')

    def test_conflict_use_tencent(self):
        """双源 close 差 > 2% → 冲突，采用腾讯 + 告警"""
        bar = self._run(tx=self._bar(), sina=self._bar(close=2.0, source='sina'))
        self.assertEqual(bar['source'], 'tencent')
        self.assertEqual(bar['close'], 1.835)

    def test_both_fail_returns_none(self):
        bar = self._run(tx=None, sina=None)
        self.assertIsNone(bar)

    def test_tencent_invalid_falls_back_sina(self):
        """腾讯数据不合法（high < close）→ 新浪兜底"""
        bad_tx = self._bar(high=1.0)
        bar = self._run(tx=bad_tx, sina=self._bar(source='sina'))
        self.assertEqual(bar['source'], 'sina')

    def test_sina_invalid_falls_back_tencent(self):
        bad_sina = self._bar(low=1.9, source='sina')  # low > open
        bar = self._run(tx=self._bar(), sina=bad_sina)
        self.assertEqual(bar['source'], 'tencent')

    def test_deviation_filters_both(self):
        """双源都超偏差 → None（宁可无数据不可用脏数据）"""
        bar = self._run(tx=self._bar(close=3.0), sina=self._bar(close=3.1, source='sina'))
        self.assertIsNone(bar)

    def test_stale_bar_date_rejected(self):
        """bar 日期 != 今日 → 拒绝（返回 None 走兜底），绝不能当今日盘中数据驱动信号"""
        stale = self._bar(date=date(2026, 8, 11))  # 昨日数据
        bar = self._run(tx=stale, sina=None)
        self.assertIsNone(bar)

    def test_stale_bar_both_rejected_fallback_sina_valid(self):
        """腾讯返回昨日 bar 被拒 → 新浪今日 bar 兜底"""
        stale = self._bar(date=date(2026, 8, 11))
        fresh = self._bar(source='sina')
        bar = self._run(tx=stale, sina=fresh)
        self.assertIsNotNone(bar)
        self.assertEqual(bar['source'], 'sina')

    def test_market_closed_returns_none(self):
        with mock.patch.object(rq, 'is_market_open', return_value=False):
            bar = rq.fetch_realtime_bar('sh588000', _fetch_fns=(lambda s: self._bar(), lambda s: self._bar()))
        self.assertIsNone(bar)

    def test_injected_fetch_exception(self):
        """拉取函数抛异常 → 不崩，走另一源"""
        def boom(s):
            raise RuntimeError('boom')
        fake_today = type('FakeDate', (), {'today': staticmethod(lambda: date(2026, 8, 12))})()
        with mock.patch.object(rq, 'is_market_open', return_value=True), \
             mock.patch.object(rq, 'date_cls', fake_today):
            bar = rq.fetch_realtime_bar('sh588000', prev_close=1.84,
                                        _fetch_fns=(boom, lambda s: self._bar(source='sina')))
        self.assertEqual(bar['source'], 'sina')


class TestIntegrationParse(unittest.TestCase):
    """真实接口文本 → fetch_realtime_bar 全链路（mock requests 层）"""

    def test_tencent_text_full_chain(self):
        """腾讯真实文本 → 双源互备 → bar"""
        def fake_get(url, headers=None, timeout=None):
            if 'qt.gtimg' in url:
                return _fake_get(TENCENT_SAMPLE)
            if 'hq.sinajs' in url:
                return _fake_get(SINA_SAMPLE)
            return _fake_get('')
        fake_today = type('FakeDate', (), {'today': staticmethod(lambda: date(2026, 8, 10))})()
        with mock.patch('requests.get', side_effect=fake_get), \
             mock.patch.object(rq, 'is_market_open', return_value=True), \
             mock.patch.object(rq, 'date_cls', fake_today):
            bar = rq.fetch_realtime_bar('588000', prev_close=1.840)
        self.assertIsNotNone(bar)
        self.assertEqual(bar['close'], 1.835)
        self.assertEqual(bar['volume'], 34364399)
        self.assertEqual(bar['source'], 'tencent')


if __name__ == '__main__':
    unittest.main()
