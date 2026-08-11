#!/usr/bin/env python3
"""health_check 实时源探活 + feishu_push 重试测试（方案 B 稳定性）"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# feishu_push 模块级要求 FEISHU_APP_SECRET → 先设再 import
os.environ.setdefault("FEISHU_APP_SECRET", "test-secret-for-unit")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "daily_pipeline"))

import health_check as hc
import feishu_push as fp


class TestRealtimeHealthCheck(unittest.TestCase):
    """实时源探活"""

    def test_tencent_ok(self):
        fake = type('R', (), {
            'encoding': 'gbk', 'text': 'v_sh588000="1~科创50~588000~1.835~1.840~1.840~34364399~'
                                       '0~0~1.834~1~1.833~1~1.832~1~1.831~1~1.830~1~1.835~1~'
                                       '1.836~1~1.837~1~1.838~1~1.839~1~~20260811103000~-0.005~'
                                       '-0.27~1.851~1.799~1.835/34364399/6255060479~34364399~'
                                       '625506~6.93~"',
        })()
        with mock.patch('requests.get', return_value=fake):
            r = hc._check_realtime_tencent('588000')
        self.assertTrue(r['ok'])
        self.assertIn('qt.gtimg.cn', r['msg'])

    def test_tencent_bad_format(self):
        fake = type('R', (), {'encoding': 'gbk', 'text': 'no quote here'})()
        with mock.patch('requests.get', return_value=fake):
            r = hc._check_realtime_tencent('588000')
        self.assertFalse(r['ok'])

    def test_tencent_network_error(self):
        with mock.patch('requests.get', side_effect=Exception('conn refused')):
            r = hc._check_realtime_tencent('588000')
        self.assertFalse(r['ok'])

    def test_sina_ok(self):
        fake = type('R', (), {
            'encoding': 'gbk',
            'text': 'var hq_str_sh588000="科创50ETF华夏,1.840,1.840,1.835,1.851,1.799,'
                    '1.834,1.835,3436439900,6255060479,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
                    '2026-08-11,10:30:00,00,";',
        })()
        with mock.patch('requests.get', return_value=fake) as m:
            r = hc._check_realtime_sina('588000')
        self.assertTrue(r['ok'])
        # 必须带 Referer
        headers = m.call_args[1]['headers']
        self.assertIn('Referer', headers)

    def test_sina_network_error(self):
        with mock.patch('requests.get', side_effect=Exception('timeout')):
            r = hc._check_realtime_sina('588000')
        self.assertFalse(r['ok'])

    def test_combined_health(self):
        """check_realtime_health 汇总：双源都 OK → ok"""
        with mock.patch.object(hc, '_check_realtime_tencent',
                               return_value={'ok': True, 'msg': 'ok'}):
            with mock.patch.object(hc, '_check_realtime_sina',
                                   return_value={'ok': True, 'msg': 'ok'}):
                r = hc.check_realtime_health()
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['sources']), 2)

    def test_combined_one_fail(self):
        with mock.patch.object(hc, '_check_realtime_tencent',
                               return_value={'ok': False, 'msg': 'fail'}):
            with mock.patch.object(hc, '_check_realtime_sina',
                                   return_value={'ok': True, 'msg': 'ok'}):
                r = hc.check_realtime_health()
        self.assertFalse(r['ok'])
        self.assertIn('14:50', r['msg'])  # 提示将走兜底


class TestFeishuPushRetry(unittest.TestCase):
    """_post_message 重试：网络层异常重试，业务错误不重试"""

    def test_success_first_try(self):
        fake = type('R', (), {'json': lambda self: {'code': 0}})()
        with mock.patch('requests.post', return_value=fake) as m:
            fp._post_message({}, {'x': 1})
        self.assertEqual(m.call_count, 1)

    def test_network_retry_then_success(self):
        fake = type('R', (), {'json': lambda self: {'code': 0}})()
        # OSError 模拟真实网络故障（连接重置/超时类）
        with mock.patch('requests.post', side_effect=[OSError('conn reset'), fake]) as m:
            with mock.patch.object(fp.time, 'sleep'):
                fp._post_message({}, {'x': 1})
        self.assertEqual(m.call_count, 2)

    def test_business_error_no_retry(self):
        fake = type('R', (), {'json': lambda self: {'code': 9499, 'msg': 'bad'}})()
        with mock.patch('requests.post', return_value=fake) as m:
            with self.assertRaises(RuntimeError):
                fp._post_message({}, {'x': 1})
        self.assertEqual(m.call_count, 1)  # 业务错误不重试

    def test_all_retries_exhausted(self):
        with mock.patch('requests.post', side_effect=OSError('always down')) as m:
            with mock.patch.object(fp.time, 'sleep'):
                with self.assertRaises(OSError):
                    fp._post_message({}, {'x': 1}, retries=2)
        self.assertEqual(m.call_count, 3)  # 1 + 2 重试


if __name__ == '__main__':
    unittest.main()
