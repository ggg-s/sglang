"""Full-model TP2 acceptance test; requires two GPUs and local DSV4 weights.

SGLANG_TEST_DSV4_FLASH_MODEL_PATH=/models/DeepSeek-V4 \
SGLANG_TEST_DSV4_DSPARK_MODEL_PATH=/models/DeepSeek-V4-DSpark \
    python -m pytest test/manual/pdmux/test_dsv4_dspark_pdmux_hicache_tp2.py -v

The draft path may be omitted when the target checkpoint bundles DSpark.
"""

from __future__ import annotations

import os
import unittest

import requests
import torch
from test_dsv4_pdmux_hicache_tp8 import DSV4PDMuxHiCacheMixin
from test_pdmux_hicache import sm_multiple

from sglang.test.test_utils import CustomTestCase

DRAFT_PATH = os.environ.get("SGLANG_TEST_DSV4_DSPARK_MODEL_PATH")


@unittest.skipUnless(
    torch.cuda.device_count() >= 2
    and os.environ.get("SGLANG_TEST_DSV4_FLASH_MODEL_PATH"),
    "Two CUDA GPUs and SGLANG_TEST_DSV4_FLASH_MODEL_PATH are required",
)
class TestDSV4DSparkTP2CPUOverlapHiCache(DSV4PDMuxHiCacheMixin, CustomTestCase):
    disable_overlap_schedule = False
    sm_group_num = 3
    extra_server_args = [
        "--trust-remote-code",
        "--tp",
        "2",
        "--ep-size",
        "1",
        "--speculative-algorithm",
        "DSPARK",
        "--chunked-prefill-size",
        "1024",
        "--max-running-requests",
        "8",
        *(["--speculative-draft-model-path", DRAFT_PATH] if DRAFT_PATH else []),
    ]

    @classmethod
    def pdmux_config_body(cls):
        total_sm = torch.cuda.get_device_properties(0).multi_processor_count
        return (
            "sm_group_num: 3\n"
            "split_forward_token_budget: 2048\n"
            "overlap_decode_full_sm: true\n"
            "manual_divisions:\n"
            f"  - [{sm_multiple(total_sm // 2)}, 0, 1]\n"
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # setUpClass failures do not call tearDownClass.
        try:
            info = requests.get(cls.base_url + "/get_server_info", timeout=120).json()
            assert info["speculative_algorithm"] == "DSPARK", info
            assert info["tp_size"] == 2, info
            assert info["enable_hierarchical_cache"], info
        except Exception:
            cls.tearDownClass()
            raise


if __name__ == "__main__":
    unittest.main()
