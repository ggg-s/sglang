"""DSV4 DSpark + PCP + CPU overlap + PDMux + HiCache; needs two CUDA GPUs.

Uses the same target/draft environment variables as the plain-TP2 test.
Checks host reload output parity and HiCache progress during concurrent prefill.
"""

from __future__ import annotations

import requests
from test_dsv4_dspark_pdmux_hicache_tp2 import (
    TestDSV4DSparkTP2CPUOverlapHiCache as _TP2HiCache,
)


class TestDSV4DSparkPCPTP2CPUOverlapHiCache(_TP2HiCache):
    cp_decode_attn_tp = False
    server_env = {**_TP2HiCache.server_env, "SGLANG_RAGGED_VERIFY_MODE": "static"}
    extra_server_args = [
        *_TP2HiCache.extra_server_args,
        "--enable-prefill-cp",
        "--cp-strategy",
        "interleave",
        "--dp-size",
        "1",
        "--moe-a2a-backend",
        "none",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            info = requests.get(cls.base_url + "/get_server_info", timeout=120).json()
            assert info["enable_prefill_cp"], info
            assert info["enable_dp_attention"], info
            assert info["attn_cp_size"] == info["tp_size"] == 2, info
            assert info["dp_size"] == 1, info
            assert info["cp_strategy"] == "interleave", info
            assert info["moe_a2a_backend"] == "none", info
            assert info["enable_cp_decode_attn_tp"] == cls.cp_decode_attn_tp, info
        except Exception:
            cls.tearDownClass()
            raise


class TestDSV4DSparkPCPTP2EagerCPUOverlapHiCache(TestDSV4DSparkPCPTP2CPUOverlapHiCache):
    """Also exercise eager target/draft execution and the unfurled logits head."""

    extra_server_args = [
        *TestDSV4DSparkPCPTP2CPUOverlapHiCache.extra_server_args,
        "--disable-cuda-graph",
    ]
    server_env = {
        **TestDSV4DSparkPCPTP2CPUOverlapHiCache.server_env,
        "SGLANG_DSPARK_FOLDED_PROPOSAL": "0",
    }


class TestDSV4DSparkPCPDecodeTP2CPUOverlapHiCache(
    TestDSV4DSparkPCPTP2CPUOverlapHiCache
):
    cp_decode_attn_tp = True
    extra_server_args = [
        *TestDSV4DSparkPCPTP2CPUOverlapHiCache.extra_server_args,
        "--enable-cp-decode-attn-tp",
    ]


class TestDSV4DSparkPCPDecodeTP2EagerCPUOverlapHiCache(
    TestDSV4DSparkPCPTP2EagerCPUOverlapHiCache
):
    cp_decode_attn_tp = True
    extra_server_args = [
        *TestDSV4DSparkPCPTP2EagerCPUOverlapHiCache.extra_server_args,
        "--enable-cp-decode-attn-tp",
    ]
