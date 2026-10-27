"""Resolve DSV4 PCP from CLI defaults through validation, readback and publish.

Only config.json is needed: no model weights, CUDA kernels or process groups.
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.runtime_context import (
    get_parallel,
    override_platform,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "intermediate_size": 2048,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "num_hidden_layers": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "max_position_embeddings": 8192,
    "vocab_size": 128,
    "q_lora_rank": 1024,
    "kv_lora_rank": 512,
    "head_dim": 512,
    "qk_nope_head_dim": 448,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "sliding_window": 128,
    "compress_ratios": [0, 0],
    "n_hash_layers": 0,
    "torch_dtype": "bfloat16",
    "rope_scaling": {},
    "dspark_block_size": 4,
    "dspark_markov_rank": 8,
    "dspark_noise_token_id": 127,
    "dspark_target_layer_ids": [0, 1],
}


class TestDSV4PCPResolution(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        self.enterContext(
            patch.dict(
                os.environ, SGLANG_RAGGED_VERIFY_MODE="static", HF_HUB_OFFLINE="1"
            )
        )
        self.enterContext(override_platform(is_hip=False, is_npu=False, is_musa=False))
        self.enterContext(patch.object(cp_base, "_STRATEGY", cp_base._STRATEGY))
        self.model_path = self.enterContext(tempfile.TemporaryDirectory())
        Path(self.model_path, "config.json").write_text(json.dumps(_CONFIG))

    def make_args(self, **overrides):
        options = dict(
            model_path=self.model_path,
            device="cuda",
            random_seed=42,
            tp_size=8,
            dp_size=1,
            ep_size=1,
            moe_dp_size=1,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            enable_cp_decode_attn_tp=True,
            enable_pdmux=True,
            pdmux_prefill_mode="layer_split",
            speculative_algorithm="DSPARK",
            speculative_dspark_block_size=4,
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            moe_a2a_backend="none",
            moe_runner_backend="triton",
            cuda_graph_backend_prefill="disabled",
            chunked_prefill_size=1024,
        )
        options.update(overrides)
        return ServerArgs(**options)

    def test_cli_default_resolves_through_validation_readback_and_child_publish(self):
        for size in (2, 4, 8):
            for decode_tp in (False, True):
                with self.subTest(size=size, decode_tp=decode_tp):
                    args = self.make_args(
                        tp_size=size, enable_cp_decode_attn_tp=decode_tp
                    )
                    # Exercise the actual ordering, starting with the CLI's
                    # dp-attention=False and attn-cp-size=1, not a resolved mock.
                    self.assertFalse(args.enable_dp_attention)
                    self.assertEqual(args.attn_cp_size, 1)
                    args.resolve_once()
                    args.check_server_args()
                    info = args.resolved_dict()  # /server_info's config source.
                    self.assertTrue(info["enable_dp_attention"])
                    self.assertEqual(info["attn_cp_size"], size)
                    self.assertEqual(info["moe_dense_tp_size"], 1)
                    self.assertFalse(info["enable_dp_lm_head"])
                    self.assertFalse(info["disable_overlap_schedule"])
                    self.assertTrue(info["enable_hierarchical_cache"])
                    self.assertEqual(info["enable_cp_decode_attn_tp"], decode_tp)
                    # The record remains raw; workers must publish the carried
                    # decisions rather than deriving from raw fields again.
                    self.assertFalse(args.enable_dp_attention)
                    child = pickle.loads(pickle.dumps(args))
                    publish(child, role="scheduler")
                    parallel = get_parallel()
                    self.assertTrue(parallel.enable_dp_attention)
                    self.assertEqual(parallel.attn_cp_size, size)
                    self.assertEqual(parallel.attn_dp_size, 1)
                    self.assertEqual(parallel.attn_tp_size, 1)
                    self.assertEqual(child.resolved_dict(), info)
                    reset_context()

    def test_cp_declaration_also_survives_without_pdmux(self):
        args = self.make_args(enable_pdmux=False)
        args.resolve_once()
        args.check_server_args()
        self.assertTrue(args.resolved_dict()["enable_dp_attention"])

    def test_plain_tp_dp1_still_disables_dp_attention_and_dp_lm_head(self):
        for requested in (False, True):
            with self.subTest(requested=requested):
                args = self.make_args(
                    enable_prefill_cp=False,
                    cp_strategy=None,
                    enable_cp_decode_attn_tp=False,
                    enable_dp_attention=requested,
                    enable_dp_lm_head=requested,
                )
                args.resolve_once()
                args.check_server_args()
                info = args.resolved_dict()
                self.assertFalse(info["enable_dp_attention"])
                self.assertFalse(info["enable_dp_lm_head"])

    def test_admission_still_rejects_lost_cp_dp_attention_resolution(self):
        args = self.make_args()
        args.resolve_once()
        declare_resolution(args, "test_lost_cp_decision", enable_dp_attention=False)
        with self.assertRaisesRegex(AssertionError, "prefill CP requires"):
            args.check_server_args()


if __name__ == "__main__":
    unittest.main()
