"""Resolve DSV4 PCP from CLI defaults through validation, readback and publish.

Only config.json is needed: no model weights, CUDA kernels or process groups.
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import cp_decode_attn_tp as decode_tp_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
    override_platform,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci

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


class TestDSV4PCPResolution(unittest.TestCase):
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

    def make_cli_args(self, *, size, decode_tp, eager, separate_draft):
        pdmux_path = Path(self.model_path, "pdmux.yaml")
        pdmux_path.write_text(
            "sm_group_num: 3\n"
            "split_forward_token_budget: 2048\n"
            "overlap_decode_full_sm: true\n"
            "manual_divisions:\n"
            "  - [8, 0, 1]\n"
        )
        target_path = Path(self.model_path)
        if separate_draft:
            target_path = target_path / "standalone_target"
            target_path.mkdir(exist_ok=True)
            (target_path / "config.json").write_text(
                json.dumps(
                    {
                        key: value
                        for key, value in _CONFIG.items()
                        if not key.startswith("dspark_")
                    }
                )
            )
        argv = [
            "--model-path",
            str(target_path),
            "--device",
            "cuda",
            "--tp",
            str(size),
            "--dp-size",
            "1",
            "--ep-size",
            "1",
            "--moe-dp-size",
            "1",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-pdmux",
            "--pdmux-prefill-mode",
            "layer_split",
            "--pdmux-config-path",
            str(pdmux_path),
            "--sm-group-num",
            "3",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "4",
            "--moe-a2a-backend",
            "none",
            "--moe-runner-backend",
            "triton",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-write-policy",
            "write_through",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--chunked-prefill-size",
            "1024",
            "--max-running-requests",
            "8",
            "--mem-fraction-static",
            "0.7",
            "--random-seed",
            "42",
        ]
        if decode_tp:
            argv.append("--enable-cp-decode-attn-tp")
        if eager:
            argv.append("--disable-cuda-graph")
        if separate_draft:
            draft_path = Path(self.model_path, "draft")
            draft_path.mkdir(exist_ok=True)
            draft_config = {
                **_CONFIG,
                "architectures": ["DeepseekV4ForCausalLMDSpark"],
            }
            (draft_path / "config.json").write_text(json.dumps(draft_config))
            argv.extend(["--speculative-draft-model-path", str(draft_path)])
        # Keep the CLI's parser/conversion path without replacing the runner's
        # logging handlers on every matrix entry.
        with patch("sglang.srt.server_args.logging.basicConfig"):
            return prepare_server_args(argv)

    def test_cli_chain_has_no_late_overrides_and_publishes_to_runtime_namespaces(self):
        for size in (2, 4, 8):
            for decode_tp in (False, True):
                for eager in (False, True):
                    for separate_draft in (False, True):
                        with self.subTest(
                            size=size,
                            decode_tp=decode_tp,
                            eager=eager,
                            separate_draft=separate_draft,
                        ):
                            args = self.make_cli_args(
                                size=size,
                                decode_tp=decode_tp,
                                eager=eager,
                                separate_draft=separate_draft,
                            )
                            args.resolve_once()
                            args.check_server_args()
                            expected = {
                                "enable_prefill_cp": True,
                                "enable_dp_attention": True,
                                "enable_cp_decode_attn_tp": decode_tp,
                                "cp_strategy": "interleave",
                                "tp_size": size,
                                "attn_cp_size": size,
                                "dp_size": 1,
                                "pp_size": 1,
                                "ep_size": 1,
                                "moe_dp_size": 1,
                                "moe_dense_tp_size": 1,
                                "dcp_size": 1,
                                "moe_a2a_backend": "none",
                                "enable_pdmux": True,
                                "pdmux_prefill_mode": "layer_split",
                                "sm_group_num": 3,
                                "disable_overlap_schedule": False,
                                "speculative_algorithm": "DSPARK",
                                "speculative_num_draft_tokens": 5,
                                "enable_hierarchical_cache": True,
                                "hicache_ratio": 2.0,
                                "hicache_write_policy": "write_through",
                                "chunked_prefill_size": 1024,
                                "max_running_requests": 8,
                                "mem_fraction_static": 0.7,
                                "enable_two_batch_overlap": False,
                                "enable_torch_compile": False,
                            }
                            graph_backend = "disabled" if eager else "full"
                            # Inspect every declaration, not just the last one:
                            # an override that is later repaired must fail too.
                            for source, fields in args._resolved_overrides:
                                for name in expected.keys() & fields.keys():
                                    self.assertEqual(
                                        fields[name], expected[name], (source, name)
                                    )
                                if "cuda_graph_config" in fields:
                                    graph = fields["cuda_graph_config"]
                                    self.assertEqual(
                                        graph.prefill.backend, "disabled", source
                                    )
                                    self.assertEqual(
                                        graph.decode.backend, graph_backend, source
                                    )
                            info = args.resolved_dict()
                            self.assertEqual(
                                {key: info[key] for key in expected}, expected
                            )
                            self.assertEqual(
                                info["cuda_graph_config"]["decode"]["backend"],
                                graph_backend,
                            )
                            self.assertEqual(
                                info["cuda_graph_config"]["prefill"]["backend"],
                                "disabled",
                            )
                            # Re-entry cannot halve chunk sizes or lose a flag.
                            args.resolve_once()
                            self.assertEqual(args.resolved_dict(), info)
                            for role in ("tokenizer", "scheduler"):
                                child = pickle.loads(pickle.dumps(args))
                                publish(child, role=role)
                                self.assertEqual(
                                    get_context().resolved_server_args_dict(), info
                                )
                                parallel = get_parallel()
                                self.assertTrue(parallel.enable_prefill_cp)
                                self.assertTrue(parallel.enable_dp_attention)
                                self.assertEqual(
                                    parallel.enable_cp_decode_attn_tp, decode_tp
                                )
                                self.assertEqual(parallel.attn_cp_size, size)
                                self.assertEqual(parallel.attn_tp_size, 1)
                                self.assertTrue(get_disagg().enable_pdmux)
                                self.assertEqual(
                                    get_disagg().pdmux_prefill_mode, "layer_split"
                                )
                                self.assertFalse(
                                    get_schedule().disable_overlap_schedule
                                )
                                self.assertTrue(get_memory().enable_hierarchical_cache)
                                self.assertEqual(get_memory().hicache_ratio, 2.0)
                                self.assertEqual(
                                    get_memory().hicache_write_policy, "write_through"
                                )
                                self.assertEqual(
                                    get_spec().speculative_algorithm, "DSPARK"
                                )
                                graph = get_exec().graph.cuda_graph_config
                                self.assertEqual(graph.decode.backend, graph_backend)
                                self.assertEqual(graph.prefill.backend, "disabled")
                                reset_context()

    def test_published_flag_enables_real_decode_context_on_every_cp_rank(self):
        for size in (2, 4, 8):
            for enabled in (False, True):
                args = self.make_args(tp_size=size, enable_cp_decode_attn_tp=enabled)
                args.resolve_once()
                args.check_server_args()
                publish(pickle.loads(pickle.dumps(args)), role="scheduler")
                for rank in range(size):
                    with (
                        self.subTest(size=size, enabled=enabled, rank=rank),
                        get_parallel().override(attn_cp_rank=rank),
                        patch.object(
                            decode_tp_module, "_global_cp_decode_attn_tp_ctx", None
                        ),
                    ):
                        # Only the rank is supplied; enable/size come from the
                        # real published configuration, not a fixture override.
                        ctx = decode_tp_module.get_cp_decode_attn_tp_ctx()
                        self.assertEqual(ctx.is_enabled, enabled)
                        self.assertEqual(ctx.decode_tp_size, size if enabled else None)
                        self.assertEqual(ctx.decode_tp_rank, rank if enabled else None)
                        self.assertIs(ctx, decode_tp_module.get_cp_decode_attn_tp_ctx())
                        for mode in (
                            ForwardMode.DECODE,
                            ForwardMode.TARGET_VERIFY,
                            ForwardMode.DRAFT_EXTEND_V2,
                            ForwardMode.EXTEND,
                            ForwardMode.SPLIT_PREFILL,
                        ):
                            batch = SimpleNamespace(
                                forward_mode=mode,
                                input_ids=list(range(16)),
                                extend_seq_lens_cpu=[16],
                                attn_cp_metadata=None,
                            )
                            expected = enabled and mode not in (
                                ForwardMode.EXTEND,
                                ForwardMode.SPLIT_PREFILL,
                            )
                            self.assertEqual(
                                ctx.should_use_decode_attn_tp(batch), expected, mode
                            )
                        # Selection stays pure under PDMux; activation is held
                        # by the selected module, not this legacy global flag.
                        self.assertFalse(ctx.use_decode_attn_tp)
                reset_context()

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
