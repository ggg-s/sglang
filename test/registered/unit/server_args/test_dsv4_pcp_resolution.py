"""Resolve DSV4 PCP through publish and runtime CUDA graph output planning.

Only config.json is needed: no model weights or process groups. CUDA output
allocation/replay is also checked when a GPU is available.
"""

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import cp_decode_attn_tp as decode_tp_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_exec,
    get_flags,
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

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

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

    def make_cli_args(
        self, *, size, decode_tp, eager, separate_draft, prefill_mode="layer_split"
    ):
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
            prefill_mode,
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

    def test_runtime_gather_predicates_use_attention_tp_not_cp_width(self):
        from sglang.srt.layers.moe.utils import MoeA2ABackend
        from sglang.srt.utils.common import (
            get_cuda_graph_batch_size_alignment,
            require_attn_tp_gather,
            require_gathered_buffer,
            require_mlp_sync,
            require_mlp_tp_gather,
        )

        args = self.make_args()
        args.resolve_once()
        publish(args, role="scheduler")
        # DP1 is not a blanket attention-gather exemption: ordinary TP with
        # dense-TP/A2A sequence scattering must retain its existing behavior.
        # dp_attention, dp, cp, dense_tp, a2a, mlp_gather, attn_gather, alignment
        cases = (
            (False, 1, 1, None, "none", False, False, 1),
            (False, 1, 1, 1, "none", False, True, 8),
            (False, 1, 1, None, "deepep", False, True, 8),
            (True, 1, 1, 1, "none", False, True, 8),
            (True, 1, 8, 1, "none", False, False, 8),
            (True, 1, 8, None, "deepep", False, False, 8),
            (True, 1, 4, 1, "none", False, True, 8),
            (True, 2, 1, 1, "none", True, True, 4),
            (True, 8, 1, 1, "none", True, False, 1),
            (True, 2, 4, 1, "none", True, False, 4),
        )
        for enabled, dp, cp, dense, a2a, mlp, attn, alignment in cases:
            topology = dict(
                enable_dp_attention=enabled,
                dp_size=dp,
                tp_size=8,
                attn_cp_size=cp,
                moe_dense_tp_size=dense,
                enable_dp_lm_head=False,
                disable_attn_tp_gather=False,
                enable_two_batch_overlap=False,
            )
            with (
                self.subTest(enabled=enabled, dp=dp, cp=cp, dense=dense, a2a=a2a),
                get_context().override_server_args(**topology),
                get_flags().moe.override(a2a_backend=MoeA2ABackend(a2a)),
            ):
                self.assertEqual(require_mlp_tp_gather(), mlp)
                self.assertEqual(require_attn_tp_gather(), attn)
                self.assertEqual(require_gathered_buffer(), mlp or attn)
                # PCP still needs scheduler metadata and CP-aligned buffers.
                self.assertEqual(require_mlp_sync(), enabled or mlp or attn)
                self.assertEqual(get_cuda_graph_batch_size_alignment(), alignment)
                with get_context().override_server_args(
                    **{**topology, "disable_attn_tp_gather": True}
                ):
                    self.assertFalse(require_attn_tp_gather())
                    self.assertEqual(require_gathered_buffer(), mlp)

    def _check_runtime_graph_output_planning(self, device, prefill_mode="layer_split"):
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
            get_batch_sizes_to_capture,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.utils.common import (
            get_cuda_graph_batch_size_alignment,
            require_gathered_buffer,
            require_mlp_sync,
        )

        for size in (2, 4, 8):
            for decode_tp in (False, True):
                args = self.make_args(
                    tp_size=size,
                    enable_cp_decode_attn_tp=decode_tp,
                    max_running_requests=16,
                    cuda_graph_bs_decode=[1, 2, 4, 8, 12, 16],
                    pdmux_prefill_mode=prefill_mode,
                )
                args.resolve_once()
                args.check_server_args()
                publish(pickle.loads(pickle.dumps(args)), role="scheduler")
                self.assertFalse(require_gathered_buffer())
                self.assertTrue(require_mlp_sync())
                self.assertEqual(get_parallel().attn_tp_size, 1)
                self.assertEqual(get_cuda_graph_batch_size_alignment(), size)
                for draft in (False, True):
                    with (
                        self.subTest(size=size, decode_tp=decode_tp, draft=draft),
                        patch.object(GraphSharedOutput, "_process_shared", None),
                    ):
                        # Skip weights and CUDA backend construction, but execute
                        # the actual failing startup chain, including ModelRunner
                        # logits-row sizing and graph bucket filtering.
                        runner = ModelRunner.__new__(ModelRunner)
                        runner.device = device
                        runner.is_draft_worker = draft
                        runner.spec_algorithm = SpeculativeAlgorithm.DSPARK
                        runner.model_config = SimpleNamespace(
                            hf_config=SimpleNamespace(**_CONFIG)
                        )
                        runner.req_to_token_pool = SimpleNamespace(size=13)
                        width = runner.decode_num_tokens_per_req()
                        self.assertEqual(width, 4 if draft else 5)
                        capture_bs, compile_bs = get_batch_sizes_to_capture(
                            runner, width
                        )
                        expected_bs = {
                            (2, False): [2, 4, 8, 12, 14],
                            (2, True): [1, 2, 4, 8, 12, 14],
                            (4, False): [4, 8, 12, 16],
                            (4, True): [1, 2, 4, 8, 12, 16],
                            (8, False): [8, 16],
                            (8, True): [2, 4, 8, 12, 16],
                        }[size, draft]
                        self.assertEqual(capture_bs, expected_bs)
                        self.assertEqual(compile_bs, [])
                        shared = GraphSharedOutput.create_for_model_runner(runner)
                        self.assertIsNotNone(shared)
                        self.assertEqual(shared.max_rows, expected_bs[-1] * width)
                        self.assertIs(
                            GraphSharedOutput.create_for_model_runner(runner), shared
                        )
                        logits = shared.get_logits_buffer(7, rows=width)
                        self.assertEqual(tuple(logits.shape), (width, 7))
                        if device == "cuda":
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                logits.add_(1)
                            torch.cuda.synchronize()
                            logits.zero_()
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(logits, torch.ones_like(logits))
                reset_context()

    def test_resolved_pcp_reaches_real_graph_shared_output_planning(self):
        self._check_runtime_graph_output_planning("cpu")

    def test_standard_pcp_reaches_real_graph_shared_output_planning(self):
        self._check_runtime_graph_output_planning("cpu", "standard")

    def test_standard_published_flag_selects_real_target_and_draft_views(self):
        # Reuse the small numerical fixture, but replace its context stand-in
        # with the real singleton built from the published CLI configuration.
        fixture_dir = Path(__file__).resolve().parents[1] / "multiplex"
        with patch.object(sys, "path", [str(fixture_dir), *sys.path]):
            from test_pdmux_cp_decode_attn_tp import fixture

        x = torch.arange(32, dtype=torch.float64).reshape(4, 8) / 100
        for size in (2, 4, 8):
            for enabled in (False, True):
                args = self.make_cli_args(
                    size=size,
                    decode_tp=enabled,
                    eager=False,
                    separate_draft=False,
                    prefill_mode="standard",
                )
                args.resolve_once()
                args.check_server_args()
                publish(pickle.loads(pickle.dumps(args)), role="scheduler")
                for draft in (False, True):
                    partitions = []
                    for rank in range(size):
                        with (
                            self.subTest(
                                size=size, enabled=enabled, draft=draft, rank=rank
                            ),
                            get_parallel().override(attn_cp_rank=rank),
                            patch.object(
                                decode_tp_module, "_global_cp_decode_attn_tp_ctx", None
                            ),
                        ):
                            torch.manual_seed(7)
                            model, _, linear = fixture(
                                size=size, rank=rank, draft=draft
                            )
                            # Eight groups permit a nonempty CP8 partition.
                            model.n_groups = model.n_local_groups = 8
                            for name, rows, cols in (
                                ("wq_b", 32, 8),
                                ("wo_a", 32, 4),
                                ("wo_b", 8, 32),
                            ):
                                layer = getattr(model, name)
                                layer.weight = torch.nn.Parameter(
                                    torch.randn(rows, cols, dtype=torch.float64),
                                    requires_grad=False,
                                )
                                setattr(
                                    layer,
                                    "input_size_per_partition"
                                    if name == "wo_b"
                                    else "output_size_per_partition",
                                    cols if name == "wo_b" else rows,
                                )
                            ctx = decode_tp_module.get_cp_decode_attn_tp_ctx()
                            scope_globals = (
                                model.maybe_use_decode_attn_tp.__wrapped__.__globals__
                            )
                            scope_globals["get_cp_decode_attn_tp_ctx"] = (
                                lambda ctx=ctx: ctx
                            )
                            scope_globals["is_pdmux_enabled"] = lambda: (
                                get_disagg().enable_pdmux
                            )
                            reference = model(x)
                            with patch.dict(sys.modules, {linear.__name__: linear}):
                                for mode in (
                                    ForwardMode.DECODE,
                                    ForwardMode.TARGET_VERIFY,
                                    ForwardMode.DRAFT_EXTEND_V2,
                                    ForwardMode.EXTEND,
                                ):
                                    batch = SimpleNamespace(
                                        forward_mode=mode,
                                        input_ids=list(range(16)),
                                        extend_seq_lens_cpu=[16],
                                        attn_cp_metadata=None,
                                    )
                                    sharded = enabled and mode != ForwardMode.EXTEND
                                    with model.maybe_use_decode_attn_tp(
                                        batch
                                    ) as selected:
                                        self.assertEqual(selected is not model, sharded)
                                        self.assertEqual(
                                            selected.attn_tp_size,
                                            size if sharded else 1,
                                        )
                                        self.assertEqual(
                                            selected.n_local_heads,
                                            8 // size if sharded else 8,
                                        )
                                        value = selected(x)
                                        if mode == ForwardMode.DECODE:
                                            partitions.append(value)
                                        if not sharded:
                                            torch.testing.assert_close(value, reference)
                            self.assertEqual(model.wq_b.weight.shape, (32, 8))
                            self.assertEqual(model.n_local_heads, 8)
                            self.assertFalse(ctx.use_decode_attn_tp)
                    if enabled:
                        torch.testing.assert_close(sum(partitions), reference)
                reset_context()

    def test_standard_cli_publishes_full_chain_with_no_fallback(self):
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
                                prefill_mode="standard",
                            )
                            args.resolve_once()
                            args.check_server_args()
                            publish(pickle.loads(pickle.dumps(args)), role="scheduler")
                            info = get_context().resolved_server_args_dict()
                            self.assertEqual(info["pdmux_prefill_mode"], "standard")
                            self.assertFalse(info["disable_overlap_schedule"])
                            self.assertTrue(info["enable_dp_attention"])
                            self.assertTrue(info["enable_hierarchical_cache"])
                            self.assertTrue(info["enable_prefill_cp"])
                            self.assertEqual(info["attn_cp_size"], size)
                            self.assertEqual(
                                info["enable_cp_decode_attn_tp"], decode_tp
                            )
                            self.assertEqual(info["speculative_algorithm"], "DSPARK")
                            self.assertEqual(get_parallel().attn_tp_size, 1)
                            self.assertEqual(
                                get_exec().graph.cuda_graph_config.decode.backend,
                                "disabled" if eager else "full",
                            )
                            self.assertEqual(
                                get_exec().graph.cuda_graph_config.prefill.backend,
                                "disabled",
                            )
                            reset_context()

    def test_standard_pcp_retains_parallelism_and_backend_guards(self):
        from sglang.srt.model_executor.cuda_graph_config import with_phase

        for fields in (
            dict(enable_dp_attention=False),
            dict(dp_size=2),
            dict(attn_cp_size=4),
            dict(ep_size=2),
            dict(moe_dp_size=2),
            dict(dcp_size=2),
            dict(cp_strategy="zigzag"),
            dict(moe_a2a_backend="deepep"),
            dict(enable_torch_compile=True),
            dict(enable_two_batch_overlap=True),
            dict(enable_unified_memory=True),
        ):
            with self.subTest(fields=fields):
                args = self.make_args(pdmux_prefill_mode="standard")
                args.resolve_once()
                declare_resolution(args, "invalid_standard_pcp", **fields)
                with self.assertRaises(AssertionError):
                    args.check_server_args()
        args = self.make_args(pdmux_prefill_mode="standard")
        args.resolve_once()
        declare_resolution(
            args,
            "invalid_standard_prefill_graph",
            cuda_graph_config=with_phase(
                resolving_view(args).cuda_graph_config, "prefill", backend="full"
            ),
        )
        with self.assertRaises(AssertionError):
            args.check_server_args()

    def test_pcp_dp1_keeps_local_scheduler_metadata_on_every_rank(self):
        from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
        from sglang.srt.managers.scheduler_components.dp_attn import (
            MLPSyncBatchInfo,
            _update_gather_batch,
            should_skip_scheduler_all_gather,
        )
        from sglang.srt.utils.common import require_mlp_tp_gather

        for size in (2, 4, 8):
            args = self.make_args(tp_size=size)
            args.resolve_once()
            publish(args, role="scheduler")
            parallel = get_parallel()
            for rank in range(size):
                with self.subTest(size=size, rank=rank):
                    self.assertEqual(
                        compute_dp_attention_world_info(
                            parallel.enable_dp_attention,
                            rank,
                            parallel.tp_size,
                            parallel.dp_size,
                            parallel.attn_cp_size,
                        ),
                        (0, 1, 0, 1),
                    )
            self.assertTrue(should_skip_scheduler_all_gather(parallel.dp_size))
            for mode, tokens in (
                (ForwardMode.DECODE, 3),
                (ForwardMode.SPLIT_PREFILL, 16),
            ):
                with self.subTest(size=size, mode=mode):
                    info = MLPSyncBatchInfo(
                        dp_size=1,
                        tp_size=parallel.attn_tp_size,
                        cp_size=size,
                        num_tokens=tokens,
                        num_tokens_for_logprob=3,
                        can_run_decode_cuda_graph=mode.is_decode(),
                        can_run_prefill_cuda_graph=False,
                        is_extend_in_batch=mode.is_extend(),
                        local_can_run_tbo=False,
                        local_forward_mode=mode.value,
                    )
                    info.finalize_local()
                    batch = SimpleNamespace()
                    _update_gather_batch(batch, info, require_mlp_tp_gather())
                    self.assertEqual(batch.global_num_tokens, [tokens])
                    self.assertEqual(batch.global_num_tokens_for_logprob, [3])
                    self.assertEqual(batch.can_run_decode_cuda_graph, mode.is_decode())
                    self.assertEqual(batch.is_extend_in_batch, mode.is_extend())
            reset_context()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for graph replay")
    def test_resolved_pcp_graph_output_allocation_and_replay_on_cuda(self):
        # CP sizes here are logical topology settings; this checks graph
        # planning/storage/replay, not multi-GPU model attention or collectives.
        self._check_runtime_graph_output_planning("cuda")
        self._check_runtime_graph_output_planning("cuda", "standard")

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
