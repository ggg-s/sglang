"""PCP admission and concurrent lane state; runnable without the GPU runtime."""

from __future__ import annotations

import ast
import contextlib
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_pdmux_sxf_port import SRT, load_methods, register_cpu_ci
from test_pdmux_tp_overlap import lane_namespace

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def load_definitions(path, names, **namespace):
    tree = ast.parse((SRT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return NS(**namespace)


class TestPdmuxPrefillCPAdmission(unittest.TestCase):
    def check(self, arch="DeepseekV4ForCausalLM", ragged_mode="static", **overrides):
        cfg = NS(
            pdmux_prefill_mode="layer_split",
            speculative_algorithm="DSPARK",
            tp_size=2,
            dp_size=1,
            enable_prefill_cp=True,
            enable_dp_attention=True,
            cp_strategy="interleave",
            attn_cp_size=2,
            ep_size=1,
            moe_dp_size=1,
            dcp_size=1,
            moe_a2a_backend="none",
            enable_cp_decode_attn_tp=False,
            enable_torch_compile=False,
            enable_two_batch_overlap=False,
            enable_hierarchical_cache=True,
            cuda_graph_config=NS(prefill=NS(backend="disabled")),
        )
        vars(cfg).update(overrides)
        methods = load_methods(
            "arg_groups/validation_hook.py",
            None,
            ["check_pdmux_overlap"],
            model_config_of=lambda _: NS(hf_config=NS(architectures=[arch])),
            envs=NS(SGLANG_RAGGED_VERIFY_MODE=NS(get=lambda: ragged_mode)),
        )
        methods["check_pdmux_overlap"](cfg)

    def test_full_width_cp_with_hicache_is_admitted(self):
        for mode in ("layer_split", "standard"):
            for size in (2, 4, 8):
                for decode_tp in (False, True):
                    with self.subTest(mode=mode, size=size, decode_tp=decode_tp):
                        self.check(
                            pdmux_prefill_mode=mode,
                            tp_size=size,
                            attn_cp_size=size,
                            enable_cp_decode_attn_tp=decode_tp,
                        )

    def test_other_topologies_and_shared_resources_remain_rejected(self):
        for overrides in (
            dict(arch="GlmMoeDsaForCausalLM"),
            dict(speculative_algorithm="DFLASH"),
            dict(cp_strategy="zigzag"),
            dict(dp_size=2),
            dict(attn_cp_size=1),
            dict(enable_dp_attention=False),
            dict(ep_size=2),
            dict(moe_dp_size=2),
            dict(dcp_size=2),
            dict(moe_a2a_backend="megamoe"),
            dict(moe_a2a_backend="deepep"),
            dict(enable_torch_compile=True),
            dict(enable_two_batch_overlap=True),
            dict(ragged_mode="compact"),
            dict(cuda_graph_config=NS(prefill=NS(backend="tc_piecewise"))),
            dict(cuda_graph_config=NS(prefill=NS(backend="full"))),
            dict(pdmux_prefill_mode="unknown"),
        ):
            for mode in ("layer_split", "standard"):
                with (
                    self.subTest(mode=mode, overrides=overrides),
                    self.assertRaises(AssertionError),
                ):
                    self.check(**{"pdmux_prefill_mode": mode, **overrides})


class TestPdmuxPrefillCPState(unittest.TestCase):
    def test_eager_draft_logits_gather_keeps_the_cp_vocab_shards(self):
        full, singleton = NS(world_size=2), NS(world_size=1)
        active_group = [full]

        @contextlib.contextmanager
        def draft_context(group):
            old = active_group[0]
            active_group[0] = group
            try:
                yield
            finally:
                active_group[0] = old

        class Logits:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, index):
                return self.values[index[-1]]

        parallel = NS(attn_cp_size=2, attn_tp_group=singleton)
        context = load_methods(
            "speculative/dspark_components/dspark_draft.py",
            "DraftBlockProposer",
            ["_base_logits_context"],
            get_parallel=lambda: parallel,
            draft_tp_context=draft_context,
            nullcontext=contextlib.nullcontext,
        )["_base_logits_context"]
        gather = load_methods(
            "models/dspark.py",
            None,
            ["gather_and_crop_vocab"],
            tensor_model_parallel_all_gather=lambda local, **kw: Logits(
                [10, 11, 20, 21] if active_group[0] is full else local.values
            ),
        )["gather_and_crop_vocab"]
        proposer = NS(_dp_moe_sync=True)
        with context(proposer):
            logits = gather(Logits([10, 11]), NS(org_vocab_size=4))
        self.assertEqual(logits, [10, 11, 20, 21])
        self.assertIs(active_group[0], full)

        # True DP still uses the singleton head holding its complete vocab.
        parallel.attn_cp_size = 1
        with context(proposer):
            self.assertIs(active_group[0], singleton)
        self.assertIs(active_group[0], full)

    def test_dspark_sync_remains_enabled_across_cp_ranks(self):
        full, singleton = NS(world_size=2), NS(world_size=1)
        sync = load_definitions(
            "speculative/spec_tp_sync.py",
            ["SpecTpSync"],
            envs=NS(SGLANG_SPEC_TP_SYNC=NS(get=lambda: "all")),
            parse_spec_tp_sync=lambda _: {"target", "accept"},
            _ALL={"target", "accept"},
        )
        tree = ast.parse(
            (SRT / "speculative/dspark_components/dspark_worker_v2.py").read_text()
        )
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "_tp_sync"
                for target in node.targets
            )
        )
        for cp_size, expected in ((1, False), (2, True)):
            with self.subTest(cp_size=cp_size):
                worker = NS()
                exec(
                    compile(
                        ast.Module(body=[assignment], type_ignores=[]), "sync", "exec"
                    ),
                    dict(
                        self=worker,
                        SpecTpSync=sync.SpecTpSync,
                        parallel=NS(
                            enable_dp_attention=True,
                            attn_cp_size=cp_size,
                            attn_tp_group=singleton,
                            tp_group=full,
                        ),
                    ),
                )
                self.assertEqual(worker._tp_sync.enabled("target"), expected)
                self.assertEqual(worker._tp_sync.enabled("accept"), expected)

    def test_cp_and_moe_groups_follow_prefill_on_another_thread(self):
        decode, prefill = object(), object()
        lanes = lane_namespace(decode, prefill)

        def submit_prefill():
            lanes.set_pdmux_status(True)
            return lanes.get_attn_cp_group(), lanes.get_moe_dp_group()

        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(pool.submit(submit_prefill).result(), (prefill, prefill))
        self.assertIs(lanes.get_attn_cp_group(), decode)
        self.assertIs(lanes.get_moe_dp_group(), decode)

    def test_concurrent_dp_buffers_keep_lengths_padding_and_count_tensors(self):
        prefill_group = object()
        lanes = lane_namespace(object(), prefill_group)
        allocation = Mock(side_effect=lambda shape, **kw: NS(shape=shape, **kw))
        dp = load_definitions(
            "layers/dp_attention.py",
            [
                "_DpGatheredBufferWrapper",
                "_PdmuxPrefillDpBuffer",
                "_PdmuxDecodeDpBuffer",
                "_dp_buffer_wrapper",
                "set_dp_buffer_len",
                "set_local_dp_buffer_len",
                "get_local_dp_buffer_len",
                "get_global_dp_buffer_len",
                "get_local_dp_buffer",
                "get_global_dp_buffer",
                "get_dp_global_num_tokens",
                "is_dp_max_padding",
            ],
            is_pdmux_enabled=lambda: True,
            is_pdmux_prefill_enabled=lambda: lanes.get_tp_group() is prefill_group,
            torch=NS(empty=allocation),
            use_symmetric_memory=lambda *a, **kw: contextlib.nullcontext(),
            get_flags=lambda: NS(
                dp=NS(buffer_hidden_size=8, buffer_dtype="bf16", buffer_device="cpu")
            ),
        )
        ready, decode_written = threading.Event(), threading.Event()
        p_counts, d_counts = object(), object()

        def submit_prefill():
            lanes.set_pdmux_status(True)
            dp.set_dp_buffer_len(1025, 1025, False, [1025], p_counts)
            # prepare_cp_forward rounds the MoE gather length up to 1026.
            dp.set_local_dp_buffer_len(1026)
            ready.set()
            self.assertTrue(decode_written.wait(5))
            return (
                dp.get_local_dp_buffer(None).shape,
                dp.get_global_dp_buffer_len(),
                dp.is_dp_max_padding(),
                dp.get_dp_global_num_tokens(),
                dp._dp_buffer_wrapper().get_dp_global_num_tokens_gpu(),
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            work = pool.submit(submit_prefill)
            self.assertTrue(ready.wait(5))
            dp.set_dp_buffer_len(12, 12, True, [12], d_counts)
            decode_written.set()
            self.assertEqual(work.result(), ((1026, 8), 1025, False, [1025], p_counts))
        self.assertEqual(dp.get_local_dp_buffer(None).shape, (12, 8))
        self.assertTrue(dp.is_dp_max_padding())
        self.assertIs(dp._dp_buffer_wrapper().get_dp_global_num_tokens_gpu(), d_counts)

    def test_prefill_moe_skip_reduce_does_not_leak_to_decode(self):
        prefill = object()
        lanes = lane_namespace(object(), prefill)
        runtime = load_definitions(
            "runtime_context.py",
            ["ForwardFlags", "get_forward"],
            contextmanager=contextlib.contextmanager,
            _ps=lambda: NS(
                is_pdmux_prefill_enabled=lambda: lanes.get_tp_group() is prefill
            ),
        )
        context = NS(
            forward=runtime.ForwardFlags(), prefill_forward=runtime.ForwardFlags()
        )
        runtime.get_forward.__globals__["_CONTEXT"] = context
        ready, decode_written = threading.Event(), threading.Event()

        def submit_prefill():
            lanes.set_pdmux_status(True)
            with runtime.get_forward().scoped(mlp_reduce_scatter=True):
                runtime.get_forward().set("is_extend_in_batch", True)
                ready.set()
                self.assertTrue(decode_written.wait(5))
                return runtime.get_forward().mlp_reduce_scatter

        with ThreadPoolExecutor(max_workers=1) as pool:
            work = pool.submit(submit_prefill)
            self.assertTrue(ready.wait(5))
            runtime.get_forward().set("is_extend_in_batch", False)
            self.assertFalse(runtime.get_forward().mlp_reduce_scatter)
            decode_written.set()
            self.assertTrue(work.result())
        self.assertFalse(runtime.get_forward().is_extend_in_batch)
        self.assertFalse(context.prefill_forward.mlp_reduce_scatter)

    def test_cp_continuation_restores_physical_length_before_each_segment(self):
        calls = []
        forward = load_methods(
            "model_executor/model_runner.py",
            "ModelRunner",
            ["forward_split_prefill"],
            device_timer_ctx=lambda *a: contextlib.nullcontext(),
        )["forward_split_prefill"]
        # The production method imports CP helpers lazily.
        import sys
        from unittest.mock import patch

        cp_module = NS(
            is_cp_active=lambda _: True,
            prepare_cp_forward=lambda _: calls.append("cp"),
        )
        runner = NS(
            attn_backend=NS(init_forward_metadata=lambda _: calls.append("metadata")),
            model_config=NS(num_hidden_layers=2),
            model=NS(forward_split_prefill=lambda *a: calls.append("model")),
            device_timer=None,
        )
        batch = NS(split_index=0, input_ids=None, positions=None)
        with patch.dict(sys.modules, {"sglang.srt.layers.cp.utils": cp_module}):
            forward(runner, batch)
            forward(runner, batch)
        self.assertEqual(calls, ["cp", "metadata", "model", "cp", "model"])

    def test_prefill_sync_uses_full_cp_group_even_with_dp_attention_flag(self):
        prefill, singleton = object(), object()
        parallel = NS(enable_dp_attention=True, attn_cp_size=2, attn_tp_group=singleton)
        sync = Mock()
        finalize = load_methods(
            "speculative/dspark_components/dspark_worker_v2.py",
            "DSparkWorkerV2",
            ["_finalize_prefill"],
            get_parallel=lambda: parallel,
            get_tp_group=lambda: prefill,
            SpecTpSyncSite=NS(DSPARK_TARGET="target"),
        )["_finalize_prefill"]
        output = NS(logits_output=NS(hidden_states=None), next_token_ids=object())
        worker = NS(_target_hidden_projection_enabled=False, _tp_sync=sync)
        with self.assertRaisesRegex(RuntimeError, "aux hidden capture"):
            finalize(worker, NS(seq_lens=None), output, None)
        sync.sync.assert_called_once_with(
            "target", output.next_token_ids, group=prefill
        )


if __name__ == "__main__":
    unittest.main()
