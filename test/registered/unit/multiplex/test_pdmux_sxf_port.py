"""Exercise ported scheduling logic without importing CUDA runtime modules.

Load the production method bodies and substitute only their collaborators. This
also runs on development hosts without torch, Triton, or a GPU.
"""

import ast
import contextlib
import gc
import importlib.util
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
_spec = importlib.util.spec_from_file_location(
    "pdmux_port_ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ci)
register_cpu_ci = _ci.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def load_methods(path, class_name, names, **globals_):
    tree = ast.parse((SRT / path).read_text(encoding="utf-8"))
    cls = (
        next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        if class_name is not None
        else tree
    )
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in methods} == set(names)
    for method in methods:
        method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    namespace = dict(globals_)
    exec(compile(ast.fix_missing_locations(module), str(SRT / path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def bind(instance, methods):
    for name, method in methods.items():
        setattr(instance, name, types.MethodType(method, instance))
    return instance


class TestPdmuxSxfPort(unittest.TestCase):
    def test_split_activations_release_before_batch_becomes_decode(self):
        class ActivationOwner:
            pass

        for running_empty in (True, False):
            with self.subTest(running_empty=running_empty):
                owner = ActivationOwner()
                ref = weakref.ref(owner)
                batch = SimpleNamespace(
                    split_forward_batch=owner,
                    split_prefill_finished=True,
                    split_index=43,
                    split_forward_count=4,
                    chunked_req=None,
                    batch_size=lambda: 1,
                    is_empty=lambda: False,
                    filter_batch=Mock(),
                    reqs=[object()],
                )
                del owner
                order = []

                def consume(*args):
                    self.assertIsNotNone(ref())
                    order.append("consume")

                def merge(other):
                    self.assertIs(other, batch)
                    self.assertIsNone(ref())
                    order.append("merge")

                scheduler = SimpleNamespace(
                    split_prefill_batch=batch,
                    enable_overlap=False,
                    chunked_req=None,
                    process_batch_result=consume,
                    prefill_launched_reqs=SimpleNamespace(leave=Mock()),
                    running_reqs=SimpleNamespace(join=Mock()),
                )
                bind(
                    scheduler,
                    load_methods(
                        "multiplex/multiplexing_mixin.py",
                        "SchedulerMultiplexMixin",
                        [
                            "_merge_finished_prefill_batch",
                            "_merge_completed_prefill_batch",
                        ],
                        publish_carried_tensors=Mock(),
                    ),
                )
                running = SimpleNamespace(
                    is_empty=lambda: running_empty, merge_batch=merge
                )
                event = object()
                prefill = SimpleNamespace(record_event=Mock(return_value=event))
                decode = SimpleNamespace(wait_event=Mock())
                result = scheduler._merge_finished_prefill_batch(
                    object(), prefill, decode, running
                )
                gc.collect()
                self.assertIsNone(ref())
                self.assertIs(result, batch if running_empty else running)
                self.assertEqual(
                    order, ["consume"] if running_empty else ["consume", "merge"]
                )
                self.assertIsNone(scheduler.split_prefill_batch)
                self.assertEqual(
                    (
                        batch.split_index,
                        batch.split_forward_count,
                        batch.split_prefill_finished,
                    ),
                    (0, 1, False),
                )
                decode.wait_event.assert_called_once_with(event)

    def test_peer_dp_idle_batch_participates_in_split_prefill(self):
        idle_mode = SimpleNamespace(is_idle=lambda: True)
        split_mode = object()
        method = load_methods(
            "multiplex/multiplexing_mixin.py",
            "SchedulerMultiplexMixin",
            ["update_split_prefill_batch"],
            ForwardMode=SimpleNamespace(SPLIT_PREFILL=split_mode),
        )["update_split_prefill_batch"]
        for idle in (True, False):
            with self.subTest(idle=idle):
                batch = SimpleNamespace(
                    forward_mode=(
                        idle_mode if idle else SimpleNamespace(is_idle=lambda: False)
                    )
                )
                running = object()
                adapter = SimpleNamespace(
                    maybe_prepare_mlp_sync_batch=Mock(return_value=batch)
                )
                scheduler = SimpleNamespace(
                    split_prefill_batch=None,
                    process_pending_chunked_abort=Mock(),
                    _process_hicache_events=Mock(),
                    get_new_batch_prefill=Mock(
                        return_value=SimpleNamespace(
                            batch_to_run=None, running_batch=running
                        )
                    ),
                    dp_attn_adapter=adapter,
                )
                created, returned = method(scheduler, 128, running)
                self.assertTrue(created)
                self.assertIs(returned, running)
                self.assertIs(batch.forward_mode, idle_mode if idle else split_mode)
                self.assertIsNone(batch.split_forward_batch)
                self.assertEqual(batch.split_index, 0)
                adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(None)

    def test_dp_sync_resolves_the_current_lane_group(self):
        prefill_group, decode_group = object(), object()
        parallel = SimpleNamespace(num_dp_ranks=2, attn_tp_size=4, attn_cp_size=1)
        gather = Mock()
        info = SimpleNamespace(
            all_gather=gather,
            tp0_info_cpu=None,
            global_num_tokens=[0, 0],
            sync_wait_seconds=0.0,
        )
        method = load_methods(
            "managers/scheduler_components/dp_attn.py",
            None,
            ["prepare_mlp_sync_batch_raw"],
            get_parallel=lambda: parallel,
            get_disagg=lambda: SimpleNamespace(enable_pdmux=False),
            _local_decode_cuda_graph_vote=lambda **kwargs: True,
            _spec_input_cuda_graph_compatible=lambda batch: True,
            _local_prefill_cuda_graph_vote=lambda **kwargs: False,
            check_cuda_graph_backend=lambda *args: False,
            Phase=SimpleNamespace(PREFILL="prefill"),
            Backend=SimpleNamespace(BREAKABLE="breakable", FULL="full"),
            TboDPAttentionPreparer=lambda: SimpleNamespace(
                prepare_all_gather=lambda batch: (False, 0)
            ),
            world_dp_gather_enabled=lambda: False,
            should_skip_scheduler_all_gather=lambda n: False,
            MLPSyncBatchInfo=lambda **kwargs: info,
            _local_attention_pairs=lambda batch: 0,
            _ENABLE_METRICS_DP_ATTENTION=False,
        )["prepare_mlp_sync_batch_raw"]
        for group in (prefill_group, decode_group):
            parallel.tp_group = SimpleNamespace(device_group=group, device="cuda")
            method(
                None,
                model_runner=SimpleNamespace(spec_algorithm=None, model_config=None),
                get_idle_batch=Mock(),
                disable_cuda_graph=False,
                require_mlp_tp_gather=True,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )
            self.assertIs(gather.call_args.kwargs["group"], group)

    def test_intermediate_split_unpads_even_without_logits(self):
        method = load_methods(
            "model_executor/forward_batch_info.py",
            "ForwardBatch",
            ["post_forward_mlp_sync_batch"],
            SpecInputType=SimpleNamespace(EAGLE_DRAFT_EXTEND="draft_extend"),
        )["post_forward_mlp_sync_batch"]
        batch = SimpleNamespace(
            _original_forward_mode="split",
            _original_batch_size=2,
            _original_num_tokens=3,
            batch_size=4,
            spec_info=None,
            positions=[0, 1, 2, 0],
            seq_lens=[1, 2, 0, 0],
            req_pool_indices=[5, 6, 0, 0],
            req_pool_indices_cpu=[5, 6, 0, 0],
            seq_lens_cpu=[1, 2, 0, 0],
        )
        method(batch, None)
        self.assertEqual(batch.positions, [0, 1, 2])
        self.assertEqual(batch.req_pool_indices, [5, 6])
        self.assertEqual(batch.seq_lens, [1, 2])
        self.assertEqual(batch.seq_lens_cpu, [1, 2])
        self.assertEqual(batch.forward_mode, "split")

    def test_speculative_idle_unpads_inputs_with_and_without_logits(self):
        method = load_methods(
            "model_executor/forward_batch_info.py",
            "ForwardBatch",
            ["post_forward_mlp_sync_batch"],
            SpecInputType=SimpleNamespace(EAGLE_DRAFT_EXTEND="draft_extend"),
        )["post_forward_mlp_sync_batch"]
        idle_mode = SimpleNamespace(
            is_idle=lambda: True,
            is_decode=lambda: False,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_extend=lambda: False,
        )
        for has_logits in (False, True):
            for has_cpu_lengths in (False, True):
                with self.subTest(logits=has_logits, cpu_lengths=has_cpu_lengths):
                    batch = SimpleNamespace(
                        forward_mode=idle_mode,
                        batch_size=0,
                        spec_info=SimpleNamespace(
                            hidden_states=[], spec_input_type="draft"
                        ),
                        positions=[],
                    )
                    # Each draft step pads again to join peer ranks' forward.
                    # A leftover row would become the next step's "real" size.
                    for padded_width in (1, 8, 2):
                        batch._original_num_tokens = len(batch.positions)
                        batch._original_batch_size = 0
                        batch._original_forward_mode = idle_mode
                        batch.batch_size = padded_width
                        batch.positions = [0] * padded_width
                        batch.seq_lens = [1] * padded_width
                        batch.req_pool_indices = [0] * padded_width
                        batch.req_pool_indices_cpu = [0] * padded_width
                        batch.seq_lens_cpu = (
                            [1] * padded_width if has_cpu_lengths else None
                        )
                        logits = (
                            SimpleNamespace(
                                next_token_logits=[[0.0] * 16] * padded_width,
                                last_hidden_states=None,
                                hidden_states=[[0.0] * 4] * padded_width,
                            )
                            if has_logits
                            else None
                        )
                        method(batch, logits)
                        self.assertEqual(batch.batch_size, 0)
                        self.assertIs(batch.forward_mode, idle_mode)
                        self.assertEqual(batch.positions, [])
                        self.assertEqual(batch.seq_lens, [])
                        self.assertEqual(batch.req_pool_indices, [])
                        self.assertEqual(
                            batch.seq_lens_cpu, [] if has_cpu_lengths else None
                        )
                        if has_logits:
                            self.assertEqual(logits.next_token_logits, [])
                            self.assertEqual(logits.hidden_states, [])
                            self.assertEqual(
                                len(batch.positions), len(logits.next_token_logits)
                            )

    def test_dspark_finalizes_only_the_last_nonidle_split(self):
        capture_mode = SimpleNamespace(FULL=object())
        method = load_methods(
            "speculative/dspark_components/dspark_worker_v2.py",
            "DSparkWorkerV2",
            ["forward_batch_split_prefill"],
            CaptureHiddenMode=capture_mode,
        )["forward_batch_split_prefill"]
        for idle in (True, False):
            with self.subTest(idle=idle):
                intermediate = SimpleNamespace(logits_output=None)
                final = SimpleNamespace(logits_output=object())
                placement = object()
                worker = SimpleNamespace(
                    _deferred_commit=lambda: None,
                    _kv_injector=SimpleNamespace(
                        place_prefill=Mock(return_value=placement)
                    ),
                    target_worker=SimpleNamespace(
                        forward_batch_split_prefill=Mock(
                            side_effect=[intermediate, final]
                        )
                    ),
                    _verify_planner=SimpleNamespace(note_non_decode_step=Mock()),
                    _observers=SimpleNamespace(note_prefill_step=Mock()),
                    _finalize_prefill=Mock(return_value="final"),
                    _decode_idle_result=Mock(return_value="idle"),
                )
                batch = SimpleNamespace(
                    split_index=0, forward_mode=SimpleNamespace(is_idle=lambda: idle)
                )
                self.assertIs(method(worker, batch), intermediate)
                worker._finalize_prefill.assert_not_called()
                batch.split_index = 1
                self.assertEqual(method(worker, batch), "idle" if idle else "final")
                worker._verify_planner.note_non_decode_step.assert_called_once_with()
                worker.target_worker.forward_batch_split_prefill.assert_called_with(
                    batch, capture_hidden_mode=capture_mode.FULL
                )
                if idle:
                    worker._finalize_prefill.assert_not_called()
                else:
                    worker._finalize_prefill.assert_called_once_with(
                        batch, final, None, placement
                    )

    def test_dspark_scheduler_proxy_reaches_target_prefill(self):
        methods = load_methods(
            "speculative/dspark_components/dspark_worker_v2.py",
            "DSparkWorkerV2",
            ["forward_batch_generation", "_forward_prefill"],
            CaptureHiddenMode=SimpleNamespace(FULL="full"),
        )
        worker = bind(
            SimpleNamespace(
                _hosts_draft=True,
                enable_dp_spec_prefill_coordination=False,
                _deferred_commit=lambda: None,
                _kv_injector=SimpleNamespace(place_prefill=Mock(return_value=object())),
                _verify_planner=SimpleNamespace(note_non_decode_step=Mock()),
                _observers=SimpleNamespace(note_prefill_step=Mock()),
                target_worker=SimpleNamespace(
                    forward_batch_generation=Mock(return_value="target")
                ),
                _finalize_prefill=Mock(return_value="prefill"),
                _forward_decode=Mock(return_value="decode"),
            ),
            methods,
        )
        proxy = object()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True, is_idle=lambda: False),
            is_extend_in_batch=True,
        )
        self.assertEqual(
            worker.forward_batch_generation(batch, pp_proxy_tensors=proxy), "prefill"
        )
        worker.target_worker.forward_batch_generation.assert_called_once_with(
            batch, pp_proxy_tensors=proxy, capture_hidden_mode="full"
        )
        batch.forward_mode.is_extend = lambda: False
        batch.is_extend_in_batch = False
        self.assertEqual(
            worker.forward_batch_generation(batch, pp_proxy_tensors=proxy), "decode"
        )
        worker._forward_decode.assert_called_once_with(batch, None, None)

    def test_dspark_finalize_uses_the_current_version_interface(self):
        method = load_methods(
            "speculative/dspark_components/dspark_worker_v2.py",
            "DSparkWorkerV2",
            ["_finalize_prefill"],
            SpecTpSyncSite=SimpleNamespace(DSPARK_TARGET="target"),
            get_parallel=lambda: SimpleNamespace(enable_dp_attention=False),
            get_dp_tp_group=lambda: None,
        )["_finalize_prefill"]
        batch = SimpleNamespace(seq_lens=object())
        output = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=None), next_token_ids=None
        )
        with self.assertRaisesRegex(RuntimeError, "aux hidden capture"):
            method(
                SimpleNamespace(
                    _target_hidden_projection_enabled=False, _tp_sync=Mock()
                ),
                batch,
                output,
                None,
                object(),
            )

    def test_split_dspark_captures_use_current_cp_gather(self):
        stream = object()
        hidden = MagicMock()
        packed_aux, capture_slot = object(), object()
        captures = SimpleNamespace(
            slot=lambda like: capture_slot, finalize=lambda: packed_aux
        )
        pre_head, normalized = object(), object()
        hidden.flatten.return_value = pre_head
        gathered_hidden, gathered_pre_head, gathered_aux = object(), object(), object()
        gather = Mock(side_effect=[(gathered_hidden, gathered_pre_head), gathered_aux])
        method = load_methods(
            "models/deepseek_v4.py",
            "DeepseekV4Model",
            ["forward_split_prefill"],
            is_cp_active=lambda batch: True,
            cp_gather_after_forward=gather,
            torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: stream)),
            check_cuda_graph_backend=lambda *args: True,
            Phase=SimpleNamespace(PREFILL="prefill"),
            Backend=SimpleNamespace(TC_PIECEWISE="piecewise"),
            nullcontext=contextlib.nullcontext,
            _is_cuda=True,
            residual_batch=SimpleNamespace(
                read_output=lambda *args: hidden, clear=lambda batch: None
            ),
            hc_boundary=SimpleNamespace(
                materialize=lambda *args, **kwargs: hidden, FinalRead=lambda: None
            ),
        )["forward_split_prefill"]
        layer = Mock(return_value=hidden)
        model = SimpleNamespace(
            layers=[None, layer],
            end_layer=2,
            dspark_layers_to_capture=[0, 1],
            use_fused_mhc_post_pre=False,
            hc_pre_from_prev_sublayer=False,
            hc_head=Mock(return_value=hidden),
            hc_head_fn=None,
            hc_head_scale=None,
            hc_head_base=None,
            norm=Mock(return_value=normalized),
        )
        batch = SimpleNamespace(
            hidden_states=hidden,
            model_specific_states=dict(
                prev_residual=None,
                prev_post=None,
                prev_comb=None,
                positions=object(),
                input_ids=object(),
                input_ids_global=object(),
                dspark_aux_hidden_states=captures,
            ),
        )
        mhc = types.ModuleType("sglang.kernels.ops.layernorm.mhc")
        mhc.mhc_stream_mean = Mock()
        with patch.dict("sys.modules", {mhc.__name__: mhc}):
            result, captures = method(model, None, None, batch, (1, 2))
        self.assertEqual(result, (gathered_hidden, gathered_pre_head))
        self.assertIs(captures, gathered_aux)
        self.assertEqual(
            [call.args for call in gather.call_args_list],
            [((normalized, pre_head), batch, stream), (packed_aux, batch, stream)],
        )

    def test_split_prefill_keeps_cuda_residual_state_across_segments(self):
        """Yielding a split must retain the residual update owed by its last layer."""

        class Hidden:
            def unsqueeze(self, dim):
                return self

            def repeat(self, *shape):
                return self

            def flatten(self, dim):
                return self

        hidden = Hidden()
        batch = SimpleNamespace(residual_stream=None)

        def start(batch):
            batch.residual_stream = []

        def layer(*, hidden_states, forward_batch, **kwargs):
            forward_batch.residual_stream.append(hidden_states)
            return hidden_states

        def read_output(value, batch, *args):
            self.assertEqual(batch.residual_stream, [hidden, hidden])
            return value

        residual = SimpleNamespace(
            start=start,
            read_output=read_output,
            clear=lambda batch: setattr(batch, "residual_stream", None),
        )
        method = load_methods(
            "models/deepseek_v4.py",
            "DeepseekV4Model",
            ["forward_split_prefill"],
            _is_cuda=True,
            residual_batch=residual,
            hc_boundary=SimpleNamespace(FinalRead=lambda: None),
            get_parallel=lambda: SimpleNamespace(attn_dp_size=1),
            is_cp_active=lambda batch: False,
            check_cuda_graph_backend=lambda *args: True,
            Phase=SimpleNamespace(PREFILL="prefill"),
            Backend=SimpleNamespace(TC_PIECEWISE="piecewise"),
            nullcontext=contextlib.nullcontext,
        )["forward_split_prefill"]
        model = SimpleNamespace(
            layers=[layer, layer],
            end_layer=2,
            hc_mult=4,
            embed_tokens=lambda ids: hidden,
            dspark_layers_to_capture=None,
            use_fused_mhc_post_pre=True,
            hc_pre_from_prev_sublayer=False,
            hc_head=lambda value, *args: value,
            hc_head_fn=None,
            hc_head_scale=None,
            hc_head_base=None,
            norm=lambda value: value,
        )
        self.assertIsNone(method(model, None, None, batch, (0, 1)))
        self.assertEqual(batch.residual_stream, [hidden])
        self.assertEqual(method(model, None, None, batch, (1, 2)), (hidden, hidden))
        self.assertIsNone(batch.residual_stream)

    def test_standard_accepts_dp_and_retains_other_parallelism_guards(self):
        backend_module = types.ModuleType("sglang.srt.model_executor.cuda_graph_config")
        backend_module.Backend = SimpleNamespace(DISABLED="disabled")
        method = load_methods(
            "arg_groups/validation_hook.py", None, ["check_pdmux_standard_prefill"]
        )["check_pdmux_standard_prefill"]
        args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            ),
            enable_dp_attention=True,
            enable_multi_layer_eagle=False,
            enable_two_batch_overlap=False,
            enable_unified_memory=False,
            ep_size=1,
            attn_cp_size=1,
            dcp_size=1,
        )
        with patch.dict("sys.modules", {backend_module.__name__: backend_module}):
            method(args)
            args.ep_size = 2
            with self.assertRaisesRegex(AssertionError, "ep-size"):
                method(args)


if __name__ == "__main__":
    unittest.main()
