"""Exercise ported scheduling logic without importing CUDA runtime modules.

Load the production method bodies and substitute only their collaborators. This
also runs on development hosts without torch, Triton, or a GPU.
"""

import ast
import gc
import importlib.util
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
_spec = importlib.util.spec_from_file_location(
    "pdmux_port_ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ci)
_ci.register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def load_methods(path, class_name, names, **globals_):
    tree = ast.parse((SRT / path).read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
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
                    chunked_req=None,
                    process_batch_result=consume,
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
        get_group = Mock(side_effect=[prefill_group, decode_group])
        prepare = Mock(return_value="prepared")
        method = load_methods(
            "managers/scheduler_components/dp_attn.py",
            "SchedulerDPAttnAdapter",
            ["prepare_mlp_sync_batch"],
            prepare_mlp_sync_batch_raw=prepare,
            get_tp_group=get_group,
            get_parallel=lambda: SimpleNamespace(dp_size=2, dwdp_size=1),
            get_schedule=lambda: SimpleNamespace(disable_overlap_schedule=True),
            cuda_graph_fully_disabled=lambda: False,
            require_mlp_tp_gather=lambda args: True,
        )["prepare_mlp_sync_batch"]
        adapter = SimpleNamespace(
            model_runner=object(),
            ps=SimpleNamespace(attn_tp_size=4, attn_cp_size=1),
            get_idle_batch=Mock(),
            server_args=object(),
            offload_tags=set(),
            tp_group=object(),
        )
        for group in (prefill_group, decode_group):
            self.assertEqual(method(adapter, None), "prepared")
            self.assertIs(prepare.call_args.kwargs["tp_group"], group)

    def test_intermediate_split_unpads_even_without_logits(self):
        method = load_methods(
            "model_executor/forward_batch_info.py",
            "ForwardBatch",
            ["post_forward_mlp_sync_batch"],
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
            seq_lens_cpu=[1, 2, 0, 0],
        )
        method(batch, None)
        self.assertEqual(batch.positions, [0, 1, 2])
        self.assertEqual(batch.req_pool_indices, [5, 6])
        self.assertEqual(batch.seq_lens, [1, 2])
        self.assertEqual(batch.seq_lens_cpu, [1, 2])
        self.assertEqual(batch.forward_mode, "split")

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
                worker = SimpleNamespace(
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
                        batch, final, on_publish=None
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
        )["_finalize_prefill"]
        batch = SimpleNamespace(seq_lens=object())
        output = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=None), next_token_ids=None
        )
        with self.assertRaisesRegex(RuntimeError, "aux hidden capture"):
            method(SimpleNamespace(), batch, output, None)

    def test_standard_accepts_dp_and_retains_other_parallelism_guards(self):
        backend_module = types.ModuleType("sglang.srt.model_executor.cuda_graph_config")
        backend_module.Backend = SimpleNamespace(DISABLED="disabled")
        method = load_methods(
            "server_args.py", "ServerArgs", ["_check_pdmux_standard_prefill"]
        )["_check_pdmux_standard_prefill"]
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
