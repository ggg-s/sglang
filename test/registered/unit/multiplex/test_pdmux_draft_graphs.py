"""Exercise stream-bound draft graph ownership without a CUDA installation."""

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_pdmux_sxf_port import load_methods, register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPDMuxDraftGraphs(unittest.TestCase):
    def setUp(self):
        self.state = NS(idx=1, prefill=True, pool=object())
        self.original_pool = self.state.pool
        self.groups = [(object(), object()) for _ in range(3)]
        self.worker = NS(
            server_args=object(),
            draft_runner=object(),
            topk=1,
            speculative_num_steps=3,
            seed_dsa_topk_from_draft_extend=False,
            draft_attn_backend=object(),
        )
        self.original_backend = self.worker.draft_attn_backend
        self.factory = NS(
            create_decode_backend=Mock(side_effect=lambda: object()),
            create_draft_extend_backend=Mock(side_effect=lambda: object()),
        )
        self.globals = dict(
            torch=NS(cuda=NS(graph_pool_handle=object)),
            logger=Mock(),
            get_stream_groups=lambda: self.groups,
            get_current_stream_idx=lambda: self.state.idx,
            set_current_stream_idx=lambda v: setattr(self.state, "idx", v),
            is_pdmux_prefill_enabled=lambda: self.state.prefill,
            set_pdmux_status=lambda v: setattr(self.state, "prefill", v),
            get_global_graph_memory_pool=lambda: self.state.pool,
            set_global_graph_memory_pool=lambda v: setattr(self.state, "pool", v),
            DraftBackendFactory=lambda *a, **k: self.factory,
        )
        self.runner_type = type(
            "PDMuxDraftCudaGraphRunner",
            (),
            load_methods(
                "speculative/pdmux_draft_cuda_graph_runner.py",
                "PDMuxDraftCudaGraphRunner",
                ["__init__", "__getattr__", "can_run_graph", "execute"],
                **self.globals,
            ),
        )
        self.captures = []

    def capture(self, worker, *, capture_stream, **kwargs):
        idx = self.state.idx
        backend = next(iter(kwargs.values()))
        self.assertIs(capture_stream, self.groups[idx][1])
        self.assertFalse(self.state.prefill)
        self.assertIsNot(self.state.pool, self.original_pool)
        if "draft_attn_backend" in kwargs:
            self.assertIs(worker.draft_attn_backend, backend)
        runner = NS(
            backend=backend,
            capture_bs=[1, 2, 4],
            max_bs=4,
            buffers=object(),
            can_run_graph=Mock(return_value=idx != 1),
            execute=Mock(return_value=(idx, object())),
        )
        self.captures.append(runner)
        return runner

    def assert_restored(self):
        self.assertEqual(self.state.idx, 1)
        self.assertTrue(self.state.prefill)
        self.assertIs(self.state.pool, self.original_pool)
        self.assertIs(self.worker.draft_attn_backend, self.original_backend)

    def test_capture_and_dispatch_each_group_for_both_phases(self):
        for extend in (False, True):
            with self.subTest(extend=extend):
                wrapper = self.runner_type(
                    self.worker, self.capture, draft_extend=extend
                )
                self.assert_restored()
                self.assertEqual(len({id(r.backend) for r in wrapper.runners}), 3)
                for idx in (0, 1, 2, 0):
                    self.state.idx = idx
                    batch = NS(batch_size=1, can_run_dp_cuda_graph=True)
                    runner = wrapper.runners[idx]
                    self.assertEqual(wrapper.can_run_graph(batch), idx != 1)
                    self.assertIs(wrapper.buffers, runner.buffers)
                    self.assertEqual(wrapper.execute(batch)[0], idx)
                    runner.execute.assert_called_with(batch)
                self.state.idx = 1
        self.assertEqual(len({id(r.backend) for r in self.captures}), 6)

    def test_capture_failure_restores_backend_stream_lane_and_pool(self):
        def fail(worker, **kwargs):
            runner = self.capture(worker, **kwargs)
            if self.state.idx == 1:
                raise RuntimeError("capture failed")
            return runner

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            self.runner_type(self.worker, fail)
        self.assert_restored()

    def test_no_groups_fails_before_mutating_state(self):
        self.groups.clear()
        with self.assertRaisesRegex(RuntimeError, "requires stream groups"):
            self.runner_type(self.worker, self.capture)
        self.assert_restored()

    def test_decode_phases_reuse_only_the_private_pool(self):
        self.runner_type(self.worker, self.capture)
        pool = self.worker._pdmux_draft_graph_memory_pool
        self.runner_type(self.worker, self.capture, draft_extend=True)
        self.assertIs(self.worker._pdmux_draft_graph_memory_pool, pool)
        self.assert_restored()

    def test_parent_capture_uses_requested_stream(self):
        streams = []

        @contextlib.contextmanager
        def graph_capture(*, stream):
            streams.append(stream)
            yield NS(stream=stream)

        capture = load_methods(
            "model_executor/runner/decode_cuda_graph_runner.py",
            "DecodeCudaGraphRunner",
            ["capture"],
            freeze_gc=lambda *a: contextlib.nullcontext(),
            empty_context=contextlib.nullcontext,
            graph_capture=graph_capture,
        )["capture"]
        for stream in (None, self.groups[1][1]):
            runner = NS(
                warmup=Mock(),
                enable_torch_compile=False,
                enable_profile_cuda_graph=False,
                _init_profile_context=lambda: contextlib.nullcontext(),
                buffers=NS(
                    seq_lens=Mock(), seq_lens_cpu=Mock(), reset_index_buffers=Mock()
                ),
                seq_len_fill_value=1,
                model_runner=NS(server_args=NS(enable_cudagraph_gc=False)),
                enable_pdmux=False,
                capture_stream=stream,
                backend=NS(capture_session=lambda s: contextlib.nullcontext()),
                _capture_one_stream=Mock(),
            )
            capture(runner)
            self.assertIs(runner.stream, stream)
            runner._capture_one_stream.assert_called_once_with()
        self.assertEqual(streams, [None, self.groups[1][1]])

    def test_dp_idle_and_busy_ranks_make_same_graph_decision(self):
        for filename, cls in (
            ("eagle_draft_cuda_graph_runner.py", "EAGLEDraftCudaGraphRunner"),
            (
                "eagle_draft_extend_cuda_graph_runner.py",
                "EAGLEDraftExtendCudaGraphRunner",
            ),
        ):
            can_run = load_methods("speculative/" + filename, cls, ["can_run_graph"])[
                "can_run_graph"
            ]
            runner = NS(
                captured_req_width=1,
                require_mlp_tp_gather=True,
                require_mlp_sync=True,
                disable_padding=False,
                max_bs=4,
            )
            for local_bs in (0, 1):
                batch = NS(
                    batch_size=local_bs,
                    spec_info=NS(num_tokens_per_req=1),
                    original_global_num_tokens_cpu=[1, 1, 0],
                    can_run_dp_cuda_graph=True,
                )
                self.assertTrue(can_run(runner, batch))
                batch.can_run_dp_cuda_graph = False
                self.assertFalse(can_run(runner, batch))
                batch.can_run_dp_cuda_graph = True
                batch.original_global_num_tokens_cpu = [5, 1, 0]
                self.assertFalse(can_run(runner, batch))


if __name__ == "__main__":
    unittest.main()
