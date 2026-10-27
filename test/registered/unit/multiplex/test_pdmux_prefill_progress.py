"""Chunk boundaries must not drain an unchanged PDMux stream group."""

import unittest
from types import SimpleNamespace as NS

from test_pdmux_sxf_port import load_methods, register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPdmuxPrefillProgress(unittest.TestCase):
    def test_cp_logs_and_default_metrics_have_one_reporter_per_dp_worker(self):
        for enable_metrics in (False, True):
            for all_schedulers in (False, True):
                obs = NS(
                    enable_metrics=enable_metrics,
                    enable_metrics_for_all_schedulers=all_schedulers,
                    kv_events_config=None,
                    extra_metric_labels={},
                )
                init = load_methods(
                    "observability/metrics_collector.py",
                    "SchedulerMetricsCollector",
                    ["init_new"],
                    get_observability=lambda: obs,
                    get_disagg=lambda: NS(disaggregation_mode="null"),
                    get_serving=lambda: NS(
                        served_model_name="fixture", enable_streaming_session=False
                    ),
                    DisaggregationMode=NS(to_engine_type=lambda mode: mode),
                    resolve_collector_class=lambda *args: lambda **kw: NS(**kw),
                    STAT_LOGGER_ROLE_SCHEDULER="scheduler",
                    SchedulerMetricsCollectorContext=lambda **kw: NS(**kw),
                )["init_new"]
                for cp_size in (1, 2, 8):
                    for dp_rank in (0, 1):
                        reporters = []
                        for cp_rank in range(cp_size):
                            ctx = init(
                                object,
                                server_args=None,
                                ps=NS(
                                    attn_tp_rank=0,
                                    attn_cp_rank=cp_rank,
                                    pp_rank=0,
                                    moe_ep_rank=0,
                                ),
                                tp_rank=cp_rank,
                                pp_rank=0,
                                dp_rank=dp_rank,
                                enable_priority_scheduling=False,
                                enable_lora=False,
                                enable_hierarchical_cache=True,
                            )
                            reporters.append(ctx.is_stats_logging_rank)
                            self.assertEqual(
                                ctx.current_scheduler_metrics_enabled,
                                enable_metrics and (cp_rank == 0 or all_schedulers),
                            )
                        self.assertEqual(sum(reporters), 1)

    def scheduler(self, current=1):
        trace, state = [], [current]
        methods = load_methods(
            "multiplex/multiplexing_mixin.py",
            "SchedulerMultiplexMixin",
            [
                "adjust_stream_groups",
                "_activate_pdmux_stream_group",
                "_select_pdmux_stream_idx",
            ],
            get_current_stream_idx=lambda: state[0],
            set_current_stream_idx=lambda idx: (
                state.__setitem__(0, idx),
                trace.append(("select", idx)),
            ),
        )
        scheduler = type("Scheduler", (), methods)()
        scheduler.stream_groups = [
            tuple(
                NS(synchronize=lambda i=i, lane=lane: trace.append(("drain", i, lane)))
                for lane in ("P", "D")
            )
            for i in range(3)
        ]
        scheduler._update_decode_attn_backends = lambda idx: trace.append(
            ("backend", idx)
        )
        scheduler.pdmux_config = NS(manual_divisions=[[8, 0, 1]])
        scheduler.real_sm_group_num = 3
        return scheduler, trace, state

    def test_repeated_chunk_selection_keeps_pending_decode_and_backend(self):
        scheduler, trace, state = self.scheduler()
        for decode_bs in (1, 3, 2, 7) * 25:
            idx, streams = scheduler.adjust_stream_groups(
                running_batch=NS(batch_size=lambda: decode_bs), has_prefill=True
            )
            self.assertEqual(idx, 1)
            self.assertIs(streams, scheduler.stream_groups[1])
        self.assertEqual(state, [1])
        self.assertEqual(trace, [])

    def test_real_transition_drains_old_group_before_backend_rebind(self):
        scheduler, trace, state = self.scheduler()
        idx, streams = scheduler.adjust_stream_groups(
            running_batch=NS(batch_size=lambda: 4), has_prefill=False
        )
        self.assertEqual(idx, 2)
        self.assertIs(streams, scheduler.stream_groups[2])
        self.assertEqual(
            trace, [("drain", 1, "P"), ("drain", 1, "D"), ("select", 2), ("backend", 2)]
        )
        self.assertEqual(state, [2])
        trace.clear()
        scheduler._activate_pdmux_stream_group(2)
        self.assertEqual(trace, [])
        scheduler.adjust_stream_groups(
            running_batch=NS(batch_size=lambda: 0), has_prefill=True
        )
        self.assertEqual(
            trace, [("drain", 2, "P"), ("drain", 2, "D"), ("select", 0), ("backend", 0)]
        )

    def test_rank_uniform_dp_selection_uses_same_transition_contract(self):
        scheduler, trace, state = self.scheduler(current=0)
        scheduler._activate_pdmux_stream_group(1)
        self.assertEqual(trace[-2:], [("select", 1), ("backend", 1)])
        trace.clear()
        # DP ranks compute this index collectively outside the helper.
        for _ in range(32):
            scheduler._activate_pdmux_stream_group(1)
        self.assertEqual(trace, [])
        self.assertEqual(state, [1])

    def test_temporal_selection_preserves_the_same_physical_group(self):
        scheduler, trace, _ = self.scheduler(current=0)
        scheduler.pdmux_temporal_layers = 4
        for decode_bs, has_prefill in ((0, True), (8, True), (8, False)):
            idx, _ = scheduler.adjust_stream_groups(
                running_batch=NS(batch_size=lambda: decode_bs), has_prefill=has_prefill
            )
            self.assertEqual(idx, 0)
        self.assertEqual(trace, [])

    def test_32k_budget_keeps_global_moe_work_and_avoids_idle_decode_yields(self):
        count = load_methods(
            "multiplex/multiplexing_mixin.py",
            "SchedulerMultiplexMixin",
            ["_get_split_forward_count"],
        )["_get_split_forward_count"]
        scheduler = NS(
            ps=NS(attn_dp_size=1),
            model_config=NS(num_hidden_layers=43),
            pdmux_config=NS(
                split_forward_token_budget=32768, split_forward_max_layers=0
            ),
            split_prefill_batch=NS(split_index=0, extend_num_tokens=16384),
            running_batch=NS(is_empty=lambda: False),
        )
        # TP and CP=TP/DP1 both process this global token count in TP MoE.
        self.assertEqual(count(scheduler), 2)
        scheduler.split_prefill_batch.split_index = 42
        self.assertEqual(count(scheduler), 1)
        scheduler.split_prefill_batch.split_index = 0
        scheduler.running_batch = NS(is_empty=lambda: True)
        self.assertEqual(count(scheduler), 43)
        scheduler.ps.attn_dp_size = 8
        scheduler.split_prefill_batch.global_num_tokens = [2048] * 8
        self.assertEqual(count(scheduler, NS(global_num_tokens=[1] * 8)), 2)
        self.assertEqual(count(scheduler, NS(global_num_tokens=[0] * 8)), 43)


if __name__ == "__main__":
    unittest.main()
