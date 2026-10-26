"""Check private scheduler handoff/lifetime without importing the CUDA stack."""

import ast
import copy
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "_live_mixed_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ci
spec.loader.exec_module(ci)
ci.register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def controller_class(namespace):
    source = ROOT / "test/manual/pdmux/live_mixed_target.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LiveMixedTarget"
    )
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == "process_batch_result":
            module = ast.Module(body=[function], type_ignores=[])
            exec(compile(module, str(source), "exec"), namespace)
    return namespace["LiveMixedTarget"]


class TestLiveMixedTarget(unittest.TestCase):
    def exercise(self, fail=False, overlap=False):
        old_backend, mixed_backend = object(), object()
        original_target = lambda *a, **kw: self.fail("unexpected native target call")
        mr = NS(
            attn_backend=old_backend,
            _maybe_execute_deferred_mamba_cow_and_clear=lambda batch: None,
            sample=lambda output, batch: [42],
        )
        worker = NS(
            target_worker=NS(
                model_runner=mr,
                forward_batch_generation=original_target,
                set_hicache_consumer=lambda index: None,
            )
        )
        decode = NS(is_empty=lambda: False, reqs=[object()] * 3, copy=lambda: decode)
        prefill = NS(hicache_consumer_index=0, reqs=[object()])
        pfb, vfb = NS(input_ids=[0] * 6970), NS(batch_size=3)
        events = []
        scheduler = NS(
            running_batch=decode,
            future_map=None,
            update_running_batch=lambda batch: batch,
            _isolate_forward_fields=lambda batch: nullcontext(),
            process_batch_result=lambda batch, result: events.append(
                "processed_decode"
            ),
            enable_overlap=overlap,
            forward_stream_ctx=nullcontext(),
            forward_stream=NS(wait_stream=lambda stream: None),
            schedule_stream=object(),
            batch_record_buf=[[object()]],
            batch_record_ct=0,
        )
        pout = NS(hidden_states=object(), next_token_logits=NS(clone=lambda: "owned"))
        vout = NS(hidden_states=object())
        original_p_hidden, original_v_hidden = pout.hidden_states, vout.hidden_states

        def run_batch(sched, batch, **kwargs):
            if batch is decode:
                result = worker.target_worker.forward_batch_generation(
                    None, forward_batch=vfb, is_verify=True
                )
                self.assertIs(mr.attn_backend, mixed_backend)
                result.logits_output.hidden_states = None
                if fail:
                    raise RuntimeError("decode failed")
                events.append("decode")
                return result
            self.assertEqual(events, ["decode"] if overlap else ["decode", "processed_decode"])
            self.assertIs(mr.attn_backend, old_backend)
            result = worker.target_worker.forward_batch_generation(prefill)
            self.assertEqual(result.logits_output.next_token_logits, "owned")
            result.logits_output.hidden_states = None
            events.append("prefill")
            return result

        namespace = dict(
            MODE="mixed",
            RUN_BATCH=run_batch,
            resolve_forward_inputs=lambda *a: None,
            ForwardBatch=NS(init_new=lambda *a, **kw: pfb),
            CaptureHiddenMode=NS(FULL="full"),
            GenerationBatchResult=NS,
            copy=copy,
            nullcontext=nullcontext,
            PROCESS_RESULT=lambda sched, batch, result: events.append(
                "processed_decode" if batch is decode else (
                    "processed_prefill" if batch is prefill else "processed_previous"
                )
            ),
        )
        cls = controller_class(namespace)
        controller = cls.__new__(cls)
        controller.scheduler, controller.worker, controller.mr = scheduler, worker, mr
        controller.stats = dict(pairs=0, mixed=0, fallbacks=0, mixed_prefill_tokens=0)
        controller._mixed = lambda *a: (
            NS(verify_backend=mixed_backend), (pout, vout, [])
        )
        if fail:
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                controller.run(prefill)
        else:
            result = controller.run(prefill)
            self.assertEqual(result.next_token_ids, [42])
            if overlap:
                self.assertEqual(events, ["decode", "prefill"])
                namespace["process_batch_result"](scheduler, object(), NS())
                namespace["process_batch_result"](scheduler, prefill, result)
                self.assertEqual(events[-3:], ["processed_previous", "processed_decode", "processed_prefill"])
                self.assertFalse(hasattr(result, "_mixed_decode_keepalive"))
            else:
                self.assertEqual(events, ["decode", "processed_decode", "prefill"])
        self.assertIs(mr.attn_backend, old_backend)
        self.assertIs(worker.target_worker.forward_batch_generation, original_target)
        self.assertIs(pout.hidden_states, original_p_hidden)
        self.assertIs(vout.hidden_states, original_v_hidden)

    def test_commit_backend_and_captured_output_lifetime(self):
        self.exercise()

    def test_failure_restores_target_and_backend(self):
        self.exercise(fail=True)

    def test_overlap_keeps_result_queue_order_and_decode_lifetime(self):
        self.exercise(overlap=True)

    def test_serving_limits_are_group_independent(self):
        source = ROOT / "test/manual/pdmux/run_live_mixed_target.py"
        node = next(
            node for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "serving_settings"
        )
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        args = NS(chunked_prefill_size=None, max_prefill_tokens=None, cpu_overlap=None, hicache=None)
        settings = namespace["serving_settings"](dict(CPU_OVERLAP="1", HICACHE="0"), args)
        self.assertEqual(settings, dict(chunk=None, max_prefill="32768", cpu_overlap="1", hicache="0"))
        inherited = namespace["serving_settings"](dict(CHUNKED_PREFILL_SIZE="4096"), args)
        self.assertEqual(inherited["chunk"], "4096")
        self.assertEqual(inherited["max_prefill"], "32768")

    def test_tailfill_respects_chunk_and_whole_verify_blocks(self):
        source = ROOT / "test/manual/pdmux/tailfill_policy.py"
        namespace = {}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        slots = namespace["decode_slots"]
        self.assertEqual(slots(4096, 4096, 32768, 3, 8), 0)
        self.assertEqual(slots(4090, 4096, 32768, 3, 8), 0)
        self.assertEqual(slots(4080, 4096, 32768, 3, 8), 2)
        self.assertEqual(slots(3000, 4096, 32768, 3, 8), 3)
        self.assertEqual(slots(2040, 4096, 2048, 3, 8), 1)
        self.assertEqual(slots(100, None, 32768, 3, 8), 0)

    def test_partial_decode_keeps_unselected_metadata_independent(self):
        source = ROOT / "test/manual/pdmux/tailfill_policy.py"
        namespace = {}
        exec(compile(source.read_text(), str(source), "exec"), namespace)

        class Batch:
            def __init__(self):
                self.reqs = [0, 1, 2]
                self.spec_info = NS(rows=[10, 11, 12])
                self.sampling_info = NS(vocab_size=42, rows=[0, 1, 2])

            def batch_size(self):
                return len(self.reqs)

            def filter_batch(self, *, keep_indices):
                self.reqs = [self.reqs[i] for i in keep_indices]
                self.spec_info.rows = [self.spec_info.rows[i] for i in keep_indices]
                self.sampling_info.rows = [self.sampling_info.rows[i] for i in keep_indices]

        factory = NS(from_schedule_batch=lambda b, v: NS(vocab_size=v, rows=b.reqs[:]))
        module = NS(SamplingBatchInfo=factory)
        original = Batch()
        with patch.dict(sys.modules, {"sglang.srt.sampling.sampling_batch_info": module}):
            selected, pending = namespace["split_running"](original, 2)
        self.assertEqual(selected.reqs, [0, 1])
        self.assertEqual(pending.reqs, [2])
        self.assertEqual(original.spec_info.rows, [10, 11, 12])
        self.assertEqual(original.sampling_info.rows, [0, 1, 2])
        self.assertIsNot(selected.spec_info, pending.spec_info)


if __name__ == "__main__":
    unittest.main()
