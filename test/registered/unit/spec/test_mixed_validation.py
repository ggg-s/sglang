"""Validate offline gates and preparation without loading torch or CUDA."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
MANUAL = ROOT / "test/manual/pdmux"
sys.path.insert(0, str(MANUAL))
import mixed_eager_validation as validation
import run_mixed_eager as driver

# Register CPU CI without importing the serving platform from sglang.__init__.
spec = importlib.util.spec_from_file_location(
    "_mixed_validation_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ci
spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def gate_report():
    metrics = {"conv": dict(mismatch=0, elements=8, max_abs=0.0, rel_l2=0.0)}
    return dict(
        stage="completed",
        gate_passed=True,
        pools_flushed=True,
        formal_serving_changed=False,
        tolerances=dict(validation.TOLERANCES),
        numerical_source_sha256=validation.source_hashes(ROOT),
        config=validation.build_jobs(smoke=False, numerics="stable", mixed_graph=True)[
            0
        ],
        draft_graph_hits=[True] * 9,
        cases=[
            dict(
                turn=i,
                p_tokens=n,
                p_capacity=capacity,
                proposal_ids=[[11] * 8, [22] * 8, [33] * 8],
                acceptance_equal=True,
                native_acceptance_equal=True,
                native_verify_graph=True,
                safety_passed=True,
                safety_checks=[dict(padding=True, rejected_tail=True)],
                state_mismatches={},
                state_comparison=copy.deepcopy(metrics),
                repeat_control_metrics=copy.deepcopy(metrics),
            )
            for i, (n, capacity) in enumerate(
                zip(validation.EAGER_CHUNKS, validation.EAGER_CAPACITIES)
            )
        ],
    )


class TestMixedValidation(unittest.TestCase):
    def check_gate(self, report):
        return validation.validate_eager_gate(
            report, expected_sources=validation.source_hashes(ROOT), rejection=False
        )

    def test_complete_gate_and_lane_identity(self):
        report = gate_report()
        self.assertEqual(
            self.check_gate(report), validation.proposals_digest(report["cases"])
        )
        with self.assertRaises(ValueError):
            validation.validate_eager_gate(
                report, expected_sources=validation.source_hashes(ROOT), rejection=True
            )

    def test_missing_repeat_is_not_a_pass(self):
        report = gate_report()
        del report["cases"][4]["repeat_control_metrics"]
        with self.assertRaises(ValueError):
            self.check_gate(report)

    def test_stale_or_missing_source_evidence_is_rejected(self):
        for value in ({}, {"old.py": "stale"}):
            report = gate_report()
            report["numerical_source_sha256"] = value
            with self.assertRaises(ValueError):
                self.check_gate(report)

    def test_no_diagnostic_can_qualify_a_graph(self):
        for flag in validation.DIAGNOSTIC_FLAGS:
            report = gate_report()
            report["config"][flag] = True
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.check_gate(report)

    def test_wrong_tolerance_or_incomplete_cleanup_is_rejected(self):
        for change in (
            dict(tolerances=dict(rtol=0.03, atol=0.002)),
            dict(pools_flushed=False),
            dict(stage="failed"),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check_gate(gate_report() | change)

    def test_actual_metrics_override_forged_pass_flags(self):
        for field in ("state_comparison", "repeat_control_metrics"):
            for change in (dict(mismatch=1), dict(max_abs=float("nan"))):
                report = gate_report()
                report["cases"][2][field]["conv"].update(change)
                with (
                    self.subTest(field=field, change=change),
                    self.assertRaises(ValueError),
                ):
                    self.check_gate(report)
        report = gate_report()
        report["cases"][2]["safety_checks"][0]["padding"] = False
        with self.assertRaises(ValueError):
            self.check_gate(report)

    def test_native_graph_coverage_cannot_be_missing(self):
        report = gate_report()
        report["draft_graph_hits"][5] = False
        with self.assertRaises(ValueError):
            self.check_gate(report)
        report = gate_report()
        report["cases"][0]["native_verify_graph"] = False
        with self.assertRaises(ValueError):
            self.check_gate(report)

    def test_changed_proposals_fail_the_prepared_digest(self):
        report = gate_report()
        report["config"]["proposals_sha256"] = validation.proposals_digest(
            report["cases"]
        )
        report["cases"][7]["proposal_ids"][2][3] += 1
        with self.assertRaisesRegex(ValueError, "proposals differ"):
            self.check_gate(report)

    def test_partial_or_invalid_verify_blocks_are_rejected(self):
        for ids in ([[0] * 7] * 3, [[0] * 8] * 2, [[False] * 8] * 3, [[-1] * 8] * 3):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validation.validate_proposals(
                    dict(cases=[dict(proposal_ids=ids)]), rounds=1
                )
        with self.assertRaises(ValueError):
            validation.validate_proposals(gate_report(), rounds=10)

    def test_graph_plan_cannot_bypass_full_stable_eager(self):
        for smoke, numerics in ((True, "stable"), (False, "original")):
            with self.assertRaises(ValueError):
                validation.build_jobs(smoke=smoke, numerics=numerics, mixed_graph=True)
        jobs = validation.build_jobs(smoke=False, numerics="stable", mixed_graph=True)
        self.assertEqual(
            [job["mixed_graph_candidate"] for job in jobs], [False, False, True, True]
        )
        self.assertTrue(all(job["rounds"] == 9 for job in jobs))

    def test_prepare_only_cli_freezes_proposals_without_starting_a_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            baseline = folder / "base.json"
            validation.write_json(
                baseline,
                dict(REPLAYSSM_SPEC="1", MEM_FRACTION="0.9", MAX_TOTAL_TOKENS="65536"),
            )
            report = gate_report()
            for lane in ("natural", "rejection"):
                validation.write_json(
                    folder / f"native-reductions-{lane}-report.json", report
                )
            output = folder / "prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    str(MANUAL / "run_mixed_eager.py"),
                    "--base-config",
                    str(baseline),
                    "--model-path",
                    "/models/Qwen3.5-35B-A3B-AWQ",
                    "--draft-model-path",
                    "/models/Qwen3.5-35B-A3B-DFlash",
                    "--output-dir",
                    str(output),
                    "--proposals-from",
                    str(folder),
                    "--mixed-graph",
                    "--prepare-only",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((output / "plan.json").read_text())
            self.assertFalse(plan["gpu_executed"])
            self.assertIsNone(plan["correctness_passed"])
            self.assertEqual(len(plan["jobs"]), 4)
            self.assertFalse((output / "server.log").exists())
            self.assertFalse((output / "server-process.json").exists())
            for job in plan["jobs"]:
                frozen = json.loads(Path(job["proposals_file"]).read_text())
                self.assertEqual(
                    validation.validate_proposals(frozen, rounds=9),
                    job["proposals_sha256"],
                )
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["MEM_FRACTION"], "0.9")
            self.assertEqual(config["MAX_TOTAL_TOKENS"], "65536")

    def test_preparation_rejects_configuration_drift_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for drift in (
                dict(BLOCK_SIZE="4"),
                dict(DTYPE="bfloat16"),
                dict(DETERMINISTIC="1"),
                dict(DECODE_GRAPH="disabled"),
                dict(PDMUX="1"),
            ):
                baseline = folder / "base.json"
                validation.write_json(baseline, dict(REPLAYSSM_SPEC="1") | drift)
                args = types.SimpleNamespace(
                    base_config=baseline,
                    output_dir=folder / "output",
                    smoke=False,
                    numerics="stable",
                    mixed_graph=True,
                    proposals_from=None,
                    compat_script=None,
                )
                with self.subTest(drift=drift), self.assertRaises(ValueError):
                    driver.prepare_run(args)
                self.assertFalse(args.output_dir.exists())

    def test_graph_without_previous_inputs_replays_its_own_eager_proposals(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            baseline = folder / "base.json"
            validation.write_json(baseline, dict(REPLAYSSM_SPEC="1"))
            args = types.SimpleNamespace(
                base_config=baseline,
                output_dir=folder / "output",
                smoke=False,
                numerics="stable",
                mixed_graph=True,
                proposals_from=None,
                compat_script=None,
                model_path="/models/Qwen3.5-35B-A3B-AWQ",
                draft_model_path="/models/Qwen3.5-35B-A3B-DFlash",
                port=39927,
            )
            _, _, _, _, plan = driver.prepare_run(args)
            for lane, job in zip(("natural", "rejection"), plan["jobs"][2:]):
                self.assertTrue(job["proposals_from_eager_gate"])
                self.assertEqual(
                    Path(job["proposals_file"]).name, f"native-{lane}-report.json"
                )
            self.assertNotIn("proposals_file", plan["jobs"][0])

    def test_idle_hook_records_cleanup_failures_and_nonfinite_metrics(self):
        for outcome in (
            "success",
            "flush_false",
            "flush_error",
            "nonfinite",
            "run_error",
        ):
            with (
                self.subTest(outcome=outcome),
                tempfile.TemporaryDirectory() as temporary,
            ):
                folder = Path(temporary)
                validation.write_json(folder / "trigger.json", dict(nonce="test"))

                class Scheduler:
                    def on_idle(self, *args, **kwargs):
                        return "native idle"

                    def flush_cache(self):
                        if outcome == "flush_error":
                            raise RuntimeError("flush error")
                        return outcome != "flush_false"

                scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")
                scheduler_module.Scheduler = Scheduler
                harness = types.ModuleType("mixed_eager_harness")

                def run(*args):
                    if outcome == "run_error":
                        raise RuntimeError("run error")
                    return dict(
                        gate_passed=True,
                        metric=float("nan") if outcome == "nonfinite" else 0,
                    )

                harness.run = run
                spec = importlib.util.spec_from_file_location(
                    "_private_mixed_control", MANUAL / "mixed_eager_control.py"
                )
                control = importlib.util.module_from_spec(spec)
                with (
                    patch.dict("os.environ", MIXED_OFFLINE_ROOT=str(folder)),
                    patch.dict(
                        sys.modules,
                        {
                            "sglang.srt.managers.scheduler": scheduler_module,
                            "mixed_eager_harness": harness,
                        },
                    ),
                ):
                    spec.loader.exec_module(control)
                    with patch.object(
                        control.importlib, "reload", return_value=harness
                    ):
                        self.assertEqual(
                            control.controlled_idle(Scheduler()), "native idle"
                        )
                result = json.loads((folder / "test-report.json").read_text())
                self.assertEqual(
                    result["stage"], "completed" if outcome == "success" else "failed"
                )
                self.assertEqual(result["gate_passed"], outcome == "success")
                self.assertFalse(list(folder.glob("*.pending")))


if __name__ == "__main__":
    unittest.main()
