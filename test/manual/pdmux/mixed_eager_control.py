"""Private worker hook for run_mixed_eager.py, never a serving entry point."""

import importlib
import json
import os
import time
import traceback
from pathlib import Path

from mixed_eager_validation import source_hashes, write_json

from sglang.srt.managers.scheduler import Scheduler

ROOT = Path(os.environ["MIXED_OFFLINE_ROOT"])
REPO = Path(__file__).resolve().parents[3]
IDLE = Scheduler.on_idle
last_check = 0.0
last_nonce = None


def controlled_idle(self, *args, **kwargs):
    global last_check, last_nonce
    if time.monotonic() - last_check >= 1:
        last_check = time.monotonic()
        trigger = ROOT / "trigger.json"
        if trigger.exists():
            config = json.loads(trigger.read_text())
            if config["nonce"] != last_nonce:
                last_nonce = config["nonce"]
                output = ROOT / (last_nonce + "-report.json")
                result = dict(stage="running", config=config, started=time.time())
                write_json(output, result)
                try:
                    import mixed_eager_harness

                    result["numerical_source_sha256"] = source_hashes(REPO)
                    importlib.reload(mixed_eager_harness)
                    result.update(mixed_eager_harness.run(self, config))
                    if result["numerical_source_sha256"] != source_hashes(REPO):
                        raise RuntimeError("Numerical source changed during validation")
                    result["stage"] = "completed"
                except Exception:
                    result.update(
                        stage="failed", gate_passed=False, error=traceback.format_exc()
                    )
                finally:
                    try:
                        result["pools_flushed"] = bool(self.flush_cache())
                        if not result["pools_flushed"]:
                            raise RuntimeError("Private validation cache did not flush")
                    except Exception:
                        result.update(
                            stage="failed",
                            gate_passed=False,
                            cleanup_error=traceback.format_exc(),
                        )
                    result["ended"] = time.time()
                    try:
                        write_json(output, result)
                    except ValueError:
                        # Preserve a terminal error even for nonfinite metrics.
                        write_json(
                            output,
                            dict(
                                stage="failed",
                                gate_passed=False,
                                config=config,
                                error=traceback.format_exc(),
                            ),
                        )
    return IDLE(self, *args, **kwargs)


Scheduler.on_idle = controlled_idle
