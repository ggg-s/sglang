"""Two real Gloo ranks: native reservation consensus and freeze-before-mutation."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import torch.distributed as dist

# Load lightweight core transport contracts without importing SGLang's serving
# package in these isolated CPU/Gloo pool fixtures.
_package = types.ModuleType("sglang")
_package.__path__ = [str(Path(__file__).resolve().parents[3] / "python/sglang")]
sys.modules.setdefault("sglang", _package)
from qwen_reservation_probe_support import Fixture, Sampling
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import qwen_reservation as reservation

rank = int(sys.argv[1])
root = Path(sys.argv[2])
rendezvous = Path(sys.argv[3])
output = Path(sys.argv[4])
dist.init_process_group(
    "gloo", init_method="file://" + str(rendezvous), rank=rank, world_size=2
)
f = Fixture(root)
try:
    f.parallel.tp_size = 2
    f.parallel.tp_rank = rank
    f.parallel.tp_group = NS(cpu_group=dist.group.WORLD)
    req = f.new(f.scheduler, "source", list(range(5)), [11, 12, 13], Sampling())
    req.kv.req_pool_idx = 0
    req.kv.kv_committed_len = req.kv.kv_allocated_len = 7
    f.scheduler.handoff_tracked_reqs[req.rid] = req
    p = reservation.plan(
        f.scheduler,
        dict(
            epoch="tp-native",
            items=[dict(rid="source", target_rid="target", growth_tokens=8)],
        ),
    )["plans"]
    plans = [None, None]
    dist.all_gather_object(plans, p)
    result = reservation.reserve(f.scheduler, dict(epoch="tp-native", plans=plans))
    reservation.reserve(f.scheduler, dict(epoch="tp-native", plans=plans))
    certificates = [None, None]
    dist.all_gather_object(certificates, result["reservations"])
    reservation.confirm(f.scheduler, dict(epoch="tp-native", certificates=certificates))
    if rank == 1:
        req.kv.kv_committed_len = 16
    try:
        rt._handle(
            f.scheduler,
            dict(cmd="freeze", rid="source", epoch="tp-native", transfer="remote"),
        )
        raise AssertionError("partial TP freeze admitted")
    except ValueError:
        pass
    assert not getattr(req, "_handoff_frozen", False)
    rec = f.scheduler._handoff_reserved["target"]
    if rank == 1:
        f.scheduler._handoff_reserved.pop("target")
    before = f.pool.available_size()
    try:
        reservation.reserve(f.scheduler, dict(epoch="tp-native", plans=plans))
        raise AssertionError("asymmetric reservation ownership admitted")
    except ValueError:
        pass
    assert f.pool.available_size() == before and rec["native_req"].kv.holds_kv
    output.write_text(
        json.dumps(
            dict(
                rank=rank,
                native_reservation_once=True,
                asymmetric_freeze_rejected=True,
                asymmetric_ticket_rejected=True,
                unproven_row_retained=True,
            )
        )
    )
finally:
    f.close()
    dist.destroy_process_group()
