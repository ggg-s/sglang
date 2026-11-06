"""Real Mooncake classic metadata and forced TCP on the CUDA validation host.

No HCA is emulated: MC_FORCE_HCA exercises override precedence, not RDMA.
"""

import argparse
import ast
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
from unittest.mock import patch

import torch
from mooncake.engine import TransferEngine
from sglang.srt import mooncake_contract as contract
from sglang_remote_attn import mooncake_backend as mb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = (
        args.source_root
        / "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py"
    )
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MooncakeTransferEngine"
    )
    fn = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize"
    )
    scope = dict(
        Optional=Optional,
        os=os,
        logger=logging.getLogger(__name__),
        envs=NS(
            ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=NS(get=lambda: False),
            MOONCAKE_PROTOCOL=NS(get=lambda: "rdma"),
            SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT=NS(get=lambda: 10),
        ),
        run_with_deadline=lambda fn, **kw: fn(),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    cases, retained = [], []
    for name in (*contract.NON_CLASSIC, "MC_FORCE_HCA", "MC_FORCE_TCP"):
        os.environ.pop(name, None)
    for value in ("1", "0", "", "true", None):
        if value is None:
            os.environ.pop("MC_FORCE_TCP", None)
        else:
            os.environ["MC_FORCE_TCP"] = value
        native = NS(engine=TransferEngine())
        retained.append(native)
        scope["initialize"](native, "127.0.0.1", "")
        assert (
            native.protocol
            == contract.local_protocol(native.engine, "127.0.0.1")
            == "tcp"
        )
        assert native.d2d_protocol_verified and not native.d2d_rdma_eligible
        cases.append(
            dict(
                MC_FORCE_TCP=value, requested="rdma", actual=native.protocol, gpu=False
            )
        )

    # Even an explicit force-HCA setting cannot steal the dedicated TCP engine.
    with patch.dict(os.environ, {"MC_FORCE_HCA": "1"}):
        fallback = mb.tcp_engine()
        assert (
            fallback.protocol
            == contract.local_protocol(fallback.raw, "127.0.0.1")
            == "tcp"
        )
        assert "MC_FORCE_TCP" not in os.environ
        assert mb.tcp_engine() is fallback
        assert contract.local_protocol(retained[0].engine, "127.0.0.1") == "tcp"
        cases.append(
            dict(tcp_fallback=True, force_hca=True, verified=True, shared=True)
        )

    # Real single-submit/status API plus receiver-arrival proof for CPU buffers.
    source = torch.arange(4096, dtype=torch.int32).pin_memory()
    target = torch.full_like(source, -1, pin_memory=True)
    peer = retained[0].engine
    size = source.numel() * source.element_size()
    assert fallback.batch_register([source.data_ptr()], [size]) == 0
    assert peer.batch_register_memory([target.data_ptr()], [size]) == 0
    assert (
        fallback.batch_transfer_sync(
            f"127.0.0.1:{peer.get_rpc_port()}",
            [source.data_ptr()],
            [target.data_ptr()],
            [size],
        )
        == 0
    )
    deadline = time.monotonic() + 5
    while not torch.equal(source, target):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert fallback.batch_deregister([source.data_ptr()]) == 0
    assert peer.batch_unregister_memory([target.data_ptr()]) == 0
    cases.append(dict(single_submit_native_tcp=True, bytes=size, payload_equal=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                status="PASS",
                device=torch.cuda.get_device_name(),
                real_rdma=False,
                cases=cases,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(status="PASS", cases=len(cases), real_rdma=False)))


if __name__ == "__main__":
    main()
