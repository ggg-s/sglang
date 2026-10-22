"""ReplaySSM verify tile sweep with FP32 checkpoints and compact high/low history.

Use only on an idle GPU. Timings use CUDA Graph replay; repeated calls overwrite
the same speculative window without advancing cursors. This is a one-layer
microbenchmark, not serving throughput. Explicit knobs do not change production
defaults. Each candidate is compared with the original launch configuration.
"""

import argparse
import json

import torch
import triton.testing

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)


def inputs(batch, width, history, dtype):
    h, hv, dim, ring, slots = 16, 32, 128, 64, batch + 1
    device = "cuda"
    state = torch.randn(slots, hv, dim, dim, device=device) * 0.05
    keys = torch.randn(slots, h, ring, dim, device=device)
    keys = keys / (keys.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    updates = torch.randn(slots, hv, ring, dim, device=device) * 0.05
    khi, dhi = keys.to(dtype), updates.to(dtype)
    return dict(
        q=torch.randn(batch * width, h, dim, device=device, dtype=dtype),
        k=torch.randn(batch * width, h, dim, device=device, dtype=dtype),
        v=torch.randn(batch * width, hv, dim, device=device, dtype=dtype),
        a=torch.randn(batch * width, hv, device=device, dtype=dtype) - 2,
        b=torch.randn(batch * width, hv, device=device, dtype=dtype),
        A_log=torch.zeros(hv, device=device),
        dt_bias=torch.zeros(hv, device=device),
        checkpoint_state=state,
        d_cache=dhi,
        k_cache=khi,
        g_cache=-torch.rand(slots, hv, ring, device=device) * 0.1,
        rawv_cache=(updates - dhi.float()).to(dtype),
        rawk_cache=(keys - khi.float()).to(dtype),
        beta_cache=None,
        out=torch.empty(batch * width, hv, dim, device=device, dtype=dtype),
        query_start_loc=torch.arange(batch + 1, device=device, dtype=torch.int32)
        * width,
        ssm_state_indices=torch.arange(1, slots, device=device, dtype=torch.int32),
        replay_indices=torch.arange(1, slots, device=device, dtype=torch.int32),
        write_pos=torch.full((slots,), history, device=device, dtype=torch.int32),
        cache_base=torch.full((slots,), 7, device=device, dtype=torch.int32),
        is_flush=torch.zeros(slots, device=device, dtype=torch.int8),
        max_cache_len=ring,
        max_spec_len=17,
        null_block_id=-1,
        launch_mode="verify",
    )


def main():
    configs = [
        (64, 1, 2, 2),
        (64, 2, 2, 2),
        (64, 4, 2, 2),
        (64, 4, 1, 2),
        (32, 1, 2, 2),
        (32, 2, 2, 2),
        (32, 4, 2, 2),
        (32, 4, 1, 2),
        (64, 4, 2, 1),
        (32, 4, 2, 1),
        (64, 4, 2, 4),
        (32, 4, 2, 4),
    ]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--widths", nargs="+", type=int, default=[9, 17])
    parser.add_argument("--histories", nargs="+", type=int, default=[0, 16, 30])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    torch.manual_seed(20261022)
    dtype = getattr(torch, args.dtype)
    results = []
    for batch in args.batch_sizes:
        for width in args.widths:
            for history in args.histories:
                kw = inputs(batch, width, history, dtype)
                gdn_replayssm_spec_decode(
                    **kw, block_v=64, num_warps=1, num_stages=2, nk=2
                )
                expected = kw["out"].clone()
                baseline_d = kw["d_cache"].clone()
                baseline_lo = kw["rawv_cache"].clone()
                for bound, (bv, warps, stages, nk) in [
                    (bound, cfg) for bound in sorted({17, width}) for cfg in configs
                ]:
                    kw["max_spec_len"] = bound

                    def run():
                        gdn_replayssm_spec_decode(
                            **kw, block_v=bv, num_warps=warps, num_stages=stages, nk=nk
                        )

                    try:
                        run()
                        torch.testing.assert_close(
                            kw["out"], expected, rtol=0.02, atol=0.003
                        )
                        torch.testing.assert_close(
                            kw["d_cache"].float() + kw["rawv_cache"].float(),
                            baseline_d.float() + baseline_lo.float(),
                            rtol=0.002,
                            atol=0.0002,
                        )
                        ms = triton.testing.do_bench_cudagraph(run, rep=100)
                        row = dict(
                            batch=batch,
                            width=width,
                            bound=bound,
                            history=history,
                            block_v=bv,
                            warps=warps,
                            stages=stages,
                            nk=nk,
                            us=ms * 1000,
                        )
                    except Exception as error:
                        row = dict(
                            batch=batch,
                            width=width,
                            bound=bound,
                            history=history,
                            block_v=bv,
                            warps=warps,
                            stages=stages,
                            nk=nk,
                            error=str(error)[:1000],
                        )
                    results.append(row)
                    print(json.dumps(row), flush=True)
    with open(args.output, "w") as stream:
        json.dump(
            dict(
                gpu=torch.cuda.get_device_name(),
                dtype=args.dtype,
                state_dtype="float32",
                results=results,
            ),
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
