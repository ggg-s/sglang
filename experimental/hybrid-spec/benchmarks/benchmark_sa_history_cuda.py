"""A/B the actual captured consume-and-query kernel with long imported histories.

Builds both Git-baseline and working-tree CUDA extensions in one process. Each
sample restores the same prompt/feedback outside timing (warm-cache component
measurement). Includes hash splits and suffix query, excludes TP/model/feedback
production and H2D import. Requires NVIDIA CUDA, nvcc, torch and numpy.
"""

import argparse
import ctypes
import hashlib
import json
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def sequence(length, vocab):
    state = 17
    result = []
    for _ in range(length):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        result.append(state % vocab)
    return result


def measure(
    module,
    state_fields,
    edge_factor,
    edge_fields,
    meta_fields,
    length,
    vocab,
    batch_size,
    repeats,
):
    k, tail = 8, 9
    capacity = length + tail + k + 1
    data = sequence(length + tail, vocab)
    prompt = torch.tensor(data[:length], dtype=torch.int64)
    host = [
        torch.empty(length, dtype=torch.int32),
        torch.empty((2 * length + 1, state_fields), dtype=torch.int32),
        torch.empty((edge_factor * length + 1, edge_fields), dtype=torch.int32),
        torch.empty(meta_fields, dtype=torch.int32),
    ]
    library = ctypes.CDLL(module.__file__)
    build = library.hybrid_sa_build_host
    build.argtypes = (
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        + [ctypes.c_void_p] * 4
        + [ctypes.c_int]
    )
    build.restype = ctypes.c_int
    if build(prompt.data_ptr(), length, vocab, *(t.data_ptr() for t in host), length):
        raise RuntimeError("host prompt construction failed")
    used = [length, host[3][1].item(), host[3][2].item()]
    frozen = [tensor[:count].cuda() for tensor, count in zip(host[:3], used)]
    frozen.append(host[3].cuda())
    history = torch.empty((batch_size, capacity), dtype=torch.int32, device="cuda")
    states = torch.empty(
        (batch_size, 2 * capacity + 1, state_fields), dtype=torch.int32, device="cuda"
    )
    edges = torch.empty(
        (batch_size, edge_factor * capacity + 1, edge_fields),
        dtype=torch.int32,
        device="cuda",
    )
    meta = torch.empty((batch_size, meta_fields), dtype=torch.int32, device="cuda")
    owners = torch.ones(batch_size, dtype=torch.int64, device="cuda")
    limits = torch.full_like(owners, capacity)
    pending = torch.tensor(data[length:], dtype=torch.int32, device="cuda").repeat(
        batch_size, 1
    )
    feedback = torch.tensor([1, length, tail], dtype=torch.int64, device="cuda").repeat(
        batch_size, 1
    )
    pending_meta = torch.empty_like(feedback)
    slots = torch.arange(batch_size, device="cuda")
    seq_lens = torch.full_like(owners, length + tail - 1)
    tokens = torch.empty((batch_size, k), dtype=torch.int64, device="cuda")
    mask = torch.empty(batch_size, dtype=torch.bool, device="cuda")

    def restore():
        for target, source, count in zip((history, states, edges), frozen, used):
            target[:, :count].copy_(
                source.unsqueeze(0).expand(batch_size, *source.shape)
            )
        meta.copy_(frozen[3].expand(batch_size, -1))
        pending_meta.copy_(feedback)

    def consume():
        module.consume_and_query(
            history,
            states,
            edges,
            meta,
            owners,
            limits,
            pending,
            pending_meta,
            slots,
            owners,
            seq_lens,
            tokens,
            mask,
            4,
            True,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        restore()
        consume()
    torch.cuda.current_stream().wait_stream(stream)
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        consume()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    samples = []
    for sample in range(repeats + 5):
        restore()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if sample >= 5:
            samples.append(start.elapsed_time(end) * 1000)
    if (
        not (meta[:, 0] == length + tail).all().item()
        or not (meta[:, 4] == 1).all().item()
    ):
        raise RuntimeError("captured append did not commit healthy history")
    # All rows use identical inputs. Check their complete stored state, then
    # canonicalize row zero so different physical layouts can be compared.
    ns, ne = meta[0, 1:3].tolist()
    for tensor in (
        meta,
        history[:, : length + tail],
        states[:, :ns],
        edges[:, :ne],
        tokens,
        mask,
    ):
        if not torch.equal(tensor, tensor[:1].expand_as(tensor)):
            raise RuntimeError("inconsistent CUDA batch rows")
    cpu_history = history[0, : length + tail].cpu().tolist()
    cpu_states = states[0, :ns].cpu().tolist()
    cpu_edges = edges[0, :ne, :3].cpu().tolist()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [
                meta[0, [0, 1, 3, 4]].tolist(),
                cpu_history,
                tokens[0].tolist(),
                mask[0].item(),
            ]
        ).encode()
    )
    for state in cpu_states:
        outgoing = []
        if state_fields == 5:
            if state[3] != -1:
                outgoing.append((cpu_history[state[2] + 1], state[3]))
            head = state[4]
            if head < -1:
                target = ~head
                outgoing.append((cpu_history[cpu_states[target][2]], target))
                head = -1
        else:
            head = state[3]
        while head != -1:
            token, target, head = cpu_edges[head]
            outgoing.append((token, target))
        digest.update(json.dumps([state[:3], sorted(outgoing)]).encode())
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "semantic_checksum": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="88f8d971eb")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 65536, 262144])
    parser.add_argument("--vocabs", nargs="+", type=int, default=[1, 2, 64, 8192])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 32])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip:
        parser.error("requires NVIDIA CUDA and nvcc")
    if min(*args.lengths, *args.vocabs, *args.batch_sizes, args.repeats) < 1:
        parser.error("all sizes and repeats must be positive")
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    relative = "experimental/hybrid-spec/src/hybrid_spec/backends/gpu_sa"
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "baseline_ref": args.baseline_ref,
        "scope": __doc__,
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="hybrid-sa-cuda-") as directory:
        baseline = Path(directory)
        for filename in (
            "automaton.h",
            "host_builder.h",
            "host_builder.cpp",
            "suffix_automaton.cu",
        ):
            content = subprocess.check_output(
                ["git", "show", f"{args.baseline_ref}:{relative}/{filename}"], cwd=repo
            )
            (baseline / filename).write_bytes(content)
        modules = {}
        for name, source in (("before", baseline), ("after", repo / relative)):
            content = (source / "automaton.h").read_text()
            fields = [
                re.search(rf"{field}_FIELDS = (\d+)", content)
                for field in ("EDGE", "META")
            ]
            edge_fields, meta_fields = [
                int(match[1]) if match else fallback
                for match, fallback in zip(fields, (3, 5))
            ]
            module = load(
                name=f"hybrid_sa_history_{name}",
                sources=[
                    str(source / "suffix_automaton.cu"),
                    str(source / "host_builder.cpp"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
            )
            compact = "STATE_FIELDS" in content
            modules[name] = (
                module,
                5 if compact else 4,
                1 if compact else 3,
                edge_fields,
                meta_fields,
            )
        for length in args.lengths:
            for vocab in args.vocabs:
                for batch_size in args.batch_sizes:
                    row = {
                        "history_tokens": length,
                        "vocab": vocab,
                        "batch_size": batch_size,
                    }
                    for name, native in modules.items():
                        row[name] = measure(
                            *native, length, vocab, batch_size, args.repeats
                        )
                    if (
                        row["before"]["semantic_checksum"]
                        != row["after"]["semantic_checksum"]
                    ):
                        raise RuntimeError(f"CUDA semantic state mismatch: {row}")
                    row["speedup"] = (
                        row["before"]["median_us"] / row["after"]["median_us"]
                    )
                    report["results"].append(row)
                    print(json.dumps(row), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )


if __name__ == "__main__":
    main()
