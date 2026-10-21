"""Pair SA layouts by phase; time inside CUDA Graph and around Python replay.

No production files are modified. A private JIT binding exposes the existing
finalize_shared_kernel with final_lengths=-1 as the update-only phase. Query
uses consume_and_query with empty pending feedback on the post-update state.
Both include their production validation/housekeeping. Separate launches and
shared checks mean their times must not be added to predict the fused kernel.
"""

import argparse
from collections import Counter
import ctypes
import gzip
import hashlib
import json
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
from hybrid_spec.backends.sa_layout import unpack_states
from torch.utils.cpp_extension import load

from benchmark_sa_history_cuda import sequence

UPDATE_BINDING = r"""
void benchmark_update(torch::Tensor history, torch::Tensor states,
    torch::Tensor edges, torch::Tensor meta, torch::Tensor owners,
    torch::Tensor pending, torch::Tensor pending_meta, torch::Tensor slots,
    torch::Tensor generations, torch::Tensor lengths) {
  check_storage(history, states, edges, meta);
  check_tensor(lengths, torch::kInt64, history.device());
  c10::cuda::CUDAGuard guard(history.device());
  dispatch_states(states, [&](auto type) {
    using SA = decltype(type);
    finalize_shared_kernel<SA><<<slots.numel(), 1, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        history.data_ptr<int>(), states.data_ptr<typename SA::StateWord>(),
        edges.data_ptr<int>(), meta.data_ptr<int>(), owners.data_ptr<int64_t>(),
        pending.data_ptr<int>(), pending_meta.data_ptr<int64_t>(),
        slots.data_ptr<int64_t>(), generations.data_ptr<int64_t>(),
        lengths.data_ptr<int64_t>(), history.size(0), history.size(1),
        pending.size(1) - 1);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_only", &benchmark_update);
  m.def("consume_and_query", &consume_and_query);
}
"""


def native_module(source, output):
    files = ("automaton.h", "host_builder.h", "host_builder.cpp", "suffix_automaton.cu")
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files
    }
    # Preserve the exact production kernel definitions, replacing only bindings.
    original = (source / "suffix_automaton.cu").read_text()
    marker = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {"
    if original.count(marker) != 1:
        raise RuntimeError("unrecognised production binding boundary")
    generated = original.split(marker)[0] + UPDATE_BINDING
    build = output / "jit-source"
    build.mkdir(parents=True, exist_ok=True)
    for name in files[:-1]:
        target = build / name
        if not target.exists() or target.read_bytes() != (source / name).read_bytes():
            shutil.copyfile(source / name, target)
    target = build / "phase_probe.cu"
    if not target.exists() or target.read_text() != generated:
        target.write_text(generated)
    digest = hashlib.sha256(generated.encode()).hexdigest()
    module = load(
        name=f"hybrid_sa_phases_{digest[:12]}",
        sources=[str(build / "phase_probe.cu"), str(build / "host_builder.cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )
    return module, hashes | {"generated_probe": digest}


def describe(values):
    values = sorted(values)

    def percentile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(0.1),
        "p90_us": percentile(0.9),
        "max_us": max(values),
        "min_us": min(values),
    }


class Telemetry:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.rows = []

    def sample(self):
        # Timestamp the host query as well as nvidia-smi's lower-rate sensors.
        while not self.stop.is_set():
            before = time.time_ns()
            p = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            row = {
                "start_ns": before,
                "end_ns": time.time_ns(),
                "csv": p.stdout.strip(),
                "exit_code": p.returncode,
            }
            self.rows.append(row)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)


class Arm:
    def __init__(self, module, capacity, alphabet, bs, compact):
        self.module, self.capacity, self.bs = module, capacity, bs
        self.k, self.tail = 8, 9
        self.length = capacity - 18
        packed = compact and capacity <= 32767
        self.layout = "72" if packed else "108"
        fields, dtype = (11, torch.uint16) if packed else (10, torch.int32)
        data = sequence(self.length + self.tail, alphabet)
        self.history = torch.empty((bs, capacity), dtype=torch.int32, device="cuda")
        self.states = torch.empty(
            (bs, 2 * capacity + 1, fields), dtype=dtype, device="cuda"
        )
        self.edges = torch.empty(
            (bs, capacity + 1, 6), dtype=torch.int32, device="cuda"
        )
        self.meta = torch.empty((bs, 7), dtype=torch.int32, device="cuda")
        prompt = torch.tensor(data[: self.length], dtype=torch.int64)
        host = [
            torch.empty(self.length, dtype=torch.int32),
            torch.empty((2 * self.length + 1, fields), dtype=dtype),
            torch.empty((self.length + 1, 6), dtype=torch.int32),
            torch.empty(7, dtype=torch.int32),
        ]
        library = ctypes.CDLL(module.__file__)
        build = (
            library.hybrid_sa_build_host_compact
            if packed
            else library.hybrid_sa_build_host
        )
        build.argtypes = (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            + [ctypes.c_void_p] * 4
            + [ctypes.c_int]
        )
        build.restype = ctypes.c_int
        if build(
            prompt.data_ptr(),
            self.length,
            alphabet,
            *(t.data_ptr() for t in host),
            self.length,
        ):
            raise RuntimeError("host construction failed")
        used = [self.length, host[3][1].item(), host[3][2].item()]
        self.pre = [t[:n].cuda() for t, n in zip(host[:3], used)] + [host[3].cuda()]
        self.slots = torch.arange(bs, device="cuda")
        self.owners = torch.ones(bs, device="cuda", dtype=torch.int64)
        self.limits = torch.full_like(self.owners, capacity)
        self.final_lengths = torch.full_like(self.owners, -1)
        self.seq_lens = torch.full_like(self.owners, self.length + self.tail - 1)
        self.pending = torch.tensor(
            data[self.length :], dtype=torch.int32, device="cuda"
        ).repeat(bs, 1)
        self.feedback = torch.tensor(
            [1, self.length, self.tail], device="cuda", dtype=torch.int64
        ).repeat(bs, 1)
        self.pending_meta = torch.empty_like(self.feedback)
        self.tokens = torch.empty((bs, self.k), device="cuda", dtype=torch.int64)
        self.mask = torch.empty(bs, device="cuda", dtype=torch.bool)
        self.restore("combined")
        self.run("combined")
        torch.cuda.synchronize()
        nstates, nedges = self.meta[0, 1:3].tolist()
        self.post = [
            self.history[0, : self.length + self.tail].clone(),
            self.states[0, :nstates].clone(),
            self.edges[0, :nedges].clone(),
            self.meta[0].clone(),
        ]
        self.expected_tokens = self.tokens.cpu().tolist()
        self.expected_mask = self.mask.cpu().tolist()
        self.graphs = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for phase in ("update", "query", "query_min1", "combined"):
                self.restore(phase)
                self.run(phase)
                self.restore(phase)
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    start.record()
                    self.run(phase)
                    end.record()
                self.graphs[phase] = graph, start, end
        torch.cuda.current_stream().wait_stream(stream)
        self.outer_start, self.outer_end = (
            torch.cuda.Event(enable_timing=True) for _ in range(2)
        )

    def restore(self, phase):
        frozen = self.post if phase.startswith("query") else self.pre
        for target, source in zip((self.history, self.states, self.edges), frozen[:3]):
            target[:, : source.shape[0]].copy_(
                source[None].expand(self.bs, *source.shape)
            )
        self.meta.copy_(frozen[3].expand(self.bs, -1))
        if phase.startswith("query"):
            self.pending_meta.zero_()
        else:
            self.pending_meta.copy_(self.feedback)

    def run(self, phase):
        if phase == "update":
            self.module.update_only(
                self.history,
                self.states,
                self.edges,
                self.meta,
                self.owners,
                self.pending,
                self.pending_meta,
                self.slots,
                self.owners,
                self.final_lengths,
            )
        else:
            self.module.consume_and_query(
                self.history,
                self.states,
                self.edges,
                self.meta,
                self.owners,
                self.limits,
                self.pending,
                self.pending_meta,
                self.slots,
                self.owners,
                self.seq_lens,
                self.tokens,
                self.mask,
                1 if phase == "query_min1" else 4,
                True,
                None,
                None,
                None,
                128,
                None,
                False,
                True,
            )

    def sample(self, phase, *, inject_delay=0):
        self.restore(phase)
        graph, start, end = self.graphs[phase]
        before = time.time_ns()
        self.outer_start.record()
        if inject_delay:
            self.outer_start.synchronize()
            time.sleep(inject_delay)
        t0 = time.perf_counter_ns()
        graph.replay()
        t1 = time.perf_counter_ns()
        self.outer_end.record()
        self.outer_end.synchronize()
        return {
            "wall_start_ns": before,
            "wall_end_ns": time.time_ns(),
            "inner_us": start.elapsed_time(end) * 1000,
            "outer_us": self.outer_start.elapsed_time(self.outer_end) * 1000,
            "host_replay_us": (t1 - t0) / 1000,
        }

    def validate(self):
        phase_outputs = {}
        for phase in self.graphs:
            self.sample(phase)
            ns, ne = self.post[3][1:3].tolist()
            assert torch.equal(self.meta, self.post[3][None].expand_as(self.meta))
            for actual, expected in zip(
                (self.history, self.states, self.edges), self.post[:3]
            ):
                assert torch.equal(
                    actual[:, : expected.shape[0]],
                    expected[None].expand(self.bs, *expected.shape),
                )
            assert not self.pending_meta[:, 2].any().item()
            if phase in ("query", "combined"):
                assert self.tokens.tolist() == self.expected_tokens
                assert self.mask.tolist() == self.expected_mask
            if phase != "update":
                phase_outputs[phase] = {
                    "tokens": self.tokens.tolist(),
                    "eligible": self.mask.tolist(),
                }
        semantic = unpack_states(self.post[1].cpu())
        return {
            "state_sha256": hashlib.sha256(semantic.numpy().tobytes()).hexdigest(),
            "edges_sha256": hashlib.sha256(
                self.post[2].cpu().numpy().tobytes()
            ).hexdigest(),
            "history_sha256": hashlib.sha256(
                self.post[0].cpu().numpy().tobytes()
            ).hexdigest(),
            "tokens": self.expected_tokens,
            "eligible": self.expected_mask,
            "num_states": ns,
            "num_edges": ne,
            "phase_outputs": phase_outputs,
        }


def profile_arms(arms, output, rounds, samples, warmup):
    """Read actual kernel durations from CUPTI, excluding event-node gaps."""
    from torch.profiler import ProfilerActivity, profile, record_function

    for phase in ("update", "query", "query_min1", "combined"):
        for arm in arms.values():
            for _ in range(warmup):
                arm.sample(phase)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for phase in ("update", "query", "query_min1", "combined"):
            for round_index in range(rounds):
                names = (
                    ("wide", "compact") if round_index % 2 == 0 else ("compact", "wide")
                )
                for name in names:
                    arm = arms[name]
                    with record_function(f"SA_PHASE|{phase}|{round_index}|{name}"):
                        for _ in range(samples):
                            arm.restore(phase)
                            arm.graphs[phase][0].replay()
                        torch.cuda.synchronize()
    path = output.with_suffix(".json")
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    spans = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name", "").startswith("SA_PHASE|")
    ]
    kernels = [
        e
        for e in events
        if e.get("cat") == "kernel"
        and any(
            k in e.get("name", "")
            for k in ("consume_query_kernel", "finalize_shared_kernel")
        )
    ]
    # CPU/GPU timestamp conversion can put a kernel end ~1 us past its CPU
    # annotation. Attribute by the captured graph ID, with exact total counts.
    graph_labels = {}
    for span in spans:
        _, phase, _, name = span["name"].split("|")
        selected = [
            e for e in kernels if span["ts"] <= e["ts"] < span["ts"] + span["dur"]
        ]
        counts = Counter(e["args"]["graph id"] for e in selected)
        if not counts or counts.most_common(1)[0][1] < samples - 2:
            raise RuntimeError(f"Cannot attribute {span['name']} to a graph")
        graph_id = counts.most_common(1)[0][0]
        label = (phase, name)
        if label in graph_labels and graph_labels[label] != graph_id:
            raise RuntimeError("Graph ID changed within a phase/layout")
        graph_labels[label] = graph_id
    if len(set(graph_labels.values())) != 8:
        raise RuntimeError("Expected eight distinct phase/layout CUDA graphs")
    records = []
    for (phase, name), graph_id in graph_labels.items():
        selected = sorted(
            (e for e in kernels if e["args"]["graph id"] == graph_id),
            key=lambda e: e["ts"],
        )
        if len(selected) != rounds * samples:
            raise RuntimeError(f"Missing CUDA kernels for {phase}/{name}")
        for round_index in range(rounds):
            chunk = selected[round_index * samples : (round_index + 1) * samples]
            durations = [e["dur"] for e in chunk]
            records.append(
                {
                    "phase": phase,
                    "round": round_index,
                    "layout": name,
                    "graph_id": graph_id,
                    "stats": describe(durations),
                    "kernel_samples_us": durations,
                    "kernel_name": chunk[0]["name"],
                }
            )
    if len(records) != rounds * 4 * 2:
        raise RuntimeError("Profiler did not capture every phase/layout")
    with gzip.open(str(path) + ".gz", "wb") as f:
        f.write(path.read_bytes())
    path.unlink()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/hybrid_spec/backends/gpu_sa",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--jit-output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--anomaly-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module, hashes = native_module(args.source, args.jit_output or args.output)
    report = {
        "scope": __doc__,
        "source_sha256": hashes,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "samples_per_round": args.samples,
        "rounds": args.rounds,
        "results": [],
        "delay_controls": [],
        "mode": "cupti_kernel" if args.profile else "cuda_events",
    }
    cases = (
        [(1024, a, b) for a in (1, 2, 64) for b in (1, 4)]
        + [(16384, a, b) for a in (1, 2, 64) for b in (1, 4)]
        + [(32768, 2, 4)]
    )
    if args.anomaly_only:
        cases = [(16384, 1, 4), (16384, 2, 4)]
    if args.smoke:
        cases = [(16384, 1, 4)]
    # Raise clocks with bounded GPU work, without changing device settings.
    for _ in range(3):
        torch.cuda._sleep(100000000)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    empty_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty_graph):
        start.record()
        end.record()
    event_costs = []
    for _ in range(50):
        empty_graph.replay()
        end.synchronize()
        event_costs.append(start.elapsed_time(end) * 1000)
    report["empty_graph_event_interval_us"] = describe(event_costs)
    with Telemetry(args.output / "telemetry.jsonl"):
        for capacity, alphabet, bs in cases:
            arms = {
                name: Arm(module, capacity, alphabet, bs, name == "compact")
                for name in ("wide", "compact")
            }
            valid = {name: arm.validate() for name, arm in arms.items()}
            assert valid["wide"] == valid["compact"]
            row = {
                "capacity": capacity,
                "alphabet": alphabet,
                "batch_size": bs,
                "semantics": valid["wide"],
                "rounds": [],
            }
            if args.profile:
                row["profile_rounds"] = profile_arms(
                    arms,
                    args.output / f"trace-{capacity}-{alphabet}-{bs}",
                    args.rounds,
                    args.samples,
                    args.warmup,
                )
                report["results"].append(row)
                (args.output / "phases.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            "capacity": capacity,
                            "alphabet": alphabet,
                            "batch": bs,
                            "profile_spans": len(row["profile_rounds"]),
                        }
                    ),
                    flush=True,
                )
                del arms
                continue
            for phase in ("update", "query", "query_min1", "combined"):
                for arm in arms.values():
                    for _ in range(args.warmup):
                        arm.sample(phase)
                for round_index in range(args.rounds):
                    samples = {name: [] for name in arms}
                    for i in range(args.samples):
                        # Sample-level ABBA reverses order; both allocations live.
                        names = (
                            ("wide", "compact")
                            if (i + round_index) % 2 == 0
                            else ("compact", "wide")
                        )
                        for name in names:
                            samples[name].append(arms[name].sample(phase))
                    stats = {
                        name: {
                            kind: describe([s[kind] for s in values])
                            for kind in ("inner_us", "outer_us", "host_replay_us")
                        }
                        for name, values in samples.items()
                    }
                    record = {
                        "phase": phase,
                        "round": round_index,
                        "stats": stats,
                        "samples": samples,
                    }
                    row["rounds"].append(record)
                    print(
                        json.dumps(
                            {
                                "capacity": capacity,
                                "alphabet": alphabet,
                                "batch": bs,
                                "phase": phase,
                                "round": round_index,
                                "wide_us": stats["wide"]["inner_us"]["median_us"],
                                "compact_us": stats["compact"]["inner_us"]["median_us"],
                                "wide_outer_us": stats["wide"]["outer_us"]["median_us"],
                                "compact_outer_us": stats["compact"]["outer_us"][
                                    "median_us"
                                ],
                            }
                        ),
                        flush=True,
                    )
            if (capacity, alphabet, bs) == (16384, 1, 4):
                for name, arm in arms.items():
                    for delay in (0, 0.001):
                        samples = [
                            arm.sample("combined", inject_delay=delay) for _ in range(5)
                        ]
                        report["delay_controls"].append(
                            {"layout": name, "delay_seconds": delay, "samples": samples}
                        )
            report["results"].append(row)
            (args.output / "phases.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            del arms
    print("All phase/layout semantic checks passed", flush=True)


if __name__ == "__main__":
    main()
