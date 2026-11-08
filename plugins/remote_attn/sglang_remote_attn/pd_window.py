"""Bounded, D-clock diagnostics for hybrid PD; never a readiness authority."""

import time
from collections import deque

from .pre_layerwise import LayerCoverage


class WindowTrace:
    def __init__(self, layer_entries):
        self.layer_entries = {int(k): tuple(v) for k, v in layer_entries.items()}
        self.first_layer = min(self.layer_entries)
        self.started = time.monotonic()
        self.entries = {i: {} for ids in self.layer_entries.values() for i in ids}
        self.receipts = None
        self.layer_counts = {
            layer: dict(remote_calls=0, remote_after_receipt=0, remote_after_visible=0)
            for layer in self.layer_entries
        }
        self.steps = deque(maxlen=256)
        self.step_count = self.remote_steps = self.remote_layer_calls = 0

    def elapsed(self):
        return (time.monotonic() - self.started) * 1000

    def received(self, piece, header):
        # A validated P completion receipt observed on D, not a P timestamp
        # and not proof that a subsequent CUDA consumer can read these bytes.
        if self.receipts is None:
            self.receipts = LayerCoverage(header["length"], header["resources"])
            if header["base"]:
                for entry in self.entries:
                    self.receipts.complete(entry, 0, header["base"])
        for entry in piece["resources"]:
            self.receipts.complete(
                entry,
                header["base"] + piece["lo"] * header["page"],
                min(header["length"], header["base"] + piece["hi"] * header["page"]),
            )
            if self.receipts.ready(entry):
                self.entries[entry].setdefault("receipt_ms", self.elapsed())

    def visible(self, coverage):
        for entry, record in self.entries.items():
            if "visible_ms" not in record and coverage.ready(entry):
                record["visible_ms"] = self.elapsed()

    def local(self, entry):
        self.entries[entry].setdefault("local_ms", self.elapsed())

    def layer(self, layer, remote):
        # Only ordinary one-token decode is admitted by the hybrid adapter.
        # Start at the first full-attention layer; FB objects may be reused.
        if layer == self.first_layer:
            self.step_count += 1
            self.steps.append(
                dict(
                    step=self.step_count,
                    at_ms=self.elapsed(),
                    remote_layers=[],
                    layers_seen=[],
                )
            )
        if not self.steps:
            return
        step = self.steps[-1]
        step["layers_seen"].append(layer)
        if remote:
            if not step["remote_layers"]:
                self.remote_steps += 1
            step["remote_layers"].append(layer)
            self.remote_layer_calls += 1
            counts = self.layer_counts[layer]
            counts["remote_calls"] += 1
            for field, count in (
                ("receipt_ms", "remote_after_receipt"),
                ("visible_ms", "remote_after_visible"),
            ):
                if all(field in self.entries[i] for i in self.layer_entries[layer]):
                    counts[count] += 1

    def snapshot(self):
        layers = {}
        for layer, entries in self.layer_entries.items():
            record = dict(self.layer_counts[layer])
            for field in ("receipt_ms", "visible_ms", "local_ms"):
                values = [self.entries[i].get(field) for i in entries]
                record[field] = (
                    max(values) if all(v is not None for v in values) else None
                )
            for name, start, end in (
                ("receipt_to_visible_ms", "receipt_ms", "visible_ms"),
                ("visible_to_local_ms", "visible_ms", "local_ms"),
            ):
                record[name] = (
                    record[end] - record[start]
                    if record[start] is not None and record[end] is not None
                    else None
                )
            layers[str(layer)] = record
        return dict(
            clock="D monotonic; ms since early admission",
            layers=layers,
            observed_steps=self.step_count,
            remote_steps=self.remote_steps,
            remote_layer_calls=self.remote_layer_calls,
            recent_steps=list(self.steps),
            dropped_steps=max(0, self.step_count - len(self.steps)),
        )
