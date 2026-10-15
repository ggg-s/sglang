# Algorithm references

This package contains an original suffix-automaton implementation and does not
vendor or link Baseten sa_spec, TensorRT-LLM or ArcticInference code. No third-party
checkout is required at runtime.

The design's pinned references remain in [DESIGN.md](../DESIGN.md#references):

- Baseten sa_spec: `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969`.
- TensorRT-LLM SA/MTP integration: `e99d74f1da60be174a4e34b412c4891460b9ff70`.
- ArcticInference: `aca5d9a8a62474035c15d114d40a01abc8c94b51` (future optional backend).

These are design provenance, not dependencies or claims of matching their
performance, implementation details, or sampling behavior. Before vendoring any
future backend, preserve its actual license and pinned source separately.
