# Algorithm references

This package contains an original suffix-automaton implementation and does not
vendor or link Baseten sa_spec or TensorRT-LLM code. The default GPU SA path
requires no third-party checkout. Optional `backend="arctic"` imports the official
ArcticInference package and its native CPU suffix extension; it is not vendored.

The design's pinned references remain in [DESIGN.md](../DESIGN.md#references):

- Baseten sa_spec: `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969`.
- TensorRT-LLM SA/MTP integration: `e99d74f1da60be174a4e34b412c4891460b9ff70`.
- ArcticInference: `aca5d9a8a62474035c15d114d40a01abc8c94b51` (original design reference).

These are design provenance, not dependencies or claims of matching their
performance, implementation details, or sampling behavior. Before vendoring any
future backend, preserve its actual license and pinned source separately.

The optional Arctic adapter was tested with upstream
`9d4643881608178f0f58a33d1b61919ca246997f` (Apache-2.0), building its suffix-only
C++ extension unchanged. See [installation and validation](../validation/arctic-suffix.md).
