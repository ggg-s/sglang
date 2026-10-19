"""Keep test-created Triton modules from being GC-unloaded during capture.

The synthetic model fixture replaces/restores modules on every parameterized
case. Unlike serving, many obsolete CompiledKernel cycles can accumulate and
be collected inside a later graph capture. Collect between cases and defer
automatic cycle collection while a case runs; operators/model logic stay intact.
"""

import gc

import pytest


@pytest.fixture(autouse=True)
def defer_synthetic_kernel_gc(request):
    if request.node.originalname != (
        "test_compact_tail_graph_reorders_rows_without_computing_sa_requests"
    ):
        yield
        return
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()
