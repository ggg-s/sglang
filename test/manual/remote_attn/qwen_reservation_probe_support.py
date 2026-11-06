"""Source-loaded native Qwen paged allocator/PD prealloc with small scheduler ABI."""

import ast
from array import array
import dataclasses
import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang_remote_attn import handoff_control as rt


def load(root, relative, name, members=None, extra=None):
    path = Path(root) / "python/sglang/srt" / relative
    node = next(
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name
    )
    if isinstance(node, ast.ClassDef):
        node.bases, node.keywords = [], []
        if members is not None:
            node.body = [n for n in node.body if getattr(n, "name", None) in members]
    ns = dict(torch=torch, dataclasses=dataclasses, __name__=__name__)
    ns.update(extra or {})
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns[name]


class Sampling:
    def __init__(self, **kw):
        self.max_new_tokens = 64
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 1
        self.ignore_eos = False
        self.__dict__.update(kw)

    def normalize(self, tokenizer):
        pass


class Fixture:
    def __init__(self, root, device="cpu", page=4, capacity=64):
        self.stack = ExitStack()
        self.page = page
        self.root = Path(root)
        ceil = lambda a, b: (a + b - 1) // b * b
        num = load(root, "utils/common.py", "get_num_new_pages")
        kv = load(root, "managers/schedule_batch.py", "ReqKvInfo")
        Pool = load(
            root,
            "mem_cache/memory_pool.py",
            "ReqToTokenPool",
            {"alloc", "alloc_rows", "free", "free_rows", "available_size", "write"},
        )
        pool = Pool()
        pool.free_slots = list(range(1, 4))
        pool.req_generation = torch.zeros(4, dtype=torch.int64)
        pool.req_to_token = torch.full((4, 128), -1, dtype=torch.int32, device=device)
        pool._on_alloc_rows = pool._aux_cache = None
        naive = load(root, "mem_cache/allocator/paged.py", "alloc_extend_naive")
        methods = {
            "clear",
            "available_size",
            "get_all_free_pages",
            "merge_and_sort_free",
            "alloc_extend",
            "alloc",
            "free",
            "_release_page_ids",
            "_debug_check_no_duplicate_pages",
        }
        extra = dict(
            alloc_extend_naive=naive,
            get_num_new_pages=num,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        )
        if device == "cuda":
            path = self.root / "python/sglang/kernels/ops/memory/allocator.py"
            spec = importlib.util.spec_from_file_location(
                "range_native_allocator_kernels", path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            extra["alloc_extend_kernel"] = module.alloc_extend_kernel
        Alloc = load(
            root,
            "mem_cache/allocator/paged.py",
            "PagedTokenToKVPoolAllocator",
            methods,
            extra,
        )
        allocator = Alloc()
        allocator.size = capacity
        allocator.num_pages = capacity // page
        allocator.page_size = page
        allocator.device = device
        allocator.need_sort = False
        allocator.debug_mode = True
        allocator.use_triton_kernels = device == "cuda"
        allocator.clear()
        allocator.prealloc_fits = load(
            root,
            "mem_cache/allocator/base.py",
            "BaseTokenToKVPoolAllocator",
            {"prealloc_fits"},
        ).prealloc_fits.__get__(allocator)
        key = [
            torch.zeros((capacity + page, 1, 8), device=device, dtype=torch.float16)
            for _ in range(3)
        ]
        cache = NS(
            layer_num=3,
            get_key_buffer=key.__getitem__,
            get_value_buffer=key.__getitem__,
        )
        allocator.get_kvcache = lambda: cache
        tree = NS(
            disable=True,
            claim_kv_row=lambda req: False,
            maybe_hand_to_session=lambda req: None,
            unlock=lambda lock: None,
            on_release=lambda *a, **kw: None,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            supports_mamba=lambda: False,
        )

        def free_row(record, ranges):
            for lo, hi in ranges:
                allocator.free(pool.req_to_token[record.req_pool_idx, lo:hi])

        tree.free_kv_row = free_row
        helper = dict(
            get_disagg=lambda: NS(disaggregation_decode_enable_radix_cache=False),
            get_num_new_pages=num,
            ceil_align=ceil,
            get_memory=lambda: NS(disable_radix_cache=True),
            async_h2d=lambda values, **kw: torch.tensor(values, **kw),
        )
        alloc = load(
            root, "disaggregation/decode.py", "alloc_for_decode_prealloc", extra=helper
        )
        helper["alloc_for_decode_prealloc"] = alloc
        Queue = load(
            root,
            "disaggregation/decode.py",
            "DecodePreallocQueue",
            {
                "_pre_alloc",
                "_pre_alloc_fill_len",
                "_prealloc_kv_lens",
                "_prealloc_required_tokens",
                "_required_alloc_tokens",
                "_alloc_for_decode_prealloc",
                "_prealloc_reservation_fits",
            },
            helper,
        )
        queue = Queue()
        scheduler = NS(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree,
            tokenizer=None,
            enable_hisparse=False,
            disagg_decode_prealloc_queue=queue,
            handoff_tracked_reqs={},
            waiting_queue=[],
        )
        queue.scheduler = scheduler
        queue.tree_cache = tree
        queue.req_to_token_pool = pool
        queue.token_to_kv_pool_allocator = allocator
        queue.num_reserved_decode_tokens = 0
        queue._uses_swa_tail_prealloc = lambda: False
        queue._uses_swa_reservation = lambda: False
        queue._swa_tail_len = lambda n: n
        queue._allocatable_token_budgets = lambda **kw: allocator.available_size()
        queue._radix_full_available = allocator.available_size

        def new(s, rid, origin, output, sp, **kw):
            req = NS(
                rid=rid,
                origin_input_ids=array("q", origin),
                output_ids=array("q", output),
                sampling_params=sp,
                kv=kv(),
                lock=None,
                skip_radix_cache_insert=False,
                finished_len=None,
                finished=lambda: False,
            )
            req.owned_kv_len = lambda: req.kv.kv_allocated_len
            return req

        self.new = new
        self.scheduler = scheduler
        self.allocator = allocator
        self.pool = pool
        self.queue = queue
        self.parallel = NS(tp_size=1, tp_rank=0, tp_group=NS(cpu_group=None))
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": NS(
                        get_parallel=lambda: self.parallel,
                        get_schedule=lambda: NS(max_running_requests=3),
                    ),
                    "sglang.srt.sampling.sampling_params": NS(SamplingParams=Sampling),
                },
            )
        )
        self.stack.enter_context(patch.object(rt, "_new_req", side_effect=new))
        over = load(
            root,
            "mem_cache/common.py",
            "_release_overallocated_kv_indices",
            extra=dict(
                ceil_align=ceil,
                get_spec=lambda: NS(speculative_algorithm=None),
                get_serving=lambda: NS(strip_thinking_cache=False),
            ),
        )
        release = load(
            root,
            "mem_cache/common.py",
            "release_kv_cache",
            extra=dict(_release_overallocated_kv_indices=over),
        )
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {"sglang.srt.mem_cache.common": NS(release_kv_cache=release)},
            )
        )

    def close(self):
        self.stack.close()
