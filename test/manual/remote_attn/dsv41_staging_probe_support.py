"""Native low-ratio pool allocation/registration with lightweight KV boundaries."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


def allocated_pool(root, *, device="cpu", full_size=2048, page_size=256):
    path = Path(root) / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
    nodes = ast.parse(path.read_text()).body
    namespace = {
        "torch": torch,
        "nullcontext": nullcontext,
        "ceil_div": lambda a, b: (a + b - 1) // b,
        "GPU_MEMORY_TYPE_KV_CACHE": None,
        "_is_hip": False,
        "DSV41_INDEX_PAGE_SIZE": 64,
    }
    wanted = {
        "_num_dsv4_physical_kv_pages": None,
        "get_dsv4_indexer_bytes_per_token": None,
        "DeepSeekV4TokenToKVPool": {
            "_init_compressed_pools",
            "_make_indexer_pool",
            "get_contiguous_buf_infos",
        },
        "DeepSeekV4IndexerPool": {
            "_create_buffer",
            "get_bytes_per_token",
            "contiguous_page_row_buffers",
        },
    }
    selected = []
    for node in nodes:
        if getattr(node, "name", None) not in wanted:
            continue
        if isinstance(node, ast.ClassDef):
            node.bases, node.keywords = [], []
            node.body = [
                n for n in node.body if getattr(n, "name", None) in wanted[node.name]
            ]
        selected.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    index_cls = namespace["DeepSeekV4IndexerPool"]
    index_cls.index_k_with_scale_buffer_dtype = torch.uint8

    def index_init(
        self,
        size,
        physical_page_size,
        dtype,
        head_dim,
        layers,
        dev,
        saver,
        *,
        use_fp4_indexer,
        global_page_size,
    ):
        # Only the base KVCache constructor/memory-saver adapter is substituted.
        # Native _create_buffer and byte geometry perform the real allocation.
        self.size, self.page_size = size, physical_page_size
        self.global_page_size = global_page_size or physical_page_size
        self.index_head_dim, self.layer_num, self.device = head_dim, layers, dev
        self.use_fp4_indexer, self.uses_aiter_fp4_layout = use_fp4_indexer, False
        self.custom_mem_pool = None
        self.memory_saver_adapter = SimpleNamespace(region=lambda _: nullcontext())
        self._create_buffer()

    index_cls.__init__ = index_init
    pool = namespace["DeepSeekV4TokenToKVPool"]()
    pool.page_size, pool.full_size = page_size, full_size
    pool.compression_ratios = [1, 2]
    pool._stage_start, pool._stage_end = 0, 2
    pool._unified_kv = pool._unified_kv_fp8 = False
    pool.compressed_pool_configs = {}
    pool.compressed_kv_layout = lambda ratio: None
    pool.sources_by_ratio = {1: [0], 2: [1]}
    pool.indexer_head_dim = 128
    pool._make_kv_pool = lambda **kw: SimpleNamespace(
        kv_buffer=[
            torch.zeros(
                (
                    namespace["_num_dsv4_physical_kv_pages"](
                        kw["size"], kw["page_size"], kw["global_page_size"]
                    ),
                    16,
                ),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(kw["layer_num"])
        ]
    )
    pool._init_compressed_pools(
        stage_ratios=[1, 2],
        page_size=page_size,
        dtype=torch.uint8,
        device=device,
        enable_memory_saver=False,
        enable_hisparse=False,
        kv_pool_cls=None,
    )
    return pool
