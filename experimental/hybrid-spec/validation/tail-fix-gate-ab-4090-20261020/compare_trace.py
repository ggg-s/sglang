import gzip
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
x = json.loads(
    gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_text()
)
a = x["trace"]["(4, 4)"]
b = x["trace"]["(4, 1)"]
for step, (left, right) in enumerate(zip(a, b)):
    print("step", step)
    for key in (
        "input",
        "positions",
        "seq_lens",
        "reqs",
        "cache",
        "tokens_top4",
        "logits_top4",
    ):
        print(key, left[key][3], right[key][0])
    li = left["indices"][left["indptr"][3] : left["indptr"][4]]
    ri = right["indices"][right["indptr"][0] : right["indptr"][1]]
    print(
        "KV equal",
        li == ri,
        "lengths",
        len(li),
        len(ri),
        "diff",
        [(i, u, v) for i, (u, v) in enumerate(zip(li, ri)) if u != v][:8],
    )
