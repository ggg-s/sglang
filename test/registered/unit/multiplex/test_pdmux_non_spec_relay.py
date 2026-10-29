"""Non-spec split-prefill relay lands on the FutureMap device.

Without speculation the PDMux split-prefill merge relays a CPU sampled-token
tensor. Stashing it without a device move raised a CUDA device-mismatch error
while finalizing the first prefill, so the server died on the first request.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestNonSpecRelay(unittest.TestCase):
    def test_cpu_bonus_tokens_are_moved_to_the_relay_device(self):
        pool = SimpleNamespace(
            req_to_token=torch.zeros(
                (8, 4), dtype=torch.int32, device="cuda"
            )
        )
        future_map = FutureMap(
            torch.device("cuda"),
            SpeculativeAlgorithm.NONE,
            pool,
            needs_cpu_seq_lens=False,
        )
        indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        future_map.stash(
            indices, RelayPayload(bonus_tokens=torch.tensor([7, 9], dtype=torch.int64))
        )
        self.assertEqual(future_map.output_tokens_buf[indices].tolist(), [7, 9])


if __name__ == "__main__":
    unittest.main()
