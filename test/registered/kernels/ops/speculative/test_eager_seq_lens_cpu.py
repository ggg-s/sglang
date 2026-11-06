"""CUDA ordering and ownership checks for experimental PDMux length tickets."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class EagerSeqLensCpuTests(unittest.TestCase):
    def setUp(self):
        self.pool = SimpleNamespace(
            req_to_token=torch.zeros((8, 32), dtype=torch.int32, device="cuda"),
            req_generation=torch.ones(8, dtype=torch.int64),
        )
        self.relay = FutureMap(
            torch.device("cuda"), SpeculativeAlgorithm.DFLASH, self.pool
        )
        self.relay.enable_eager_seq_lens_cpu()
        self.batch = self.make_batch([1, 3])

    def tearDown(self):
        torch.cuda.synchronize()

    def make_batch(self, indices):
        return SimpleNamespace(
            spec_info=SimpleNamespace(
                future_indices=torch.tensor(indices, dtype=torch.int64, device="cuda")
            ),
            req_pool_indices_cpu=torch.tensor(indices, dtype=torch.int64),
        )

    def publish(self, lengths, indices=None):
        if indices is None:
            indices = self.batch.spec_info.future_indices
        self.relay.publish(
            indices, torch.tensor(lengths, dtype=torch.int64, device="cuda")
        )
        return self.relay.get_seq_lens_cpu_ticket(self.batch)

    def test_ready_mirror_does_not_wait_for_later_state_work(self):
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            ticket = self.publish([11, 31])
            # Deliberately keep the producer busy after the length snapshot.
            # The D2H ticket must not wait for this later state-work tail.
            torch.cuda._sleep(500_000_000)
            tail = producer.record_event()
        ticket.done.synchronize()
        self.assertFalse(tail.query())
        self.relay.resolve_seq_lens_cpu(self.batch)
        torch.testing.assert_close(self.batch.seq_lens_cpu, torch.tensor([11, 31]))
        torch.cuda.current_stream().wait_stream(producer)

    def test_later_relay_writes_cannot_corrupt_snapshot(self):
        ticket = self.publish([11, 31])
        self.relay.new_seq_lens_buf[self.batch.spec_info.future_indices] = 99
        torch.cuda.synchronize()
        torch.testing.assert_close(
            ticket.cpu_lens[self.batch.req_pool_indices_cpu], torch.tensor([11, 31])
        )

    def test_ring_wrap_and_consumed_host_lengths(self):
        self.publish([11, 31])
        self.relay.resolve_seq_lens_cpu(self.batch)
        consumed = self.batch.seq_lens_cpu
        for step in range(2, 12):
            self.publish([10 + step, 30 + step])
            self.relay.resolve_seq_lens_cpu(self.batch)
            torch.testing.assert_close(
                self.batch.seq_lens_cpu, torch.tensor([10 + step, 30 + step])
            )
        torch.testing.assert_close(consumed, torch.tensor([11, 31]))

    def test_superseded_ticket_and_prefill_merge_rows(self):
        old = self.publish([11, 31])
        merge_indices = torch.tensor([2], dtype=torch.int64, device="cuda")
        self.publish([22], merge_indices)
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            self.relay.seq_lens_cpu_ticket_ready(old)
        self.batch = self.make_batch([3, 2, 1])
        self.relay.resolve_seq_lens_cpu(self.batch)
        torch.testing.assert_close(self.batch.seq_lens_cpu, torch.tensor([31, 22, 11]))

    def test_snapshot_slot_reuse_before_old_copy_completes(self):
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            for step in range(8):
                self.publish([11 + step, 31 + step])
        self.relay.resolve_seq_lens_cpu(self.batch)
        torch.testing.assert_close(self.batch.seq_lens_cpu, torch.tensor([18, 38]))

    def test_pool_row_reuse_is_rejected(self):
        self.publish([11, 31])
        self.pool.req_generation[1] += 1
        with self.assertRaisesRegex(RuntimeError, "recycled"):
            self.relay.resolve_seq_lens_cpu(self.batch)


if __name__ == "__main__":
    unittest.main()
