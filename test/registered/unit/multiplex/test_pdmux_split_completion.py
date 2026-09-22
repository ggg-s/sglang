import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin


class _Event:
    def __init__(self, ready):
        self.ready = ready

    def query(self):
        return self.ready


class _Work:
    def __init__(self, flags, ready_ranks):
        self.flags = flags
        self.ready_ranks = ready_ranks
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1
        self.flags[0] = self.ready_ranks


def _scheduler(tp_size):
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(tp_size=tp_size),
        _pdmux_split_done_work=None,
        _pdmux_split_done_flags=None,
    )
    scheduler._merge_finished_prefill_batch = Mock(return_value="merged")
    return scheduler


def _advance(scheduler, event):
    return SchedulerMultiplexMixin._advance_split_prefill_completion(
        scheduler,
        prefill_exe_done=event,
        prefill_result="result",
        prefill_stream="prefill",
        decode_stream="decode",
        running_batch="running",
        decode_result_done="decode_done",
    )


class TestPdmuxSplitCompletion(unittest.TestCase):
    def test_single_rank_bypasses_collective(self):
        scheduler = _scheduler(tp_size=1)
        scheduler.tp_cpu_group = SimpleNamespace(allreduce=Mock())

        running, ready = _advance(scheduler, _Event(False))
        self.assertEqual(running, "running")
        self.assertFalse(ready)
        scheduler.tp_cpu_group.allreduce.assert_not_called()

        running, ready = _advance(scheduler, _Event(True))
        self.assertEqual(running, "merged")
        self.assertTrue(ready)
        scheduler.tp_cpu_group.allreduce.assert_not_called()
        scheduler._merge_finished_prefill_batch.assert_called_once()

    def test_multi_rank_consumes_vote_next_iteration(self):
        scheduler = _scheduler(tp_size=2)
        works = []

        def allreduce(flags, _op):
            work = _Work(flags, ready_ranks=2)
            works.append(work)
            return work

        scheduler.tp_cpu_group = SimpleNamespace(allreduce=Mock(side_effect=allreduce))

        running, ready = _advance(scheduler, _Event(True))
        self.assertEqual(running, "running")
        self.assertFalse(ready)
        self.assertEqual(works[0].wait_count, 0)
        scheduler._merge_finished_prefill_batch.assert_not_called()

        running, ready = _advance(scheduler, _Event(True))
        self.assertEqual(running, "merged")
        self.assertTrue(ready)
        self.assertEqual(works[0].wait_count, 1)
        scheduler._merge_finished_prefill_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
