"""Admission for PDMux + CPU overlap across speculative and non-speculative runs.

The PDMux overlap lane once required DFLASH/DSPARK unconditionally, which made
`--enable-pdmux` without speculation fail at startup. This pins the relaxed
gate: plain TP without speculation is admitted, and the combinations the lane
cannot drive yet (prefill CP, other speculative algorithms) still fail.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.validation_hook import check_pdmux_overlap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_cfg(**overrides):
    values = dict(
        pdmux_prefill_mode="layer_split",
        speculative_algorithm=None,
        enable_prefill_cp=False,
        tp_size=1,
        enable_dp_attention=False,
        ep_size=1,
        moe_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        enable_torch_compile=False,
        enable_two_batch_overlap=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestPdmuxOverlapAdmission(unittest.TestCase):
    def test_non_spec_plain_tp_is_admitted(self):
        check_pdmux_overlap(make_cfg())

    def test_non_spec_standard_prefill_is_admitted(self):
        check_pdmux_overlap(make_cfg(pdmux_prefill_mode="standard"))

    def test_non_spec_rejects_prefill_cp(self):
        with self.assertRaises(AssertionError):
            check_pdmux_overlap(make_cfg(enable_prefill_cp=True))

    def test_non_spec_rejects_dp_attention_with_tp(self):
        with self.assertRaises(AssertionError):
            check_pdmux_overlap(
                make_cfg(tp_size=8, enable_dp_attention=True)
            )

    def test_unsupported_speculative_algorithm_is_rejected(self):
        with self.assertRaises(AssertionError):
            check_pdmux_overlap(make_cfg(speculative_algorithm="NGRAM"))

    def test_dflash_is_still_admitted(self):
        check_pdmux_overlap(make_cfg(speculative_algorithm="DFLASH"))

    def test_eagle_mtp_is_admitted(self):
        check_pdmux_overlap(make_cfg(speculative_algorithm="EAGLE"))


if __name__ == "__main__":
    unittest.main()
