"""CPU regressions for segmented Qwen3.5 MoE text prefill state and guards."""

import ast, types, contextlib, unittest, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[4]
import runpy
register_cpu_ci = runpy.run_path(str(ROOT / 'python/sglang/test/ci/ci_register.py'))['register_cpu_ci']
register_cpu_ci(est_time=2, suite='base-a-test-cpu')
p = ROOT / 'python/sglang/srt/models/qwen3_5.py'
root = ast.parse(p.read_text())
cls = next((n for n in root.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5MoeForConditionalGeneration'))
fun = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_split_prefill'))
fun.decorator_list = []
from typing import Tuple, Optional
ns = {'torch': torch, 'Tuple': Tuple, 'Optional': Optional, 'ForwardBatch': object, 'envs': types.SimpleNamespace(SGLANG_QWEN35_NATIVE_FINAL_NORM=types.SimpleNamespace(get=lambda : False)), 'get_global_expert_distribution_recorder': lambda : types.SimpleNamespace(with_current_layer=lambda i: contextlib.nullcontext())}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fun], type_ignores=[])), str(p), 'exec'), ns)
run = ns['forward_split_prefill']

class SplitTests(unittest.TestCase):

    def fixture(self):
        calls = []

        def layer(i):

            def go(positions, hidden_states, residual, forward_batch):
                calls.append((i, positions.clone()))
                r = hidden_states if residual is None else residual + hidden_states
                return (torch.tanh(hidden_states * (i + 1) + positions[:, None] * 0.01), r)
            return go
        norm = lambda h, r: (h + r * 0.1, r)
        model = types.SimpleNamespace(end_layer=4, embed_tokens=lambda x: torch.stack([x.float(), x.float() + 1], 1), layers=[layer(i) for i in range(4)], norm=norm, flashinfer_mnnvl_cutedsl_fusion=None)
        obj = types.SimpleNamespace(model=model, pp_group=types.SimpleNamespace(is_first_rank=True, is_last_rank=True), capture_aux_hidden_states=False, is_mrope_enabled=False, lm_head=None, logits_processor=lambda ids, h, head, b: h.clone())
        batch = types.SimpleNamespace(mm_inputs=None, spec_info=None, residual=torch.ones(3, 2), mrope_positions=torch.arange(3.0) + 5)
        return (obj, batch, calls)

    def test_segments_match_full_forward(self):
        ids = torch.arange(3)
        positions = torch.arange(3.0)
        for step in [1, 2, 3, 4]:
            (o, b, calls) = self.fixture()
            h = o.model.embed_tokens(ids)
            r = None
            for layer in o.model.layers:
                (h, r) = layer(positions, h, r, b)
            expected = o.model.norm(h, r)[0]
            calls.clear()
            for a in range(0, 4, step):
                out = run(o, ids, positions, b, (a, min(a + step, 4)))
                if a + step < 4:
                    self.assertIsNone(out)
            torch.testing.assert_close(out, expected)
            self.assertEqual([x[0] for x in calls], list(range(4)))

    def test_mrope_positions_are_preserved(self):
        (o, b, calls) = self.fixture()
        o.is_mrope_enabled = True
        b.mm_inputs = [types.SimpleNamespace(contains_mm_input=lambda : False)]
        for a in range(4):
            run(o, torch.arange(3), torch.zeros(3), b, (a, a + 1))
        for (_, pos) in calls:
            torch.testing.assert_close(pos, b.mrope_positions)

    def test_input_embeddings(self):
        (o, b, _) = self.fixture()
        x = torch.randn(3, 2)
        o.model.embed_tokens = lambda _: self.fail('embedding lookup should be bypassed')
        self.assertIsNone(run(o, torch.arange(3), torch.zeros(3), b, (0, 1), x))

    def test_rejects_unsupported_paths(self):
        for variant in ['mm', 'spec', 'aux', 'pp', 'deferred']:
            (o, b, _) = self.fixture()
            if variant == 'mm':
                b.mm_inputs = [types.SimpleNamespace(contains_mm_input=lambda : True)]
            if variant == 'spec':
                b.spec_info = object()
            if variant == 'aux':
                o.capture_aux_hidden_states = True
            if variant == 'pp':
                o.pp_group.is_last_rank = False
            if variant == 'deferred':
                o.model.flashinfer_mnnvl_cutedsl_fusion = object()
            with self.assertRaises(ValueError):
                run(o, torch.arange(3), torch.zeros(3), b, (0, 1))
if __name__ == '__main__':
    unittest.main()
