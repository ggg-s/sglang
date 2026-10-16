"""Execute the repository's actual draft_forward body without importing CUDA
serving dependencies. Only the model and kernel boundaries use CPU substitutes.
These checks do not replace a real SGLang server/CUDA test.
"""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
from hybrid_spec.core.contracts import DraftOverrideView

ROOT = Path(__file__).resolve().parents[3]


def load_functions(relative, names, namespace, class_name=None):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    definitions = [
        n for n in body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in definitions} == set(names)
    for node in definitions:
        node.decorator_list = []  # Disable torch.compile for the CPU harness.
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("matrix,hot", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize(
    "mask,gate",
    [
        (None, 0),
        ([False, False], 0),
        ([True, True], 0),
        ([True, False], 0),
        ([True, True], 2),
    ],
)
@pytest.mark.parametrize("mrope", [False, True])
def test_selected_prefix_drives_model_and_verify_chain(
    k, matrix, hot, mask, mrope, gate
):
    def postprocess(logits, positions, drafts, step):
        index = logits.argmax(dim=-1, keepdim=True)
        positions.add_(1)
        if drafts is not None:
            drafts[:, step : step + 1].copy_(index)
        return torch.ones_like(index, dtype=torch.float32), index

    def check_oob(tokens, low, high, *args):
        assert ((tokens >= low) & (tokens < high)).all()

    ns = dict(
        torch=torch,
        contextlib=contextlib,
        List=List,
        Optional=Optional,
        ForwardBatch=object,
        EagleDraftInput=object,
        _is_cuda=matrix,
        _is_hip=False,
        get_spec=lambda: SimpleNamespace(speculative_use_rejection_sampling=False),
        maybe_detect_nan=lambda *args: None,
        maybe_detect_inf=lambda *args: None,
        maybe_detect_oob=check_oob,
        fast_topk=torch.topk,
        draft_topk1_postprocess=postprocess,
        forward_context=lambda *args: contextlib.nullcontext(),
        ForwardContext=lambda **kwargs: None,
        IndexTopKShareState=SimpleNamespace(
            mtp_iteration=lambda *args, **kwargs: contextlib.nullcontext()
        ),
    )
    load_functions(
        "python/sglang/srt/speculative/spec_utils.py",
        [
            "select_top_k_tokens",
            "_select_top_k_tokens_first",
            "_select_top_k_tokens_later",
        ],
        ns,
    )
    load_functions(
        "python/sglang/srt/speculative/eagle_utils.py",
        ["per_step_draft_out_cache_loc"],
        ns,
    )
    load_functions(
        "python/sglang/srt/speculative/eagle_worker_v2.py",
        ["draft_forward"],
        ns,
        "EagleDraftWorker",
    )
    inputs = []
    positions = []
    mrope_positions = []

    def model_forward(batch):
        inputs.append(batch.input_ids.clone())
        positions.append(batch.positions.clone())
        if mrope:
            mrope_positions.append(batch.mrope_positions.clone())
        logits = torch.zeros((2, 8))
        logits.scatter_(1, ((batch.input_ids + 1) % 8).view(-1, 1), 1)
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=logits, hidden_states=None)
        )

    hot_tokens = torch.arange(8) + 10 if hot else None
    retrieval = torch.arange(40, 40 + 2 * k).reshape(2, k)
    if gate:
        # Exercise agreement and later disagreement as well as immediate
        # disagreement, including the hot-token remapping recurrence.
        retrieval[0, : min(gate, k)] = torch.arange(1, 1 + min(gate, k)) + (
            10 if hot else 0
        )
    view = (
        None if mask is None else DraftOverrideView(retrieval, torch.tensor(mask), gate)
    )
    batch = SimpleNamespace(
        batch_size=2,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        out_cache_loc=torch.arange(2 * k),
        positions=torch.tensor([7, 19]),
        mrope_positions=torch.tensor([[7, 19]]).expand(3, -1).clone(),
        spec_info=SimpleNamespace(
            topk_p=torch.ones((2, 1)),
            topk_index=torch.tensor([[1], [2]]),
            hidden_states=None,
            draft_override=view,
        ),
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    worker = SimpleNamespace(
        topk=1,
        speculative_num_steps=k,
        speculative_num_draft_tokens=k + 1,
        hot_token_id=hot_tokens,
        _topk1_parents_prealloc=torch.zeros((2, k), dtype=torch.int64),
        _topk1_score_indices_prealloc=torch.arange(k).repeat(2, 1),
        index_share_for_mtp_iteration=False,
        seed_dsa_topk_from_draft_extend=False,
        draft_attn_backend=SimpleNamespace(attn_backends=[None] * k),
        draft_runner=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["TestModel"]),
                model_is_mrope=mrope,
            ),
            canary_manager=None,
            forward=model_forward,
        ),
    )
    drafts = ns["draft_forward"](worker, batch)[2]
    expected = torch.empty((2, k), dtype=torch.int64)
    for row in range(2):
        token = row + 1 + (10 if hot else 0)
        eligible = mask is not None and mask[row]
        for step in range(k):
            if step < gate and token != retrieval[row, step].item():
                eligible = False
            if eligible:
                token = retrieval[row, step].item()
            expected[row, step] = token
            token = (token + 1) % 8 + (10 if hot else 0)
    assert torch.equal(drafts, expected)
    assert len(inputs) == k - 1
    for step, tokens in enumerate(inputs):
        assert torch.equal(tokens, expected[:, step])
        assert torch.equal(positions[step], torch.tensor([7, 19]) + step)
        if mrope:
            assert torch.equal(mrope_positions[step], positions[step].expand(3, -1))
