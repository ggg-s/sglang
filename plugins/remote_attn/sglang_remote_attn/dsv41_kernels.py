"""GPU-only sparse partition and fused paged MLA; no KV writes or ownership changes.

Native top-k order/duplicates are preserved. Dequantized rows round to BF16 like
our reference reader, QK accumulates in FP32 and PV uses two BF16 probability terms with FP32 accumulation.
"""

import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["START", "END"])
def _map(
    Row,
    Inverse,
    Error,
    START,
    END,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = START + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Row + i * PAGE, i < END, 0)
    physical = slot // PAGE
    good = (slot >= 0) & (slot % PAGE == 0) & (physical < CAP)
    old = tl.atomic_cas(
        Inverse + tl.minimum(tl.maximum(physical, 0), CAP - 1),
        tl.full((BLOCK,), -1, tl.int32),
        tl.where((i < END) & good, i, -1).to(tl.int32),
    )
    # An invalid/out-of-range element must never install a logical page.
    bad = (i < END) & (~good | ((old != -1) & (old != i)))
    tl.atomic_or(Error, tl.sum(bad.to(tl.int32), 0) != 0)


@tr.jit(do_not_specialize=["K", "LOW", "HIGH"])
def _partition(
    Ids,
    Length,
    Inverse,
    Error,
    Local,
    Remote,
    K,
    CPP: tl.constexpr,
    CAP: tl.constexpr,
    LOW,
    HIGH,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(Length)
    slot = tl.load(Ids + i, i < K, -1)
    live = (i < K) & (i < n) & (slot >= 0)
    safe = live & (slot < CAP * CPP)
    page = tl.load(Inverse + slot // CPP, safe, -1)
    valid = safe & (page >= 0)
    pos = page * CPP + slot % CPP
    remote = valid & (pos >= LOW) & (pos < HIGH)
    tl.store(Local + i, tl.where(valid & ~remote, slot, -1), i < K)
    tl.store(Remote + i, tl.where(remote, pos, -1), i < K)
    tl.atomic_or(
        Error, (tl.sum((live & ~valid).to(tl.int32), 0) != 0) | (n < 0) | (n > K)
    )


@tr.jit
def _load_keys(
    Cache,
    Ids,
    offset,
    N,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    STRIDE: tl.constexpr,
    DATA: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_STEP: tl.constexpr,
    ENCODING: tl.constexpr,
    Error,
    BN: tl.constexpr,
):
    n = offset + tl.arange(0, BN)
    d = tl.arange(0, 512)
    ids = tl.load(Ids + n, n < N, -1)
    valid = (n < N) & (ids >= 0) & (ids < PAGES * PAGE)
    tl.atomic_or(Error, tl.sum(((n < N) & (ids >= PAGES * PAGE)).to(tl.int32), 0) != 0)
    base = ids // PAGE * STRIDE
    row = ids % PAGE
    if ENCODING == 0:
        # Native V4: 448 FP8 NoPE + 64 BF16 RoPE; scale/padding are separate.
        addr = base[:, None] + row[:, None] * DATA + d[None, :]
        byte = tl.load(Cache + addr, valid[:, None] & (d[None, :] < 448), 0)
        nope = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sb = tl.load(
            Cache
            + base[:, None]
            + SCALE
            + row[:, None] * SCALE_STEP
            + d[None, :] // 64,
            valid[:, None] & (d[None, :] < 448),
            127,
        )
        scale = tl.exp2(sb.to(tl.float32) - 127.0)
        scale = tl.where(scale < 1.1754943508222875e-38, 0.0, scale)
        addr = base[:, None] + row[:, None] * DATA + 448 + 2 * (d[None, :] - 448)
        lo = tl.load(Cache + addr, valid[:, None] & (d[None, :] >= 448), 0).to(
            tl.uint16
        )
        hi = tl.load(Cache + addr + 1, valid[:, None] & (d[None, :] >= 448), 0).to(
            tl.uint16
        )
        rope = (lo | (hi << 8)).to(tl.bfloat16, bitcast=True).to(tl.float32)
        values = tl.where(d[None, :] < 448, nope * scale, rope)
    elif ENCODING == 1:
        byte = tl.load(
            Cache + base[:, None] + row[:, None] * DATA + d[None, :], valid[:, None], 0
        )
        sb = tl.load(
            Cache
            + base[:, None]
            + SCALE
            + row[:, None] * SCALE_STEP
            + d[None, :] // 32,
            valid[:, None],
            127,
        )
        values = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.exp2(
            sb.to(tl.float32) - 127.0
        )
    else:
        byte = tl.load(
            Cache + base[:, None] + row[:, None] * DATA + d[None, :] // 2,
            valid[:, None],
            0,
        )
        nibble = (byte >> (4 * (d[None, :] % 2))) & 15
        x = (nibble & 7).to(tl.float32)
        value = tl.where(
            x < 2.0,
            x * 0.5,
            tl.exp2(tl.floor(x / 2.0) - 1.0) * (1.0 + (nibble & 1) * 0.5),
        )
        value = tl.where(nibble >= 8, -value, value)
        sb = tl.load(
            Cache
            + base[:, None]
            + SCALE
            + row[:, None] * SCALE_STEP
            + d[None, :] // 16,
            valid[:, None],
            0,
        )
        values = value * sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    bad = valid[:, None] & ((values != values) | (tl.abs(values) == float("inf")))
    tl.atomic_or(Error, tl.sum(tl.sum(bad.to(tl.int32), 1), 0) != 0)
    return values.to(tl.bfloat16), valid


@tr.jit
def _step(q, keys, valid, scale, m, l, acc):
    q_hi = q.to(tl.bfloat16)
    scores = tl.dot(q_hi, tl.trans(keys))
    if q.dtype != tl.bfloat16:
        q_lo = (q.to(tl.float32) - q_hi.to(tl.float32)).to(tl.bfloat16)
        scores += tl.dot(q_lo, tl.trans(keys))
    scores *= scale
    scores = tl.where(valid[None, :], scores, -float("inf"))
    next_m = tl.maximum(m, tl.max(scores, 1))
    alpha = tl.where(next_m != -float("inf"), tl.exp(m - next_m), 0.0)
    p = tl.where(
        valid[None, :] & (next_m[:, None] != -float("inf")),
        tl.exp(scores - next_m[:, None]),
        0.0,
    )
    # Two BF16 terms retain ~FP16-or-better probability precision without
    # TF32's mantissa loss or a large FP32 shared-memory tile. This also fits
    # the 4060's shared-memory limit with dynamic sparse lengths.
    p_hi = p.to(tl.bfloat16)
    p_lo = (p - p_hi.to(tl.float32)).to(tl.bfloat16)
    acc = acc * alpha[:, None] + tl.dot(p_hi, keys) + tl.dot(p_lo, keys)
    return next_m, l * alpha + tl.sum(p, 1), acc


@tr.jit(do_not_specialize=["NA", "NB", "SCALE_Q"])
def _partial(
    Q,
    A,
    AI,
    B,
    BI,
    Sink,
    Out,
    Lse,
    Error,
    H: tl.constexpr,
    QSTRIDE: tl.constexpr,
    SCALE_Q,
    NA,
    PA: tl.constexpr,
    NPA: tl.constexpr,
    SA: tl.constexpr,
    DA: tl.constexpr,
    BA: tl.constexpr,
    SSA: tl.constexpr,
    EA: tl.constexpr,
    NB,
    PB: tl.constexpr,
    NPB: tl.constexpr,
    SB: tl.constexpr,
    DB: tl.constexpr,
    BB: tl.constexpr,
    SSB: tl.constexpr,
    EB: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    q = tl.load(Q + h[:, None] * QSTRIDE + d[None, :], h[:, None] < H, 0.0)
    qf = q.to(tl.float32)
    tl.atomic_or(
        Error,
        tl.sum(tl.sum(((qf != qf) | (tl.abs(qf) == float("inf"))).to(tl.int32), 1), 0)
        != 0,
    )
    if HAS_SINK:
        m = tl.load(Sink + h, h < H, -float("inf")).to(tl.float32)
        tl.atomic_or(
            Error, tl.sum(((m != m) | (m == float("inf"))).to(tl.int32), 0) != 0
        )
        l = tl.where(m != -float("inf"), 1.0, 0.0)
    else:
        m = tl.full((BH,), -float("inf"), tl.float32)
        l = tl.full((BH,), 0.0, tl.float32)
    acc = tl.full((BH, 512), 0.0, tl.float32)
    for start in range(tl.cdiv(NA, BN)):
        keys, valid = _load_keys(
            A, AI, start * BN, NA, PA, NPA, SA, DA, BA, SSA, EA, Error, BN
        )
        m, l, acc = _step(q, keys, valid, SCALE_Q, m, l, acc)
    for start in range(tl.cdiv(NB, BN)):
        keys, valid = _load_keys(
            B, BI, start * BN, NB, PB, NPB, SB, DB, BB, SSB, EB, Error, BN
        )
        m, l, acc = _step(q, keys, valid, SCALE_Q, m, l, acc)
    out = acc / tl.where(l > 0.0, l, 1.0)[:, None]
    tl.store(Out + h[:, None] * 512 + d[None, :], out, h[:, None] < H)
    tl.store(Lse + h, tl.where(l > 0.0, m + tl.log(l), -float("inf")), h < H)


@tr.jit
def _merge(A, AL, B, BL, O, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = i // D
    a, b = tl.load(AL + h, h < H, -float("inf")), tl.load(BL + h, h < H, -float("inf"))
    m = tl.maximum(a, b)
    wa = tl.where(a != -float("inf"), tl.exp(a - m), 0.0)
    wb = tl.where(b != -float("inf"), tl.exp(b - m), 0.0)
    denom = tl.where(wa + wb > 0.0, wa + wb, 1.0)
    value = (
        tl.load(A + i, i < H * D, 0.0) * wa + tl.load(B + i, i < H * D, 0.0) * wb
    ) / denom
    tl.store(O + i, value, i < H * D)
