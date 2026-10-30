"""D scheduler overlap policy; reuse native result completion, not extra fences."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchProof:
    # Attach before launch so a failed/partial forward can never look completed.
    result: object = None

    def ready(self):
        if self.result is None:
            raise RuntimeError("D forward completion is unproven; retaining target KV")
        # CUDA query on a never-recorded event can return True. SGL allocates
        # copy_done before deferred sampling; it records/submits the copy later.
        if self.result.delay_sample_func is not None:
            return False
        event = self.result.copy_done
        if event is None:
            raise RuntimeError("D overlap result has no GPU completion proof")
        if not event.query():
            return False
        # The CC worker's HostCopyDone.query() signals failure as well as success.
        if getattr(event, "error", None) is not None:
            raise RuntimeError(
                "D result copy failed; retaining target KV"
            ) from event.error
        return True


class DecodeOverlap:
    def __init__(self, controller):
        self.controller = controller
        self.exit_boundary = False
        self.last_policy = None
        from .retirement import Retirement

        self.retirement = Retirement(controller)
        logger.info("[remote-attn-overlap] D enabled: Remote serial, Local native")

    def serial(self):
        if not self.controller.graph_fast_path:
            raise RuntimeError("D overlap requires TP-agreed progressive lifetimes")
        if self.controller.targets or self.retirement.pending is not None:
            self.exit_boundary = True
            policy = "remote-serial"
            serial = True
        elif self.exit_boundary:
            # Consume the final queued Remote result before launching Local.
            self.exit_boundary = False
            policy = "local-boundary"
            serial = True
        else:
            policy = "local-native"
            serial = False
        if self.last_policy != policy:
            self.last_policy = policy
            logger.info("[remote-attn-overlap] policy=%s", policy)
        return serial

    def run_batch(self, original, scheduler, batch, *args, **kwargs):
        self.retirement.finalize()
        if not self.controller.graph_fast_path:
            raise RuntimeError("D overlap requires TP-agreed progressive lifetimes")
        # PREBUILT carries P's sampled token; it has no model forward/copy_done.
        # Its optional inner idle batch re-enters the ordinary run_batch hook.
        if not self.controller.targets or batch.forward_mode.is_prebuilt():
            return original(scheduler, batch, *args, **kwargs)
        active = {id(t.decode_req.req): t for t in self.controller.targets}
        targets = [active[id(req)] for req in batch.reqs if id(req) in active]
        if not targets:
            return original(scheduler, batch, *args, **kwargs)
        proof = BatchProof()
        for target in targets:
            target.forward_proofs.append(proof)
            target.forward_count += 1
        proof.result = original(scheduler, batch, *args, **kwargs)
        # The native copy_done covers forward, future-map publication, sampling
        # and readback (on copy_stream after waiting for forward_stream). Retain
        # the result through delayed sampling and never query its empty event.
        return proof.result


def install(controller):
    if not controller.scheduler.enable_overlap:
        return None
    if not controller.graph_fast_path:
        raise RuntimeError("D overlap cannot start with unmanaged handoff states")
    overlap = DecodeOverlap(controller)
    from .retirement import wrap_receiver

    try:
        controller.scheduler.request_receiver = wrap_receiver(
            controller.scheduler.request_receiver, overlap.retirement
        )
    except BaseException:
        overlap.retirement.close()
        raise
    return overlap


def disable_overlap(original, scheduler, batch, last_batch):
    native = original(scheduler, batch, last_batch)
    controller = getattr(scheduler, "_pre_layerwise", None)
    overlap = getattr(controller, "decode_overlap", None)
    if overlap is None:
        return native
    return overlap.serial() or native


def reap(original, queue, *args, **kwargs):
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if getattr(controller, "decode_overlap", None) is not None:
        # Before retraction/polling gates, refresh local proofs and publish
        # state-edge notices only. This never enters a retirement collective.
        controller.retire_targets()
    return original(queue, *args, **kwargs)


def before_retract(original, *, req, **kwargs):
    source = getattr(req, "_dsv41_p_lease", None)
    if source is not None:
        # Native release_req may back up source KV before release_kv_cache;
        # waiting at the allocator alone would be too late.
        source.overlap.drain(req)
    target = getattr(req, "_pre_layerwise_target", None)
    if target is not None and getattr(target.owner, "decode_overlap", None) is not None:
        # Native release_req backs up the KV BEFORE release_kv_cache, then resets
        # the Req. Both operations need the fully Local prefix and drained GPU.
        # This TP-symmetric native retraction boundary must retire the lease first.
        target.owner.drain_targets(required=target)
        kv = getattr(req, "kv", None)
        if kv is not None and not kv.holds_kv:
            # Cancellation may have queued native free before this boundary.
            return False
    return original(req=req, **kwargs)


def after_result(original, scheduler, *args, **kwargs):
    p_overlap = getattr(
        getattr(scheduler, "_pre_layerwise", None), "prefill_overlap", None
    )
    if p_overlap is not None:
        from .prefill_overlap import after_result as process_result

        return process_result(original, scheduler, *args, **kwargs)
    result = original(scheduler, *args, **kwargs)
    controller = getattr(scheduler, "_pre_layerwise", None)
    overlap = getattr(controller, "decode_overlap", None)
    if overlap is not None:
        # Remote's native serial gate has consumed the previous result before
        # the next launch. Empty/idle iterations reach the sample hook as well.
        overlap.retirement.observe()
        overlap.retirement.finalize()
    return result
