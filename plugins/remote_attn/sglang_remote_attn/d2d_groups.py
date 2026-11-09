"""D2D transfer grouping; execution readiness remains per layer/resource."""

import os

PACKED = "kv-entry-pack-v1"
MAX_SEGMENTS = 64


def size(value=None):
    value = os.environ.get("HANDOFF_D2D_GROUP_SIZE", "1") if value is None else value
    if isinstance(value, bool):
        raise ValueError("HANDOFF_D2D_GROUP_SIZE must be a positive integer")
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("HANDOFF_D2D_GROUP_SIZE must be a positive integer") from exc
    if result < 1 or not isinstance(value, (int, str)):
        raise ValueError("HANDOFF_D2D_GROUP_SIZE must be a positive integer")
    return result


def layers(count, width):
    return [tuple(range(lo, min(lo + width, count))) for lo in range(0, count, width)]


def frames(
    entries, pages, width, capacity, *, start_page=0, only_group=None, prefixes=None
):
    """Pack rows of distinct native KV producers within logical layer ranges.

    A producer is sent once even if multiple logical layers share its KV.
    Framing stays bounded independently of group width or total request size.
    """
    groups = {}
    for i, entry in enumerate(entries):
        if entry[0] == "kv":
            groups.setdefault(entry[2] // width, []).append((i, entry[-1]))
    for group in sorted(groups):
        if only_group is not None and group != only_group:
            continue
        frame, used = [], 0
        for i, stride in groups[group]:
            if not 0 < stride <= capacity:
                raise ValueError("D2D global row exceeds frame budget")
            start = max(start_page, prefixes[i]) if prefixes is not None else start_page
            while start < pages:
                count = min(pages - start, (capacity - used) // stride)
                if not count or len(frame) == MAX_SEGMENTS:
                    yield frame
                    frame, used = [], 0
                    continue
                frame.append(dict(entry=i, start=start, count=count))
                used += count * stride
                start += count
        if frame:
            yield frame


def validate_segments(source, segments, capacity):
    if not isinstance(segments, list) or not 0 < len(segments) <= MAX_SEGMENTS:
        raise ValueError("invalid D2D packed segment count")
    total, group, selected = 0, None, []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("invalid D2D packed segment")
        i, start, count = (segment.get(k) for k in ("entry", "start", "count"))
        if not all(type(x) is int for x in (i, start, count)) or not 0 <= i < len(
            source.globals
        ):
            raise ValueError("invalid D2D packed entry")
        entry = source.schema["entries"][i]
        if (
            entry[0] != "kv"
            or start < 0
            or count < 1
            or start + count > source.pages.numel()
        ):
            raise ValueError("D2D packed segment outside frozen KV")
        current = entry[2] // source.group_size
        if group is not None and group != current:
            raise ValueError("D2D packed segment crosses layer group")
        group = current
        table = source.globals[i]
        total += count * table.shape[1]
        if total > capacity:
            raise ValueError("D2D packed frame exceeds budget")
        selected.append((table, start, count))
    return selected


def range_frames(entries, prefix, page, width, token_range, capacity):
    from .d2d_ranges import ranges

    for group in sorted({e[2] // width for e in entries if e[0] == "kv"}):
        for lo, hi in ranges(prefix, token_range, page):
            for segments in frames(
                entries,
                (hi + page - 1) // page,
                width,
                capacity,
                start_page=lo // page,
                only_group=group,
            ):
                yield dict(layer_group=group, token_range=[lo, hi], segments=segments)


def rdma_frames(entries, prefix, page, width, capacity, prefixes):
    """Pack remaining pages across the old token rectangles, layer group first."""
    pages = (prefix + page - 1) // page
    for segments in frames(entries, pages, width, capacity, prefixes=prefixes):
        group = entries[segments[0]["entry"]][2] // width
        yield dict(
            layer_group=group,
            token_range=[0, prefix],
            segments=segments,
            wide_range=True,
        )
