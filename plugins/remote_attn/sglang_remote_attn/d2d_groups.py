"""D2D transfer grouping; execution readiness remains per layer/resource."""

import os

PACKED = "kv-entry-pack-v1"
PACKED_ROWS = "state-row-pack-v1"
MAX_SEGMENTS = 64
MAX_ROW_SEGMENTS = 256


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
    entries,
    pages,
    width,
    capacity,
    *,
    start_page=0,
    only_group=None,
    prefixes=None,
    sort_by_stride=False,
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
        producers = groups[group]
        if sort_by_stride:
            producers.sort(key=lambda item: item[1])
        for i, stride in producers:
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


def row_frames(jobs, capacity):
    """Pack bounded runs of native state and index rows into one RDMA slot."""
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("D2D packed row budget must be positive")
    frame, used = [], 0
    for job in jobs:
        kind, entry, dst, indices, rows, *prefix = job
        start = prefix[0] if prefix else 0
        if indices.numel() != rows or not 0 <= start <= rows:
            raise ValueError("invalid D2D destination row range")
        stride = dst.shape[1]
        if not 0 < stride <= capacity:
            raise ValueError("D2D state/global row exceeds frame budget")
        while start < rows:
            count = min(rows - start, (capacity - used) // stride)
            if not count or len(frame) == MAX_ROW_SEGMENTS:
                yield frame
                frame, used = [], 0
                continue
            segment = dict(kind=kind, entry=entry, start=start, count=count)
            frame.append((segment, dst, indices))
            used += count * stride
            start += count
    if frame:
        yield frame


def validate_rows(source, segments, capacity):
    """Resolve packed state/index rows within the frozen source ownership."""
    if not isinstance(segments, list) or not 0 < len(segments) <= MAX_ROW_SEGMENTS:
        raise ValueError("invalid D2D packed row count")
    total, selected = 0, []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("invalid D2D packed row segment")
        kind = segment.get("kind")
        i, start, count = (segment.get(k) for k in ("entry", "start", "count"))
        if not all(type(x) is int for x in (i, start, count)):
            raise ValueError("invalid D2D packed row segment")
        if kind == "state" and 0 <= i < len(source.states):
            _, table, indices = source.states[i]
        elif (
            kind == "global"
            and 0 <= i < len(source.globals)
            and source.schema["entries"][i][0] == "index"
        ):
            table, indices = source.globals[i], source.pages
        else:
            raise ValueError("invalid D2D packed row entry")
        if start < 0 or count < 1 or start + count > indices.numel():
            raise ValueError("D2D packed rows outside frozen rows")
        total += count * table.shape[1]
        if total > capacity:
            raise ValueError("D2D packed rows exceed budget")
        selected.append((table, indices[start : start + count]))
    return selected


def scatter_rows(data, frame):
    """Scatter a complete packed frame into the reserved native rows."""
    sizes = [segment["count"] * dst.shape[1] for segment, dst, _ in frame]
    if data.numel() != sum(sizes):
        raise ValueError("D2D packed row frame size does not match destination")
    offset = 0
    for (segment, dst, indices), size in zip(frame, sizes):
        start, count = segment["start"], segment["count"]
        dst.index_copy_(
            0,
            indices[start : start + count],
            data[offset : offset + size].reshape(count, dst.shape[1]),
        )
        offset += size


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
    for segments in frames(
        entries, pages, width, capacity, prefixes=prefixes, sort_by_stride=True
    ):
        group = entries[segments[0]["entry"]][2] // width
        yield dict(
            layer_group=group,
            token_range=[0, prefix],
            segments=segments,
            wide_range=True,
        )
