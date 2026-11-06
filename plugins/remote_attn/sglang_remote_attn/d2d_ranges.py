"""Logical layer-group x token-range plans, independent of physical KV layout."""

import os

from .d2d_groups import layers, size

PROTOCOL = "layer-token-range-v1"


def tokens(value=None):
    value = (
        os.environ.get("HANDOFF_D2D_TOKEN_RANGE", "1024") if value is None else value
    )
    try:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, str))
            or int(value) < 1
        ):
            raise ValueError()
        return int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("HANDOFF_D2D_TOKEN_RANGE must be a positive integer") from exc


def ranges(length, width, alignment=1):
    if length < 1 or width < 1 or alignment < 1:
        raise ValueError("invalid D2D token range")
    # An aligned resource adapter owns the whole physical page. Only the
    # final interval may be partial; its scatter must protect the live tail.
    width = max(alignment, width // alignment * alignment)
    return [(lo, min(lo + width, length)) for lo in range(0, length, width)]


def rectangles(count, length, group, width, alignment=1):
    for members in layers(count, size(group)):
        for lo, hi in ranges(length, tokens(width), alignment):
            yield members, lo, hi


class Coverage:
    """Reject duplicates/gaps; readiness remains per resource, never per batch."""

    def __init__(self, count, length):
        self.length = length
        self.ends = [0] * count

    def add(self, entry, lo, hi):
        if (
            type(entry) is not int
            or not 0 <= entry < len(self.ends)
            or lo != self.ends[entry]
            or not lo < hi <= self.length
        ):
            raise ValueError("D2D range duplicate/gap/outside frozen ownership")
        self.ends[entry] = hi
        return hi == self.length


def check_history(origin, output, length):
    if type(length) is not int or not output or length != len(origin) + len(output) - 1:
        raise ValueError("D2D frozen KV length does not match pending-token history")
    return length
