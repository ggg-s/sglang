#pragma once

#if defined(__CUDACC__)
#define HYBRID_DEVICE __device__
#else
#define HYBRID_DEVICE
#endif

// Original implementation. One CUDA thread owns each automaton. No global
// singleton, allocator, atomics, or history updates inside a model graph.
// State: length, suffix link, first end position, adjacency head.
// Edge: target-token label, destination, next edge.
struct Automaton {
  int *history, *states, *edges, *meta;
  int capacity;
  HYBRID_DEVICE int &s(int i, int field) { return states[4 * i + field]; }
  HYBRID_DEVICE int &e(int i, int field) { return edges[3 * i + field]; }
  HYBRID_DEVICE int find(int p, int token) {
    for (int edge = s(p, 3); edge != -1; edge = e(edge, 2))
      if (e(edge, 0) == token)
        return edge;
    return -1;
  }
  HYBRID_DEVICE int new_state(int length, int link, int pos) {
    if (meta[1] >= 2 * capacity + 1) {
      meta[4] = 0;
      return -1;
    }
    int i = meta[1]++;
    s(i, 0) = length;
    s(i, 1) = link;
    s(i, 2) = pos;
    s(i, 3) = -1;
    return i;
  }
  HYBRID_DEVICE bool add_edge(int p, int token, int q) {
    if (meta[2] >= 3 * capacity + 1) {
      meta[4] = 0;
      return false;
    }
    int edge = meta[2]++;
    e(edge, 0) = token;
    e(edge, 1) = q;
    e(edge, 2) = s(p, 3);
    s(p, 3) = edge;
    return true;
  }
  HYBRID_DEVICE void append(int token) {
    if (!meta[4])
      return;
    if (meta[0] >= capacity) {
      meta[4] = 0;
      return;
    }
    int pos = meta[0]++;
    history[pos] = token;
    int current = new_state(s(meta[3], 0) + 1, 0, pos);
    if (current < 0)
      return;
    int p = meta[3], edge = -1;
    while (p >= 0 && (edge = find(p, token)) < 0) {
      if (!add_edge(p, token, current))
        return;
      p = s(p, 1);
    }
    if (p >= 0) {
      int q = e(edge, 1);
      if (s(p, 0) + 1 == s(q, 0)) {
        s(current, 1) = q;
      } else {
        int clone = new_state(s(p, 0) + 1, s(q, 1), s(q, 2));
        if (clone < 0)
          return;
        for (int copy = s(q, 3); copy != -1; copy = e(copy, 2))
          if (!add_edge(clone, e(copy, 0), e(copy, 1)))
            return;
        while (p >= 0 && (edge = find(p, token)) >= 0 && e(edge, 1) == q) {
          e(edge, 1) = clone;
          p = s(p, 1);
        }
        s(q, 1) = s(current, 1) = clone;
      }
    }
    meta[3] = current;
  }
};

#undef HYBRID_DEVICE
