#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define HYBRID_DEVICE __device__
#else
#define HYBRID_DEVICE
#endif

// Original implementation. One CUDA thread owns each automaton. No global
// singleton, allocator, or atomics. CPU construction and captured GPU updates
// share this relocatable layout.
// State: length, suffix link, first end position, adjacency head.
// Edge: token, destination, next adjacency edge, source, next hash edge, bucket.
// The last column is an independent array of hash bucket heads, packed here so
// importing used edge rows also imports every live bucket. Adjacency is retained
// for clone enumeration; transition lookup uses exact (source, token) keys.
// Meta: history length, states used, edges used, last state, healthy, hash base,
// indexed edges (= live buckets). The hash never indexes a low-degree state.
struct Automaton {
  static constexpr int EDGE_FIELDS = 6;
  static constexpr int META_FIELDS = 7;
  static constexpr int LINEAR_EDGES = 4;
  int *history, *states, *edges, *meta;
  int capacity;
  HYBRID_DEVICE int &s(int i, int field) { return states[4 * i + field]; }
  HYBRID_DEVICE int &e(int i, int field) { return edges[EDGE_FIELDS * i + field]; }
  HYBRID_DEVICE static uint32_t hash(int p, int token) {
    uint32_t x = static_cast<uint32_t>(token) ^
                 (static_cast<uint32_t>(p) * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
  }
  HYBRID_DEVICE int bucket(uint32_t key) {
    // Linear hashing: base is a power of two; buckets [0, count-base) have
    // already split. The table grows by one bucket per indexed edge, never a full
    // rebuild during decode. Neither hash addressing nor import depends on H.
    int base = meta[5];
    int b = key & (base - 1);
    return b < meta[6] - base ? key & (2 * base - 1) : b;
  }
  HYBRID_DEVICE int find(int p, int token) {
    int head = s(p, 3);
    if (head == -1 || e(head, 0) == token)
      return head;
    // Generalize sa-spec's single-edge fast path to a bounded short list.
    // -2 marks an unindexed edge; indexed chains use -1 as their terminator.
    if (e(head, 4) == -2) {
      for (int edge = e(head, 2); edge != -1; edge = e(edge, 2))
        if (e(edge, 0) == token)
          return edge;
      return -1;
    }
    for (int edge = e(bucket(hash(p, token)), 5); edge != -1; edge = e(edge, 4))
      if (e(edge, 3) == p && e(edge, 0) == token)
        return edge;
    return -1;
  }
  HYBRID_DEVICE void grow_hash() {
    int count = meta[6];
    e(count, 5) = -1;
    if (count == 0) {
      meta[5] = 1;
      return;
    }
    int base = meta[5], split = count - base;
    int edge = e(split, 5);
    e(split, 5) = -1;
    while (edge != -1) {
      int next = e(edge, 4);
      int b = (hash(e(edge, 3), e(edge, 0)) & base) ? count : split;
      e(edge, 4) = e(b, 5);
      e(b, 5) = edge;
      edge = next;
    }
    if (count + 1 == 2 * base)
      meta[5] = 2 * base;
  }
  HYBRID_DEVICE void index_edge(int edge) {
    grow_hash();
    ++meta[6];
    int b = bucket(hash(e(edge, 3), e(edge, 0)));
    e(edge, 4) = e(b, 5);
    e(b, 5) = edge;
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
    e(edge, 3) = p;
    e(edge, 4) = -2;
    e(edge, 5) = -1;
    // Hash only high-fanout states. Pure copying/repetition creates many sparse
    // states; indexing them adds work without any gain. Promotion is bounded.
    int head = s(p, 3);
    if (head != -1 && e(head, 4) != -2) {
      index_edge(edge);
    } else {
      int count = 0;
      for (int copy = head; copy != -1; copy = e(copy, 2))
        ++count;
      if (count == LINEAR_EDGES)
        for (int copy = edge; copy != -1; copy = e(copy, 2))
          index_edge(copy);
    }
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
