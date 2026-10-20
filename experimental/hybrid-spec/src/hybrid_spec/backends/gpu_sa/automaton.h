#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define HYBRID_DEVICE __device__
#else
#define HYBRID_DEVICE
#endif

// One owner thread per automaton; relocatable CPU/CUDA POD storage.
// State: length, suffix link, first end position, first target, secondary.
// Secondary: -1 empty, ~target for a second inline edge, otherwise extra head.
// All paths entering a state end in history[first_pos], so a second inline edge
// needs no stored token. Retargeting to a clone preserves that first position.
// The first edge's token is history[first_pos + 1]. New states first acquire
// an edge at that position; clones copy the same first position and inline edge.
// Extra edge: token, target, adjacency next, source, hash next, bucket head.
// Every non-sink state has a primary edge. Following only primary edges reaches
// the sink. Map each non-primary edge to a suffix whose LAST non-primary edge
// is that edge: take any root path to its source, then the primary-only tail.
// The mapping is injective since the automaton is deterministic. The whole
// string uses only primary edges, so there are at most n-1 non-primary edges.
// Stored extra edges are a subset (binary states inline their second target).
// Reserve H+1 rows, including tiny inputs; no corpus-dependent capacity guess.
// Meta: history length, states used, extra edges used, last, healthy, hash base,
// indexed extra edges (= live buckets). Hashing is only for high fanout states.
struct Automaton {
  static constexpr int STATE_FIELDS = 5;
  static constexpr int EDGE_CAPACITY_FACTOR = 1;
  static constexpr int EDGE_FIELDS = 6;
  static constexpr int META_FIELDS = 7;
  static constexpr int LINEAR_EDGES = 4;
  int *history, *states, *edges, *meta;
  int capacity;
  HYBRID_DEVICE int &s(int i, int field) { return states[STATE_FIELDS * i + field]; }
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
  // A negative secondary field stores ~target directly; -1 means absent.
  // Nonnegative values address an extra-edge list. Targets in that list and
  // in the primary field are ordinary nonnegative state IDs.
  HYBRID_DEVICE int *find(int p, int token) {
    if (s(p, 3) == -1)
      return nullptr;
    if (history[s(p, 2) + 1] == token)
      return &s(p, 3);
    int head = s(p, 4);
    if (head < 0)
      return head != -1 && history[s(~head, 2)] == token ? &s(p, 4) : nullptr;
    if (e(head, 4) < -1) {
      for (int edge = head; edge != -1; edge = e(edge, 2))
        if (e(edge, 0) == token)
          return &e(edge, 1);
      return nullptr;
    }
    for (int edge = e(bucket(hash(p, token)), 5); edge != -1; edge = e(edge, 4))
      if (e(edge, 3) == p && e(edge, 0) == token)
        return &e(edge, 1);
    return nullptr;
  }
  HYBRID_DEVICE static int target(const int *edge) {
    return *edge < 0 ? ~*edge : *edge;
  }
  // All occurrences of a SAM state have the same right language. A single
  // outgoing edge at each step proves that no *observed* K-token continuation
  // disagrees. Occurrences ending at the text boundary provide no next token;
  // they do not count as a competing token. This is not a correctness score.
  HYBRID_DEVICE bool unique_continuation(int state, int count) {
    for (int i = 0; i < count; ++i) {
      if (s(state, 3) < 0 || s(state, 4) != -1)
        return false;
      state = s(state, 3);
    }
    return true;
  }
  // Longest suffix of an external request's context present in this SA.
  // Each donor is a separate text, so a lookup can never cross requests.
  HYBRID_DEVICE int match_suffix(const int *context, int count, int &length,
                                bool require_continuation = false) {
    int state = 0;
    length = 0;
    for (int i = 0; i < count; ++i) {
      int *edge = find(state, context[i]);
      while (state && edge == nullptr) {
        state = s(state, 1);
        length = s(state, 0);
        edge = find(state, context[i]);
      }
      if (edge != nullptr) {
        state = target(edge);
        ++length;
      } else {
        state = 0;
        length = 0;
      }
    }
    // A longest match at the source's end may have no future tokens. Follow
    // suffix links to an earlier occurrence instead of dropping the source.
    while (require_continuation && state && s(state, 2) + 1 >= meta[0]) {
      state = s(state, 1);
      length = s(state, 0);
    }
    return state;
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
    s(i, 3) = s(i, 4) = -1;
    return i;
  }
  HYBRID_DEVICE bool add_edge(int p, int token, int q) {
    if (s(p, 3) == -1) {
      s(p, 3) = q;
      return true;
    }
    int second = s(p, 4);
    if (second == -1) {
      s(p, 4) = ~q;
      return true;
    }
    if (second < -1) {
      int old = ~second;
      s(p, 4) = -1;
      if (!add_extra(p, history[s(old, 2)], old))
        return false;
    }
    return add_extra(p, token, q);
  }
  HYBRID_DEVICE bool add_extra(int p, int token, int q) {
    if (meta[2] >= capacity + 1) {
      meta[4] = 0;
      return false;
    }
    int head = s(p, 4), edge = meta[2]++;
    e(edge, 0) = token;
    e(edge, 1) = q;
    e(edge, 2) = head;
    e(edge, 3) = p;
    e(edge, 5) = -1;
    if (head == -1) {
      e(edge, 4) = -2;
    } else if (e(head, 4) >= -1) {
      index_edge(edge);
    } else {
      e(edge, 4) = e(head, 4) - 1;
      // Three extra edges plus inline stay linear; promote at total degree 5.
      // Remembering the degree avoids rescanning the list on each insertion.
      if (e(edge, 4) == -LINEAR_EDGES - 1)
        for (int copy = edge; copy != -1; copy = e(copy, 2))
          index_edge(copy);
    }
    s(p, 4) = edge;
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
    // A root with no secondary transition means the history has one symbol.
    // Extending that run creates the next chain state, with no lookup/clone.
    // The first different token takes the general path and disables this case.
    if (s(0, 4) == -1 && token == history[0]) {
      s(current, 1) = meta[3];
      s(meta[3], 3) = current;
      meta[3] = current;
      return;
    }
    int p = meta[3];
    int *edge = nullptr;
    while (p >= 0 && (edge = find(p, token)) == nullptr) {
      if (!add_edge(p, token, current))
        return;
      p = s(p, 1);
    }
    if (p >= 0) {
      int q = target(edge);
      if (s(p, 0) + 1 == s(q, 0)) {
        s(current, 1) = q;
      } else {
        int clone = new_state(s(p, 0) + 1, s(q, 1), s(q, 2));
        if (clone < 0)
          return;
        s(clone, 3) = s(q, 3);
        if (s(q, 4) < 0) {
          s(clone, 4) = s(q, 4);
        } else {
          for (int copy = s(q, 4); copy != -1; copy = e(copy, 2))
            if (!add_extra(clone, e(copy, 0), e(copy, 1)))
              return;
        }
        while (p >= 0 && (edge = find(p, token)) != nullptr && target(edge) == q) {
          *edge = *edge < 0 ? ~clone : clone;
          p = s(p, 1);
        }
        s(q, 1) = s(current, 1) = clone;
      }
    }
    meta[3] = current;
  }
};

#undef HYBRID_DEVICE
