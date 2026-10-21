#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

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
// an edge at that position; clones copy the same first position and inline
// edge. Extra edge: token, target, adjacency next, source, hash next, bucket
// head. Every non-sink state has a primary edge. Following only primary edges
// reaches the sink. Map each non-primary edge to a suffix whose LAST
// non-primary edge is that edge: take any root path to its source, then the
// primary-only tail. The mapping is injective since the automaton is
// deterministic. The whole string uses only primary edges, so there are at most
// n-1 non-primary edges. Stored extra edges are a subset (binary states inline
// their second target). Reserve H+1 rows, including tiny inputs; no
// corpus-dependent capacity guess. Meta: history length, states used, extra
// edges used, last, healthy, hash base, indexed extra edges (= live buckets).
// Hashing is only for high fanout states.
template <bool Compact> struct BasicAutomaton {
  // The last five fields form a link-cut forest over suffix links: left,
  // right, parent, lazy path increment, occurrence count. Updates expose the
  // root-to-last path, so repetitive histories do not walk every suffix.
  static constexpr int STATE_FIELDS = 10;
  static constexpr int STORAGE_FIELDS = Compact ? 11 : 10;
  static constexpr int COMPACT_MAX_HISTORY = 32767;
  using StateWord = typename std::conditional<Compact, uint16_t, int>::type;
  static constexpr int EDGE_CAPACITY_FACTOR = 1;
  static constexpr int EDGE_FIELDS = 6;
  static constexpr int META_FIELDS = 7;
  static constexpr int LINEAR_EDGES = 4;
  int *history;
  StateWord *states;
  int *edges, *meta;
  int capacity;
  // A compact row has nine uint16 fields plus a signed int32 secondary edge.
  // Store that edge as two halves: 22-byte rows need no unaligned int32 access.
  // Single fields use value+1 so 0 represents the -1 sentinel losslessly.
  struct Field {
    int *wide = nullptr;
    uint16_t *small = nullptr;
    bool paired = false;
    HYBRID_DEVICE Field(std::nullptr_t = nullptr) {}
    HYBRID_DEVICE Field(const Field &) = default;
    HYBRID_DEVICE Field(int *p) : wide(p) {}
    HYBRID_DEVICE Field(uint16_t *p, bool pair) : small(p), paired(pair) {}
    HYBRID_DEVICE operator int() const {
      if (wide)
        return *wide;
      if (paired)
        return static_cast<int32_t>(static_cast<uint32_t>(small[0]) |
                                    (static_cast<uint32_t>(small[1]) << 16));
      return static_cast<int>(*small) - 1;
    }
    HYBRID_DEVICE Field &operator=(int value) {
      if (wide)
        *wide = value;
      else if (paired) {
        small[0] = static_cast<uint16_t>(value);
        small[1] = static_cast<uint32_t>(value) >> 16;
      } else
        *small = static_cast<uint16_t>(value + 1);
      return *this;
    }
    HYBRID_DEVICE Field &operator=(const Field &other) {
      return *this = static_cast<int>(other);
    }
    HYBRID_DEVICE void operator+=(int value) { *this = int(*this) + value; }
    HYBRID_DEVICE Field &operator*() { return *this; }
    HYBRID_DEVICE bool operator==(std::nullptr_t) const {
      return !wide && !small;
    }
    HYBRID_DEVICE bool operator!=(std::nullptr_t) const {
      return wide || small;
    }
    HYBRID_DEVICE bool same_field(const Field &other) const {
      return wide == other.wide && small == other.small &&
             paired == other.paired;
    }
    HYBRID_DEVICE Field &bind(const Field &other) {
      wide = other.wide;
      small = other.small;
      paired = other.paired;
      return *this;
    }
  };
  HYBRID_DEVICE Field s(int i, int field) {
    if constexpr (Compact)
      return {states + STORAGE_FIELDS * i + field + (field > 4), field == 4};
    else
      return {states + STORAGE_FIELDS * i + field};
  }
  HYBRID_DEVICE int &e(int i, int field) {
    return edges[EDGE_FIELDS * i + field];
  }
  HYBRID_DEVICE int s(int i, int field) const {
    return const_cast<BasicAutomaton *>(this)->s(i, field);
  }
  HYBRID_DEVICE const int &e(int i, int field) const {
    return edges[EDGE_FIELDS * i + field];
  }
  HYBRID_DEVICE void init_frequency(int p, int count = 0) {
    s(p, 5) = s(p, 6) = s(p, 7) = -1;
    s(p, 8) = 0;
    s(p, 9) = count;
  }
  HYBRID_DEVICE bool frequency_root(int p) const {
    int parent = s(p, 7);
    return parent < 0 || (s(parent, 5) != p && s(parent, 6) != p);
  }
  HYBRID_DEVICE void frequency_add(int p, int count) {
    if (p >= 0) {
      s(p, 8) += count;
      s(p, 9) += count;
    }
  }
  HYBRID_DEVICE void frequency_push(int p) {
    int count = s(p, 8);
    if (count) {
      frequency_add(s(p, 5), count);
      frequency_add(s(p, 6), count);
      s(p, 8) = 0;
    }
  }
  HYBRID_DEVICE void frequency_rotate(int p) {
    int parent = s(p, 7), grandparent = s(parent, 7);
    int side = s(parent, 6) == p;
    int child = s(p, 5 + (1 - side));
    if (!frequency_root(parent))
      s(grandparent, 5 + (s(grandparent, 6) == parent)) = p;
    s(p, 7) = grandparent;
    s(parent, 5 + side) = child;
    if (child >= 0)
      s(child, 7) = parent;
    s(p, 5 + (1 - side)) = parent;
    s(parent, 7) = p;
  }
  HYBRID_DEVICE void frequency_splay(int p) {
    // Only additive tags, no path reversal or subtree aggregates. Pushing the
    // rotated triple suffices; an unpushed ancestor still covers the same set.
    while (!frequency_root(p)) {
      int parent = s(p, 7), grandparent = s(parent, 7);
      bool double_rotation = !frequency_root(parent);
      if (double_rotation)
        frequency_push(grandparent);
      frequency_push(parent);
      frequency_push(p);
      if (double_rotation)
        frequency_rotate(
            (s(parent, 6) == p) == (s(grandparent, 6) == parent) ? parent : p);
      frequency_rotate(p);
    }
    frequency_push(p);
  }
  HYBRID_DEVICE void frequency_access(int p) {
    int previous = -1;
    for (int node = p; node >= 0; node = s(node, 7)) {
      frequency_splay(node);
      s(node, 6) = previous;
      if (previous >= 0)
        s(previous, 7) = node;
      previous = node;
    }
    frequency_splay(p);
  }
  HYBRID_DEVICE void frequency_cut(int p) {
    frequency_access(p);
    int left = s(p, 5);
    if (left >= 0)
      s(left, 7) = -1;
    s(p, 5) = -1;
  }
  HYBRID_DEVICE void record_occurrence(int p) {
    frequency_access(p);
    frequency_add(p, 1);
  }
  HYBRID_DEVICE int occurrences(int p) const {
    // Read-only: donor automata are queried concurrently by multiple blocks.
    // Never splay or push tags here. Ancestor tags have not reached p yet.
    int count = s(p, 9);
    while (!frequency_root(p)) {
      p = s(p, 7);
      count += s(p, 8);
    }
    return count;
  }
  HYBRID_DEVICE void rank_transition(int next, int token, int &best,
                                     int &best_token, int &best_count) const {
    int count = occurrences(next);
    if (count > best_count || (count == best_count && token < best_token)) {
      best = next;
      best_token = token;
      best_count = count;
    }
  }
  HYBRID_DEVICE int frequent_next(int state, int &token) const {
    if (s(state, 3) < 0)
      return -1;
    int best = -1, count = -1;
    token = 2147483647;
    rank_transition(s(state, 3), history[s(state, 2) + 1], best, token, count);
    int secondary = s(state, 4);
    if (secondary < -1) {
      int next = ~secondary;
      rank_transition(next, history[s(next, 2)], best, token, count);
    } else {
      for (int edge = secondary; edge >= 0; edge = e(edge, 2))
        rank_transition(e(edge, 1), e(edge, 0), best, token, count);
    }
    return best;
  }
  HYBRID_DEVICE int draft(int state, int count, int64_t *output,
                          bool frequency) const {
    if (!frequency) {
      int start = s(state, 2) + 1;
      int valid = meta[0] - start;
      if (valid > count)
        valid = count;
      if (output)
        for (int j = 0; j < valid; ++j)
          output[j] = history[start + j];
      return valid;
    }
    int valid = 0;
    for (; valid < count; ++valid) {
      int token;
      state = frequent_next(state, token);
      if (state < 0)
        break;
      if (output)
        output[valid] = token;
    }
    return valid;
  }
  HYBRID_DEVICE static uint32_t hash(int p, int token) {
    uint32_t x =
        static_cast<uint32_t>(token) ^ (static_cast<uint32_t>(p) * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
  }
  HYBRID_DEVICE int bucket(uint32_t key) {
    // Linear hashing: base is a power of two; buckets [0, count-base) have
    // already split. The table grows by one bucket per indexed edge, never a
    // full rebuild during decode. Neither hash addressing nor import depends on
    // H.
    int base = meta[5];
    int b = key & (base - 1);
    return b < meta[6] - base ? key & (2 * base - 1) : b;
  }
  // A negative secondary field stores ~target directly; -1 means absent.
  // Nonnegative values address an extra-edge list. Targets in that list and
  // in the primary field are ordinary nonnegative state IDs.
  HYBRID_DEVICE Field find(int p, int token) {
    if (s(p, 3) == -1)
      return nullptr;
    if (history[s(p, 2) + 1] == token)
      return s(p, 3);
    int head = s(p, 4);
    if (head < 0)
      return head != -1 && history[s(~head, 2)] == token ? s(p, 4) : Field{};
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
  HYBRID_DEVICE static int target(Field edge) {
    int value = edge;
    return value < 0 ? ~value : value;
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
      Field edge = find(state, context[i]);
      while (state && edge == nullptr) {
        state = s(state, 1);
        length = s(state, 0);
        edge.bind(find(state, context[i]));
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
    init_frequency(i);
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
      s(current, 7) = s(current, 1);
      record_occurrence(current);
      meta[3] = current;
      return;
    }
    int p = meta[3];
    Field edge;
    while (p >= 0 && edge.bind(find(p, token)) == nullptr) {
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
        // The clone initially inherits q's complete end-position count;
        // then q is reparented before the new suffix occurrence is recorded.
        s(clone, 9) = occurrences(q);
        s(clone, 7) = s(clone, 1);
        frequency_cut(q);
        s(q, 7) = clone;
        s(clone, 3) = s(q, 3);
        if (s(q, 4) < 0) {
          s(clone, 4) = s(q, 4);
        } else {
          for (int copy = s(q, 4); copy != -1; copy = e(copy, 2))
            if (!add_extra(clone, e(copy, 0), e(copy, 1)))
              return;
        }
        while (p >= 0 && edge.bind(find(p, token)) != nullptr &&
               target(edge) == q) {
          *edge = *edge < 0 ? ~clone : clone;
          p = s(p, 1);
        }
        s(q, 1) = s(current, 1) = clone;
      }
    }
    s(current, 7) = s(current, 1);
    record_occurrence(current);
    meta[3] = current;
  }
};

using Automaton = BasicAutomaton<false>;
using CompactAutomaton = BasicAutomaton<true>;

#undef HYBRID_DEVICE
