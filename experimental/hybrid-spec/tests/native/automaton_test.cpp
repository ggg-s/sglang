#include "automaton.h"
#include "host_builder.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#ifdef NDEBUG
#error "Native differential tests require active assertions"
#endif

static void check_index(Automaton &sa) {
  std::vector<bool> seen(sa.meta[2], false);
  for (int b = 0; b < sa.meta[6]; ++b) {
    for (int edge = sa.e(b, 5); edge != -1; edge = sa.e(edge, 4)) {
      assert(edge >= 0 && edge < sa.meta[2] && !seen[edge]);
      seen[edge] = true;
      assert(sa.bucket(Automaton::hash(sa.e(edge, 3), sa.e(edge, 0))) == b);
    }
  }
  assert(std::count(seen.begin(), seen.end(), true) == sa.meta[6]);
  for (int p = 0; p < sa.meta[1]; ++p) {
    if (sa.s(p, 3) != -1) {
      assert(sa.s(p, 2) + 1 < sa.meta[0]);
      assert(sa.find(p, sa.history[sa.s(p, 2) + 1]).same_field(sa.s(p, 3)));
    }
    if (sa.s(p, 4) < -1) {
      int second = ~sa.s(p, 4);
      assert(sa.find(p, sa.history[sa.s(second, 2)]).same_field(sa.s(p, 4)));
    }
    for (int edge = sa.s(p, 4); edge >= 0; edge = sa.e(edge, 2)) {
      assert(sa.e(edge, 3) == p);
      assert(seen[edge] == (sa.e(sa.s(p, 4), 4) >= -1));
      assert(sa.find(p, sa.e(edge, 0)).same_field(Automaton::Field(&sa.e(edge, 1))));
    }
    assert(sa.find(p, -1) == nullptr);
  }
}

static void check(const std::vector<int> &input) {
  int capacity = static_cast<int>(input.size());
  std::vector<int> history(capacity, -99);
  std::vector<int> states((2 * capacity + 1) * Automaton::STATE_FIELDS, -99);
  std::vector<int> edges((capacity + 1) * Automaton::EDGE_FIELDS, -99);
  int meta[Automaton::META_FIELDS] = {0, 1, 0, 0, 1, 1, 0};
  states[0] = 0;
  states[1] = -1;
  states[2] = -1;
  states[3] = states[4] = -1;
  Automaton sa{history.data(), states.data(), edges.data(), meta, capacity};
  sa.init_frequency(0);
  for (int n = 1; n <= capacity; ++n) {
    sa.append(input[n - 1]);
    assert(meta[4] == 1 && meta[0] == n);
    assert(history[n - 1] == input[n - 1]);
    int best_length = 0, best_end = -1;
    // Independent exhaustive oracle, longest suffix then earliest occurrence.
    for (int length = n - 1; length > 0 && !best_length; --length) {
      for (int start = 0; start < n - length; ++start) {
        bool equal = true;
        for (int offset = 0; offset < length; ++offset)
          equal &= input[start + offset] == input[n - length + offset];
        if (equal) {
          best_length = length;
          best_end = start + length - 1;
          break;
        }
      }
    }
    int suffix = states[Automaton::STATE_FIELDS * meta[3] + 1];
    int length = suffix <= 0 ? 0 : states[Automaton::STATE_FIELDS * suffix];
    assert(length == best_length);
    if (length)
      assert(states[Automaton::STATE_FIELDS * suffix + 2] == best_end);
    // Independent substring oracle: every observed continuation (including a
    // short one at EOF) must agree with one complete K-token continuation.
    // This catches clones/inline secondary edges that hide a later branch.
    if (length) {
      for (int k = 1; k <= 4; ++k) {
        std::vector<std::vector<int>> tails;
        std::vector<int> complete;
        for (int start = 0; start + length <= n; ++start) {
          if (!std::equal(input.begin() + n - length, input.begin() + n,
                          input.begin() + start))
            continue;
          int end = std::min(n, start + length + k);
          tails.emplace_back(input.begin() + start + length, input.begin() + end);
          if (tails.back().size() == static_cast<size_t>(k))
            complete = tails.back();
        }
        bool unique = complete.size() == static_cast<size_t>(k);
        if (unique)
          for (const auto &tail : tails)
            unique &= std::equal(tail.begin(), tail.end(), complete.begin());
        assert(sa.unique_continuation(suffix, k) == unique);
      }
    }
    assert(meta[1] <= 2 * capacity + 1 && meta[2] <= n - 1);
    check_index(sa);
  }
  // Build a prompt in compact host storage, copy only used POD prefixes to a
  // larger decode slot and continue appending with the device's state machine.
  int prefix = (capacity + 1) / 2;
  std::vector<int64_t> input64(input.begin(), input.end());
  std::vector<int> host_history(prefix);
  std::vector<int> host_states((2 * prefix + 1) * Automaton::STATE_FIELDS);
  std::vector<int> host_edges((prefix + 1) * Automaton::EDGE_FIELDS);
  int host_meta[Automaton::META_FIELDS];
  int status = hybrid_sa_build_host(input64.data(), prefix, 2147483647,
                                   host_history.data(), host_states.data(),
                                   host_edges.data(), host_meta, prefix);
  assert(status == 0);
  std::vector<int> resumed_history(capacity), resumed_states(states.size()),
      resumed_edges(edges.size());
  std::copy_n(host_history.begin(), prefix, resumed_history.begin());
  std::copy_n(host_states.begin(), host_meta[1] * Automaton::STATE_FIELDS, resumed_states.begin());
  std::copy_n(host_edges.begin(), host_meta[2] * Automaton::EDGE_FIELDS,
              resumed_edges.begin());
  Automaton resumed{resumed_history.data(), resumed_states.data(),
                    resumed_edges.data(), host_meta, capacity};
  for (int i = prefix; i < capacity; ++i)
    resumed.append(input[i]);
  assert(std::equal(meta, meta + Automaton::META_FIELDS, host_meta));
  assert(std::equal(states.begin(), states.begin() + meta[1] * Automaton::STATE_FIELDS,
                    resumed_states.begin()));
  assert(std::equal(edges.begin(), edges.begin() + meta[2] * Automaton::EDGE_FIELDS,
                    resumed_edges.begin()));
  assert(resumed_history == input);
  check_index(resumed);
  // Capacity exhaustion is a status, never a write beyond the reserved arrays.
  sa.append(123);
  assert(meta[4] == 0 && meta[0] == capacity);
  assert(history == input);
  // Reuse dirty storage without clearing old buckets or old edges.
  assert(hybrid_sa_build_host(input64.data(), prefix, 2147483647,
                             history.data(), states.data(), edges.data(), meta,
                             capacity) == 0);
  check_index(sa);
}

static void check_large_storage_bounds() {
  const int n = 32768, guard = -98765;
  for (int kind = 0; kind < 4; ++kind) {
    std::vector<int64_t> tokens(n);
    uint32_t random = 17;
    for (int i = 0; i < n; ++i) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      tokens[i] = kind == 0 ? i : kind == 1 ? 42 : kind == 2 ? random % 2
                             : i < n / 2 ? 42 : i;
    }
    std::vector<int> history(n + 1, guard);
    std::vector<int> states((2 * n + 1) * Automaton::STATE_FIELDS + 1, guard);
    std::vector<int> edges((n + 1) * Automaton::EDGE_FIELDS + 1, guard);
    int meta[Automaton::META_FIELDS];
    assert(hybrid_sa_build_host(tokens.data(), n, n + 100, history.data(),
                               states.data(), edges.data(), meta, n) == 0);
    Automaton sa{history.data(), states.data(), edges.data(), meta, n};
    assert(meta[2] <= n - 1);
    if (kind == 0)
      assert(meta[2] == n - 1); // Root fanout reaches the compact pool bound.
    if (kind == 1 || kind == 2)
      assert(meta[2] == 0 && meta[6] == 0); // No edge/hash maintenance at all.
    check_index(sa);
    sa.append(1);
    assert(meta[4] == 0);
    assert(history.back() == guard && states.back() == guard && edges.back() == guard);
  }
}

static void check_external_suffixes() {
  // Exhaustive substring search is independent of SA traversal and catches
  // clone states whose maximum length exceeds the matched context length.
  for (int width = 1; width <= 7; ++width) {
    for (int bits = 0; bits < (1 << width); ++bits) {
      std::vector<int64_t> input(width);
      for (int i = 0; i < width; ++i)
        input[i] = (bits >> i) & 1;
      std::vector<int> history(width), states((2 * width + 1) * Automaton::STATE_FIELDS),
          edges((width + 1) * 6);
      int meta[7];
      assert(hybrid_sa_build_host(input.data(), width, 3, history.data(),
                                 states.data(), edges.data(), meta, width) == 0);
      Automaton sa{history.data(), states.data(), edges.data(), meta, width};
      for (int qwidth = 1; qwidth <= 6; ++qwidth) {
        for (int qbits = 0; qbits < (1 << qwidth); ++qbits) {
          std::vector<int> query(qwidth + 1, 2); // Absent token forces root fallback.
          for (int i = 0; i < qwidth; ++i)
            query[i + 1] = (qbits >> i) & 1;
          int best = 0, end = -1;
          for (int length = std::min(width, qwidth); length > 0 && !best; --length) {
            for (int start = 0; start + length <= width; ++start) {
              if (std::equal(query.end() - length, query.end(), history.begin() + start)) {
                best = length;
                end = start + length - 1;
                break;
              }
            }
          }
          int length = 0;
          int state = sa.match_suffix(query.data(), query.size(), length);
          assert(length == best);
          if (state)
            assert(sa.s(state, 2) == end);
          else
            assert(best == 0);
          // The serving lookup must find a shorter repeat when the longest
          // match ends at the donor boundary and has no continuation.
          best = 0;
          end = -1;
          for (int size = std::min(width - 1, qwidth); size > 0 && !best; --size) {
            for (int start = 0; start + size < width; ++start) {
              if (std::equal(query.end() - size, query.end(), history.begin() + start)) {
                best = size;
                end = start + size - 1;
                break;
              }
            }
          }
          state = sa.match_suffix(query.data(), query.size(), length, true);
          assert(length == best);
          assert((state == 0 && best == 0) || (state > 0 && sa.s(state, 2) == end));
        }
      }
    }
  }
}

int main() {
  check_external_suffixes();
  check_large_storage_bounds();
  for (int width = 1; width <= 12; ++width) {
    for (int bits = 0; bits < (1 << width); ++bits) {
      std::vector<int> tokens(width);
      for (int i = 0; i < width; ++i)
        tokens[i] = (bits >> i) & 1;
      check(tokens);
    }
  }
  uint32_t state = 17;
  for (int test = 0; test < 256; ++test) {
    std::vector<int> tokens(128);
    for (auto &token : tokens) {
      state = state * 1664525u + 1013904223u;
      token = static_cast<int>((state >> 16) % (2 + test % 32));
    }
    check(tokens);
  }
  check(std::vector<int>(512, 42));
  check({2147483646, 0, 2147483646, 5, 0, 2147483646, 0, 5});
  // Force long collision chains, including absent-key probes, across splits.
  std::vector<int> collisions;
  for (int token = 0; collisions.size() < 48; ++token)
    if ((Automaton::hash(0, token) & 255) == 0)
      collisions.push_back(token);
  auto repeat = collisions;
  collisions.insert(collisions.end(), repeat.begin(), repeat.end());
  check(collisions);
  return 0;
}
