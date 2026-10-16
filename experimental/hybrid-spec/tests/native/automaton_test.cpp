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
    for (int edge = sa.s(p, 3); edge != -1; edge = sa.e(edge, 2)) {
      assert(sa.e(edge, 3) == p);
      assert(seen[edge] == (sa.e(sa.s(p, 3), 4) != -2));
      assert(sa.find(p, sa.e(edge, 0)) == edge);
    }
    assert(sa.find(p, -1) == -1);
  }
}

static void check(const std::vector<int> &input) {
  int capacity = static_cast<int>(input.size());
  std::vector<int> history(capacity, -99);
  std::vector<int> states((2 * capacity + 1) * 4, -99);
  std::vector<int> edges((3 * capacity + 1) * Automaton::EDGE_FIELDS, -99);
  int meta[Automaton::META_FIELDS] = {0, 1, 0, 0, 1, 1, 0};
  states[0] = 0;
  states[1] = -1;
  states[2] = -1;
  states[3] = -1;
  Automaton sa{history.data(), states.data(), edges.data(), meta, capacity};
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
    int suffix = states[4 * meta[3] + 1];
    int length = suffix <= 0 ? 0 : states[4 * suffix];
    assert(length == best_length);
    if (length)
      assert(states[4 * suffix + 2] == best_end);
    assert(meta[1] <= 2 * capacity + 1 && meta[2] <= 3 * capacity + 1);
    check_index(sa);
  }
  // Build a prompt in compact host storage, copy only used POD prefixes to a
  // larger decode slot and continue appending with the device's state machine.
  int prefix = (capacity + 1) / 2;
  std::vector<int64_t> input64(input.begin(), input.end());
  std::vector<int> host_history(prefix);
  std::vector<int> host_states((2 * prefix + 1) * 4);
  std::vector<int> host_edges((3 * prefix + 1) * Automaton::EDGE_FIELDS);
  int host_meta[Automaton::META_FIELDS];
  int status = hybrid_sa_build_host(input64.data(), prefix, 2147483647,
                                   host_history.data(), host_states.data(),
                                   host_edges.data(), host_meta, prefix);
  assert(status == 0);
  std::vector<int> resumed_history(capacity), resumed_states(states.size()),
      resumed_edges(edges.size());
  std::copy_n(host_history.begin(), prefix, resumed_history.begin());
  std::copy_n(host_states.begin(), host_meta[1] * 4, resumed_states.begin());
  std::copy_n(host_edges.begin(), host_meta[2] * Automaton::EDGE_FIELDS,
              resumed_edges.begin());
  Automaton resumed{resumed_history.data(), resumed_states.data(),
                    resumed_edges.data(), host_meta, capacity};
  for (int i = prefix; i < capacity; ++i)
    resumed.append(input[i]);
  assert(std::equal(meta, meta + Automaton::META_FIELDS, host_meta));
  assert(std::equal(states.begin(), states.begin() + meta[1] * 4,
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

int main() {
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
