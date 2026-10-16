#include "automaton.h"
#include "host_builder.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#ifdef NDEBUG
#error "Native differential tests require active assertions"
#endif

static void check(const std::vector<int> &input) {
  int capacity = static_cast<int>(input.size());
  std::vector<int> history(capacity, -99);
  std::vector<int> states((2 * capacity + 1) * 4, -99);
  std::vector<int> edges((3 * capacity + 1) * 3, -99);
  int meta[5] = {0, 1, 0, 0, 1};
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
  }
  // Build a prompt in compact host storage, copy only used POD prefixes to a
  // larger decode slot and continue appending with the device's state machine.
  int prefix = (capacity + 1) / 2;
  std::vector<int64_t> input64(input.begin(), input.end());
  std::vector<int> host_history(prefix);
  std::vector<int> host_states((2 * prefix + 1) * 4);
  std::vector<int> host_edges((3 * prefix + 1) * 3);
  int host_meta[5];
  int status = hybrid_sa_build_host(input64.data(), prefix, 1000,
                                    host_history.data(), host_states.data(),
                                    host_edges.data(), host_meta, prefix);
  assert(status == 0);
  std::vector<int> resumed_history(capacity), resumed_states(states.size()),
      resumed_edges(edges.size());
  std::copy_n(host_history.begin(), prefix, resumed_history.begin());
  std::copy_n(host_states.begin(), host_meta[1] * 4, resumed_states.begin());
  std::copy_n(host_edges.begin(), host_meta[2] * 3, resumed_edges.begin());
  Automaton resumed{resumed_history.data(), resumed_states.data(),
                    resumed_edges.data(), host_meta, capacity};
  for (int i = prefix; i < capacity; ++i)
    resumed.append(input[i]);
  assert(std::equal(meta, meta + 5, host_meta));
  assert(std::equal(states.begin(), states.begin() + meta[1] * 4,
                    resumed_states.begin()));
  assert(std::equal(edges.begin(), edges.begin() + meta[2] * 3,
                    resumed_edges.begin()));
  assert(resumed_history == input);
  // Capacity exhaustion is a status, never a write beyond the reserved arrays.
  sa.append(123);
  assert(meta[4] == 0 && meta[0] == capacity);
  assert(history == input);
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
  return 0;
}
