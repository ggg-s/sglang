#include "automaton.h"
#include <cassert>
#include <cstdint>
#include <vector>

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
