// Compile this same harness against each revision's automaton.h. Allocations
// are excluded. Build and append timings are component costs, not model TPOT.
#include "automaton.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#ifdef HYBRID_SA_LEGACY_LAYOUT
constexpr int EDGE_FIELDS = 3, META_FIELDS = 5;
#else
constexpr int EDGE_FIELDS = Automaton::EDGE_FIELDS;
constexpr int META_FIELDS = Automaton::META_FIELDS;
#endif

int main(int argc, char **argv) {
  if (argc != 5)
    return 1;
  int length = std::atoi(argv[1]), vocab = std::atoi(argv[2]);
  int tail = std::atoi(argv[3]), repeats = std::atoi(argv[4]);
  if (length < 1 || vocab < 1 || tail < 1 || repeats < 1 ||
      static_cast<int64_t>(length) + tail >= 2147483647 / 24)
    return 1;
  int capacity = length + tail;
  std::vector<int> tokens(capacity);
  uint32_t random = 17;
  for (int &token : tokens) {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    token = random % vocab;
  }
  std::unique_ptr<int[]> history(new int[capacity]);
  std::unique_ptr<int[]> states(new int[(2 * capacity + 1) * 4]);
  std::unique_ptr<int[]> edges(new int[(3 * capacity + 1) * EDGE_FIELDS]);
  int meta[META_FIELDS];
  Automaton sa{history.get(), states.get(), edges.get(), meta, capacity};
  using Clock = std::chrono::steady_clock;
  for (int repeat = -1; repeat < repeats; ++repeat) {
    std::fill_n(meta, META_FIELDS, 1);
    meta[0] = meta[2] = meta[3] = 0;
#ifndef HYBRID_SA_LEGACY_LAYOUT
    meta[6] = 0;
#endif
    states[0] = 0;
    states[1] = states[2] = states[3] = -1;
    auto start = Clock::now();
    for (int i = 0; i < length; ++i)
      sa.append(tokens[i]);
    auto built = Clock::now();
    for (int i = length; i < capacity; ++i)
      sa.append(tokens[i]);
    auto done = Clock::now();
    if (!meta[4] || meta[0] != capacity)
      return 2;
    // Hash only the shared semantic layout, so old/new results must agree.
    uint64_t checksum = 1469598103934665603ull;
    for (int i = 0; i < meta[1] * 4; ++i)
      checksum = (checksum ^ static_cast<uint32_t>(states[i])) * 1099511628211ull;
    for (int i = 0; i < meta[2]; ++i)
      for (int field = 0; field < 3; ++field)
        checksum = (checksum ^ static_cast<uint32_t>(sa.e(i, field))) *
                   1099511628211ull;
    if (repeat >= 0)
      std::cout << std::chrono::duration<double, std::micro>(built - start).count()
                << ','
                << std::chrono::duration<double, std::micro>(done - built).count() / tail
                << ',' << checksum << '\n';
  }
}
