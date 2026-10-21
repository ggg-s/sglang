// Compile this same harness against each revision's automaton.h. Allocations
// are excluded. Build and append timings are component costs, not model TPOT.
#include "automaton.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#ifdef HYBRID_SA_LEGACY_LAYOUT
constexpr int EDGE_FIELDS = 3, META_FIELDS = 5;
#else
constexpr int EDGE_FIELDS = Automaton::EDGE_FIELDS;
constexpr int META_FIELDS = Automaton::META_FIELDS;
#endif

#ifdef HYBRID_SA_COMPACT_LAYOUT
constexpr int STATE_FIELDS = Automaton::STATE_FIELDS;
constexpr int EDGE_CAPACITY_FACTOR = Automaton::EDGE_CAPACITY_FACTOR;
#else
constexpr int STATE_FIELDS = 4, EDGE_CAPACITY_FACTOR = 3;
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
  std::unique_ptr<int[]> states(new int[(2 * capacity + 1) * STATE_FIELDS]);
  std::unique_ptr<int[]> edges(new int[(EDGE_CAPACITY_FACTOR * capacity + 1) * EDGE_FIELDS]);
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
#ifdef HYBRID_SA_COMPACT_LAYOUT
    states[4] = -1;
#endif
#ifdef HYBRID_SA_FREQUENCY_LAYOUT
    sa.init_frequency(0);
#endif
    auto start = Clock::now();
    for (int i = 0; i < length; ++i)
      sa.append(tokens[i]);
    auto built = Clock::now();
    for (int i = length; i < capacity; ++i)
      sa.append(tokens[i]);
    auto done = Clock::now();
    if (!meta[4] || meta[0] != capacity)
      return 2;
    // Canonical graph checksum: independent of inline/list storage and ordering.
    uint64_t checksum = 1469598103934665603ull;
    auto mix = [&](int value) {
      checksum = (checksum ^ static_cast<uint32_t>(value)) * 1099511628211ull;
    };
    mix(meta[0]);
    mix(meta[1]);
    mix(meta[3]);
    for (int p = 0; p < meta[1]; ++p) {
      for (int field = 0; field < 3; ++field)
        mix(sa.s(p, field));
      std::vector<std::pair<int, int>> outgoing;
#ifdef HYBRID_SA_COMPACT_LAYOUT
      if (sa.s(p, 3) != -1)
        outgoing.emplace_back(history[sa.s(p, 2) + 1], sa.s(p, 3));
      int head = sa.s(p, 4);
      if (head < -1) {
        outgoing.emplace_back(history[sa.s(~head, 2)], ~head);
        head = -1;
      }
#else
      int head = sa.s(p, 3);
#endif
      for (int edge = head; edge != -1; edge = sa.e(edge, 2))
        outgoing.emplace_back(sa.e(edge, 0), sa.e(edge, 1));
      std::sort(outgoing.begin(), outgoing.end());
      mix(static_cast<int>(outgoing.size()));
      for (auto [token, target] : outgoing) {
        mix(token);
        mix(target);
      }
    }
    if (repeat >= 0)
      std::cout << std::chrono::duration<double, std::micro>(built - start).count()
                << ','
                << std::chrono::duration<double, std::micro>(done - built).count() / tail
                << ',' << checksum << '\n';
  }
}
