#include "automaton.h"
#include "host_builder.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <random>
#include <vector>

#ifdef NDEBUG
#error "Frequency tests require active assertions"
#endif

static int count(const std::vector<int> &text,
                 const std::vector<int> &pattern) {
  int result = 0;
  for (int i = 0; i + pattern.size() <= text.size(); ++i)
    result += std::equal(pattern.begin(), pattern.end(), text.begin() + i);
  return result;
}

template <class SA> static void check(const std::vector<int> &input) {
  using Word = typename SA::StateWord;
  auto build = [](const int64_t *tokens, int count, int *history, Word *states,
                  int *edges, int *meta, int capacity) {
    if constexpr (std::is_same<SA, CompactAutomaton>::value)
      return hybrid_sa_build_host_compact(tokens, count, 100, history, states,
                                          edges, meta, capacity);
    else
      return hybrid_sa_build_host(tokens, count, 100, history, states, edges,
                                  meta, capacity);
  };
  int capacity = input.size();
  std::vector<int> history(capacity), edges((capacity + 1) * SA::EDGE_FIELDS);
  std::vector<Word> states((2 * capacity + 1) * SA::STORAGE_FIELDS);
  int meta[Automaton::META_FIELDS];
  assert(build(nullptr, 0, history.data(), states.data(), edges.data(), meta,
               capacity) == 0);
  SA sa{history.data(), states.data(), edges.data(), meta, capacity};
  std::vector<int> text;
  for (int token : input) {
    sa.append(token);
    text.push_back(token);
    assert(meta[4]);
    auto before = states;
    for (int state = 1; state < meta[1]; ++state) {
      int length = sa.s(state, 0), end = sa.s(state, 2);
      std::vector<int> pattern(text.begin() + end - length + 1,
                               text.begin() + end + 1);
      assert(sa.occurrences(state) == count(text, pattern));
      int shortest = sa.s(sa.s(state, 1), 0) + 1;
      assert(sa.occurrences(state) ==
             count(text, {text.begin() + end - shortest + 1,
                          text.begin() + end + 1}));
      // Independently choose each next token by counting complete substrings.
      std::vector<int64_t> expected, actual(4);
      for (int j = 0; j < 4; ++j) {
        std::map<int, int> frequencies;
        for (int i = 0; i + pattern.size() < text.size(); ++i)
          if (std::equal(pattern.begin(), pattern.end(), text.begin() + i))
            ++frequencies[text[i + pattern.size()]];
        int best = -1, best_count = 0;
        for (auto entry : frequencies)
          if (entry.second > best_count) {
            best = entry.first;
            best_count = entry.second;
          }
        if (best < 0)
          break;
        expected.push_back(best);
        pattern.push_back(best);
      }
      int valid = sa.draft(state, 4, actual.data(), true);
      actual.resize(valid);
      assert(actual == expected);
    }
    // A shared donor lookup must never mutate lazy tags or tree pointers.
    assert(before == states);
  }
  // Host-built prompt state can be relocated, then updated on the decode side.
  int prefix = std::max(1, capacity / 2);
  std::vector<int64_t> tokens(input.begin(), input.end());
  std::vector<int> small_history(prefix),
      small_edges((prefix + 1) * SA::EDGE_FIELDS);
  std::vector<Word> small_states((2 * prefix + 1) * SA::STORAGE_FIELDS);
  assert(build(tokens.data(), prefix, small_history.data(), small_states.data(),
               small_edges.data(), meta, prefix) == 0);
  std::copy_n(small_history.begin(), prefix, history.begin());
  std::copy_n(small_states.begin(), meta[1] * SA::STORAGE_FIELDS,
              states.begin());
  std::copy_n(small_edges.begin(), meta[2] * Automaton::EDGE_FIELDS,
              edges.begin());
  for (int i = prefix; i < capacity; ++i)
    sa.append(input[i]);
  for (int state = 1; state < meta[1]; ++state) {
    int end = sa.s(state, 2), length = sa.s(state, 0);
    assert(sa.occurrences(state) ==
           count(input,
                 {input.begin() + end - length + 1, input.begin() + end + 1}));
  }
}

static void check_compact_boundary() {
  constexpr int h = CompactAutomaton::COMPACT_MAX_HISTORY;
  for (int alphabet : {1, 2, 97}) {
    std::mt19937 rng(8137);
    std::vector<int64_t> tokens(h);
    for (auto &token : tokens)
      token = rng() % alphabet;
    std::vector<int> wh(h), ch(h), ws((2 * h + 1) * 10), we((h + 1) * 6),
        ce((h + 1) * 6);
    constexpr uint16_t guard = 0xdead;
    std::vector<uint16_t> cs((2 * h + 1) * 11 + 1, guard);
    int wm[7], cm[7];
    int prefix = h - 32;
    assert(hybrid_sa_build_host(tokens.data(), prefix, 100, wh.data(),
                                ws.data(), we.data(), wm, h) == 0);
    assert(hybrid_sa_build_host_compact(tokens.data(), prefix, 100, ch.data(),
                                        cs.data(), ce.data(), cm, h) == 0);
    Automaton wide{wh.data(), ws.data(), we.data(), wm, h};
    CompactAutomaton compact{ch.data(), cs.data(), ce.data(), cm, h};
    for (int i = prefix; i < h; ++i) {
      wide.append(tokens[i]);
      compact.append(tokens[i]);
    }
    assert(std::equal(wm, wm + 7, cm) && wm[4] == 1);
    if (alphabet == 2)
      assert(wm[1] > 32768); // Exercise the unsigned high bit of state IDs.
    for (int p = 0; p < wm[1]; ++p) {
      for (int f = 0; f < 10; ++f)
        assert(int(wide.s(p, f)) == int(compact.s(p, f)));
      assert(wide.occurrences(p) == compact.occurrences(p));
    }
    assert(cs.back() == guard);
    if (alphabet == 1) {
      // Independently known exact counts at the largest compact capacity.
      for (int p : {0, 1, 16384, h})
        assert(compact.occurrences(p) == (p == 0 ? h : h - p + 1));
    }
    // Largest allocated ID and sentinel, plus full signed secondary values.
    for (int f : {1, 2, 3, 5, 6, 7}) {
      compact.s(2 * h, f) = 2 * h;
      assert(int(compact.s(2 * h, f)) == 65534);
      compact.s(2 * h, f) = -1;
      assert(int(compact.s(2 * h, f)) == -1);
    }
    for (int value : {INT32_MIN, -65535, -1, 0, INT32_MAX}) {
      compact.s(2 * h, 4) = value;
      assert(int(compact.s(2 * h, 4)) == value);
    }
    assert(hybrid_sa_build_host_compact(nullptr, 0, 100, ch.data(), cs.data(),
                                        ce.data(), cm, h + 1) == 1);
    assert(cs.back() == guard);
  }
}

int main() {
  check_compact_boundary();
  for (int width = 1; width <= 8; ++width)
    for (int bits = 0; bits < (1 << width); ++bits) {
      std::vector<int> input(width);
      for (int i = 0; i < width; ++i)
        input[i] = (bits >> i) & 1;
      check<Automaton>(input);
      check<CompactAutomaton>(input);
    }
  std::mt19937 random(71);
  for (int run = 0; run < 80; ++run) {
    std::vector<int> input(64);
    for (int &token : input)
      token = random() % (2 + run % 12);
    check<Automaton>(input);
    check<CompactAutomaton>(input);
  }
  check<Automaton>(std::vector<int>(96, 42));
  check<CompactAutomaton>(std::vector<int>(96, 42));
  check<Automaton>({1, 2, 8, 1, 2, 9, 1, 2, 9, 1, 2});
  check<CompactAutomaton>({1, 2, 8, 1, 2, 9, 1, 2, 9, 1, 2});
}
