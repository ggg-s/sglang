#include "host_builder.h"
#include "automaton.h"
#include <climits>

// Pure CPU ABI, also linked into the CUDA extension. ctypes.CDLL releases the
// GIL while this runs. Keep the layout identical to the device Automaton: all
// links/edges are row-relative indices, so used prefixes need no relocation.
template <class SA>
int build_host(const int64_t *tokens, int count, int vocab, int *history,
               typename SA::StateWord *states, int *edges, int *meta,
               int capacity) {
  if (!history || !states || !edges || !meta || (count && !tokens) ||
      capacity < 1 || capacity >= INT_MAX / 24 || count < 0 ||
      count > capacity || vocab < 1 ||
      (std::is_same<SA, CompactAutomaton>::value &&
       capacity > SA::COMPACT_MAX_HISTORY))
    return 1;
  meta[0] = 0;
  meta[1] = 1;
  meta[2] = 0;
  meta[3] = 0;
  meta[4] = 1;
  meta[5] = 1;
  meta[6] = 0;
  SA a{history, states, edges, meta, capacity};
  a.s(0, 0) = 0;
  a.s(0, 1) = a.s(0, 2) = a.s(0, 3) = a.s(0, 4) = -1;
  a.init_frequency(0);
  for (int i = 0; i < count; ++i) {
    if (tokens[i] < 0 || tokens[i] >= vocab) {
      meta[4] = 0;
      return 1;
    }
    a.append(static_cast<int>(tokens[i]));
    if (!meta[4])
      return 2;
  }
  return 0;
}

extern "C" HYBRID_EXPORT int
hybrid_sa_build_host(const int64_t *tokens, int count, int vocab, int *history,
                     int *states, int *edges, int *meta, int capacity) {
  return build_host<Automaton>(tokens, count, vocab, history, states, edges,
                               meta, capacity);
}

extern "C" HYBRID_EXPORT int
hybrid_sa_build_host_compact(const int64_t *tokens, int count, int vocab,
                             int *history, uint16_t *states, int *edges,
                             int *meta, int capacity) {
  return build_host<CompactAutomaton>(tokens, count, vocab, history, states,
                                      edges, meta, capacity);
}
