#include "automaton.h"
#include "host_builder.h"
#include <climits>

// Pure CPU ABI, also linked into the CUDA extension. ctypes.CDLL releases the
// GIL while this runs. Keep the layout identical to the device Automaton: all
// links/edges are row-relative indices, so used prefixes need no relocation.
extern "C" HYBRID_EXPORT int hybrid_sa_build_host(
    const int64_t *tokens, int count, int vocab, int *history, int *states,
    int *edges, int *meta, int capacity) {
  if (!history || !states || !edges || !meta || (count && !tokens) ||
      capacity < 1 || capacity >= INT_MAX / 12 || count < 0 || count > capacity ||
      vocab < 1)
    return 1;
  meta[0] = 0;
  meta[1] = 1;
  meta[2] = 0;
  meta[3] = 0;
  meta[4] = 1;
  states[0] = 0;
  states[1] = -1;
  states[2] = -1;
  states[3] = -1;
  Automaton a{history, states, edges, meta, capacity};
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
