#include "host_builder.h"
#include <chrono>
#include <condition_variable>
#include <mutex>

// A deterministic test-only native boundary. If the Python loader holds the
// GIL, Python cannot release this gate and the probe returns an error instead
// of hanging the test process indefinitely.
static std::mutex lock;
static std::condition_variable changed;
static bool entered = false, released = false;

extern "C" HYBRID_EXPORT int hybrid_sa_probe_wait() {
  std::unique_lock<std::mutex> guard(lock);
  return changed.wait_for(guard, std::chrono::seconds(5), [] { return entered; });
}

extern "C" HYBRID_EXPORT void hybrid_sa_probe_release() {
  std::lock_guard<std::mutex> guard(lock);
  released = true;
  changed.notify_all();
}

extern "C" HYBRID_EXPORT int hybrid_sa_probe_build(
    const int64_t *tokens, int count, int vocab, int *history, int *states,
    int *edges, int *meta, int capacity) {
  {
    std::unique_lock<std::mutex> guard(lock);
    entered = true;
    changed.notify_all();
    if (!changed.wait_for(guard, std::chrono::seconds(5), [] { return released; }))
      return 2;
  }
  return hybrid_sa_build_host(tokens, count, vocab, history, states, edges, meta,
                              capacity);
}
