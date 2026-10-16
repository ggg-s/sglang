#pragma once

#include <cstdint>

#if defined(_WIN32)
#define HYBRID_EXPORT __declspec(dllexport)
#else
#define HYBRID_EXPORT __attribute__((visibility("default")))
#endif

extern "C" HYBRID_EXPORT int hybrid_sa_build_host(
    const int64_t *tokens, int count, int vocab, int *history, int *states,
    int *edges, int *meta, int capacity);
