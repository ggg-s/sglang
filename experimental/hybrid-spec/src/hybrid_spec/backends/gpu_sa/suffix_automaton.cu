#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <optional>
#include <torch/extension.h>

#include "automaton.h"

__global__ void reset_kernel(int *states, int *meta, int slot, int h) {
  int *m = meta + slot * Automaton::META_FIELDS;
  m[0] = 0;
  m[1] = 1;
  m[2] = 0;
  m[3] = 0;
  m[4] = 1;
  m[5] = 1;
  m[6] = 0;
  int *root = states + static_cast<int64_t>(slot) * (2 * h + 1) *
                           Automaton::STATE_FIELDS;
  root[0] = 0;
  root[1] = -1;
  root[2] = -1;
  root[3] = root[4] = -1;
}

__global__ void append_kernel(int *history, int *states, int *edges, int *meta,
                              const int64_t *tokens, int count, int slot,
                              int version, int h) {
  Automaton a{history + static_cast<int64_t>(slot) * h,
              states + static_cast<int64_t>(slot) * (2 * h + 1) *
                           Automaton::STATE_FIELDS,
              edges + static_cast<int64_t>(slot) * (h + 1) *
                          Automaton::EDGE_FIELDS,
              meta + slot * Automaton::META_FIELDS, h};
  if (a.meta[0] != version) {
    a.meta[4] = 0;
    return;
  }
  for (int i = 0; i < count && a.meta[4]; ++i) {
    if (tokens[i] < 0 || tokens[i] > INT_MAX) {
      a.meta[4] = 0;
      return;
    }
    a.append(static_cast<int>(tokens[i]));
  }
}

__global__ void query_kernel(const int *history, const int *states,
                             const int *meta, const int64_t *slots,
                             int64_t *tokens, int64_t *valid, int64_t *matches,
                             int64_t *versions, bool *healthy, int bs, int n,
                             int h, int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= bs)
    return;
  valid[row] = matches[row] = versions[row] = 0;
  healthy[row] = false;
  for (int i = 0; i < k; ++i)
    tokens[static_cast<int64_t>(row) * k + i] = 0;
  int64_t slot = slots[row];
  if (slot < 0 || slot >= n)
    return; // No padding row may touch an automaton.
  const int *m = meta + slot * Automaton::META_FIELDS;
  versions[row] = m[0];
  healthy[row] = m[4] != 0;
  if (!m[4] || m[0] == 0)
    return;
  const int *s = states + slot * (2 * h + 1) * Automaton::STATE_FIELDS;
  int suffix = s[Automaton::STATE_FIELDS * m[3] + 1];
  if (suffix <= 0)
    return;
  int start = s[Automaton::STATE_FIELDS * suffix + 2] + 1;
  int count = min(k, m[0] - start);
  matches[row] = s[Automaton::STATE_FIELDS * suffix];
  valid[row] = count;
  for (int i = 0; i < count; ++i)
    tokens[static_cast<int64_t>(row) * k + i] = history[slot * h + start + i];
}

void check_tensor(const torch::Tensor &tensor, torch::ScalarType dtype,
                  const torch::Device &device) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
                  tensor.scalar_type() == dtype && tensor.is_contiguous(),
              "invalid GPU SA tensor device, dtype or layout");
}

void check_storage(torch::Tensor history, torch::Tensor states,
                   torch::Tensor edges, torch::Tensor meta) {
  for (const auto &t : {history, states, edges, meta})
    check_tensor(t, torch::kInt32, history.device());
  TORCH_CHECK(history.dim() == 2 && history.size(0) > 0 &&
                  history.size(1) > 0 && history.size(1) < INT_MAX / 24,
              "invalid history capacity");
  auto n = history.size(0), h = history.size(1);
  TORCH_CHECK(n < INT_MAX / Automaton::META_FIELDS && states.dim() == 3 &&
                  states.size(0) == n &&
                  states.size(1) == 2 * h + 1 &&
                  states.size(2) == Automaton::STATE_FIELDS &&
                  edges.dim() == 3 && edges.size(0) == n &&
                  edges.size(1) == h + 1 &&
                  edges.size(2) == Automaton::EDGE_FIELDS && meta.dim() == 2 &&
                  meta.size(0) == n && meta.size(1) == Automaton::META_FIELDS,
              "invalid automaton storage shape");
}

void reset(torch::Tensor states, torch::Tensor meta, int64_t slot) {
  check_tensor(states, torch::kInt32, states.device());
  check_tensor(meta, torch::kInt32, states.device());
  TORCH_CHECK(states.dim() == 3 && states.size(2) == Automaton::STATE_FIELDS &&
                  states.size(1) >= 3 &&
                  states.size(1) < INT_MAX / Automaton::STATE_FIELDS &&
                  states.size(1) % 2 == 1 &&
                  meta.dim() == 2 && meta.size(0) == states.size(0) &&
                  meta.size(1) == Automaton::META_FIELDS &&
                  meta.size(0) < INT_MAX / Automaton::META_FIELDS && slot >= 0 &&
                  slot < states.size(0),
              "invalid reset slot/storage");
  c10::cuda::CUDAGuard guard(states.device());
  int h = (states.size(1) - 1) / 2;
  reset_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      states.data_ptr<int>(), meta.data_ptr<int>(), slot, h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void append(torch::Tensor history, torch::Tensor states, torch::Tensor edges,
            torch::Tensor meta, torch::Tensor tokens, int64_t slot,
            int64_t version) {
  check_storage(history, states, edges, meta);
  check_tensor(tokens, torch::kInt64, history.device());
  TORCH_CHECK(slot >= 0 && slot < history.size(0) && version >= 0 &&
                  version <= history.size(1) && tokens.dim() == 1 &&
                  tokens.numel() <= history.size(1) - version,
              "invalid history delta");
  c10::cuda::CUDAGuard guard(history.device());
  append_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      history.data_ptr<int>(), states.data_ptr<int>(), edges.data_ptr<int>(),
      meta.data_ptr<int>(), tokens.data_ptr<int64_t>(), tokens.numel(), slot,
      version, history.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> query(torch::Tensor history, torch::Tensor states,
                                 torch::Tensor edges, torch::Tensor meta,
                                 torch::Tensor slots, int64_t k) {
  check_storage(history, states, edges, meta);
  check_tensor(slots, torch::kInt64, history.device());
  TORCH_CHECK(slots.dim() == 1 && slots.numel() < INT_MAX && k > 0 &&
                  k < history.size(1),
              "invalid query shape");
  c10::cuda::CUDAGuard guard(history.device());
  auto bs = slots.numel();
  auto tokens = torch::empty({bs, k}, slots.options());
  auto valid = torch::empty({bs}, slots.options());
  auto matches = torch::empty_like(valid), versions = torch::empty_like(valid);
  auto healthy = torch::empty({bs}, slots.options().dtype(torch::kBool));
  if (bs) {
    query_kernel<<<(bs + 127) / 128, 128, 0,
                   at::cuda::getCurrentCUDAStream()>>>(
        history.data_ptr<int>(), states.data_ptr<int>(), meta.data_ptr<int>(),
        slots.data_ptr<int64_t>(), tokens.data_ptr<int64_t>(),
        valid.data_ptr<int64_t>(), matches.data_ptr<int64_t>(),
        versions.data_ptr<int64_t>(), healthy.data_ptr<bool>(), bs,
        history.size(0), history.size(1), k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {tokens, valid, matches, versions, healthy};
}

// Verify produces int32 predictions, path indices and accept lengths. Only
// these device buffers are consumed; no CPU output history is inspected.
__global__ void feedback_kernel(
    int *meta, const int64_t *owners, int *pending, int64_t *pending_meta,
    const int64_t *slots, const int64_t *generations, const int *predict,
    const int *indices, const int *lengths, const int64_t *seq_lens, int n,
    int width, int index_width, int predict_size, bool has_anchor, int vocab,
    const bool *use_retrieval, int64_t *counts) {
  int row = blockIdx.x;
  int count = lengths[row];
  if (counts != nullptr) {
    // Same six counters as Metrics: requests, correct drafts (without bonus),
    // accept tokens (with bonus), each split into retrieval/neural. Record
    // before slot validation: neural rows without an SA slot still count.
    int source = use_retrieval[row] ? 0 : 1;
    auto counters = reinterpret_cast<unsigned long long *>(counts);
    atomicAdd(counters + source, 1ULL);
    atomicAdd(counters + 2 + source,
              static_cast<unsigned long long>(static_cast<int64_t>(count) - 1));
    atomicAdd(counters + 4 + source, static_cast<unsigned long long>(count));
  }
  int64_t slot = slots[row], generation = generations[row];
  if (slot < 0 || slot >= n || owners[slot] != generation)
    return;
  int64_t *pm = pending_meta + slot * 3;
  // Unconsumed feedback indicates a missing draft or duplicate delivery.
  bool valid = count > 0 && count <= width && count <= index_width && pm[2] == 0;
  for (int j = 0; valid && j < count; ++j) {
    int index = indices[row * index_width + j];
    valid = index >= 0 && index < predict_size;
    if (valid) {
      int token = predict[index];
      valid = token >= 0 && token < vocab;
      pending[slot * width + j] = token;
    }
  }
  pm[0] = generation;
  pm[1] = seq_lens[row] + static_cast<int>(has_anchor);
  pm[2] = valid ? count : -1;
}

__global__ void consume_query_kernel(
    int *history, int *states, int *edges, int *meta, const int64_t *owners,
    const int64_t *limits, const int *pending, int64_t *pending_meta,
    const int64_t *slots, const int64_t *generations, const int64_t *seq_lens,
    int64_t *tokens, bool *mask, int n, int h, int k, int min_match,
    bool zero_partial) {
  int row = blockIdx.x;
  mask[row] = false;
  for (int j = 0; j < k; ++j)
    tokens[row * k + j] = 0;
  int64_t slot = slots[row], generation = generations[row];
  if (slot < 0 || slot >= n || owners[slot] != generation)
    return;
  Automaton a{history + slot * h,
              states + slot * (2 * h + 1) * Automaton::STATE_FIELDS,
              edges + slot * (h + 1) * Automaton::EDGE_FIELDS,
              meta + slot * Automaton::META_FIELDS, h};
  int64_t *pm = pending_meta + slot * 3;
  if (pm[2] != 0) {
    int count = static_cast<int>(pm[2]);
    pm[2] = 0; // Graph replay is idempotent without new verify feedback.
    if (pm[0] != generation || pm[1] != a.meta[0] || count < 0 || count > k + 1)
      a.meta[4] = 0;
    else
      for (int j = 0; j < count && a.meta[4]; ++j)
        a.append(pending[slot * (k + 1) + j]);
  }
  // seq_lens excludes the current bonus anchor; the SA includes it.
  if (!a.meta[4] || a.meta[0] != seq_lens[row] + 1 ||
      static_cast<int64_t>(a.meta[0]) + k + 1 > limits[slot])
    return;
  int suffix = a.s(a.meta[3], 1);
  if (suffix <= 0 || a.s(suffix, 0) < min_match)
    return;
  int start = a.s(suffix, 2) + 1;
  int count = min(k, a.meta[0] - start);
  if (count < k && !zero_partial)
    return;
  for (int j = 0; j < count; ++j)
    tokens[row * k + j] = a.history[start + j];
  mask[row] = true;
}

// Consume all published slots before any cross-request reader runs. Completed
// sources also trim the last verify delta to the CPU-confirmed stop boundary.
// Published slots are unique and generation-fenced; padding never owns a slot.
__global__ void finalize_shared_kernel(
    int *history, int *states, int *edges, int *meta, const int64_t *owners,
    const int *pending, int64_t *pending_meta, const int64_t *slots,
    const int64_t *generations, const int64_t *lengths, int n, int h, int k) {
  int row = blockIdx.x;
  int64_t slot = slots[row], generation = generations[row];
  if (slot < 0 || slot >= n || owners[slot] != generation)
    return;
  Automaton a{history + slot * h,
              states + slot * (2 * h + 1) * Automaton::STATE_FIELDS,
              edges + slot * (h + 1) * Automaton::EDGE_FIELDS,
              meta + slot * Automaton::META_FIELDS, h};
  int64_t *pm = pending_meta + slot * 3;
  int64_t final_length = lengths[row];
  if (final_length >= 0 && final_length < a.meta[0]) {
    // A previous replay already indexed a discarded tail. Do not expose it:
    // this bounded implementation drops the donor instead of rebuilding it.
    a.meta[4] = 0;
    pm[2] = 0;
    return;
  }
  if (pm[2] != 0) {
    int64_t count = pm[2];
    pm[2] = 0;
    if (pm[0] != generation || pm[1] != a.meta[0] || count < 0 || count > k + 1)
      a.meta[4] = 0;
    else {
      if (final_length >= 0 && count > final_length - a.meta[0])
        count = final_length - a.meta[0];
      for (int j = 0; j < count && a.meta[4]; ++j)
        a.append(pending[slot * (k + 1) + j]);
    }
  }
  if (final_length >= 0 && a.meta[0] != final_length)
    a.meta[4] = 0;
}

struct SharedMatch {
  int length, continuation, index, start;

  __device__ bool beats(const SharedMatch &other) const {
    if (length != other.length)
      return length > other.length;
    if (continuation != other.continuation)
      return continuation > other.continuation;
    // Publication order breaks ties, including across threads/strided scans.
    return index < other.index;
  }
};

constexpr int MAX_SHARED_QUERY_THREADS = 128;

__global__ void shared_query_kernel(
    int *history, int *states, int *edges, int *meta, const int64_t *owners,
    const int64_t *limits, const int64_t *slots, const int64_t *generations,
    const int64_t *seq_lens, const int64_t *shared_slots,
    const int64_t *shared_generations, int64_t *tokens, bool *mask,
    int n, int h, int k, int num_shared, int max_context, int min_match,
    bool zero_partial) {
  int row = blockIdx.x;
  int64_t slot = slots[row];
  if (slot < 0 || slot >= n || owners[slot] != generations[row])
    return;
  int *m = meta + slot * Automaton::META_FIELDS;
  if (!m[4] || m[0] != seq_lens[row] + 1 ||
      static_cast<int64_t>(m[0]) + k + 1 > limits[slot])
    return;
  int *s = states + slot * (2 * h + 1) * Automaton::STATE_FIELDS;
  int suffix = s[m[3] * Automaton::STATE_FIELDS + 1];
  int own_length = mask[row] && suffix > 0 ? s[suffix * Automaton::STATE_FIELDS] : 0;
  SharedMatch best{0, 0, INT_MAX, 0};
  int context = min(m[0], max_context);
  // Like TensorRT-LLM's global SA search, each block handles one request and
  // threads search independent, read-only source automata before reducing.
  // Striding also covers pools larger than the bounded block size.
  for (int i = threadIdx.x; i < num_shared; i += blockDim.x) {
    int64_t other = shared_slots[i];
    if (other < 0 || other >= n || other == slot ||
        owners[other] != shared_generations[i])
      continue;
    Automaton a{history + other * h,
                states + other * (2 * h + 1) * Automaton::STATE_FIELDS,
                edges + other * (h + 1) * Automaton::EDGE_FIELDS,
                meta + other * Automaton::META_FIELDS, h};
    if (!a.meta[4] || a.meta[0] == 0)
      continue;
    int length = 0;
    int state = a.match_suffix(history + slot * h + m[0] - context, context, length, true);
    if (state <= 0 || length < min_match || length <= own_length)
      continue;
    int start = a.s(state, 2) + 1;
    int count = min(k, a.meta[0] - start);
    if (count <= 0 || (count < k && !zero_partial))
      continue;
    // Filter partial continuations before ranking. Own history wins equal
    // matches; eligible donors rank by match, uncapped tail, publication order.
    SharedMatch candidate{length, a.meta[0] - start, i, start};
    if (candidate.beats(best))
      best = candidate;
  }
  __shared__ SharedMatch matches[MAX_SHARED_QUERY_THREADS];
  matches[threadIdx.x] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride &&
        matches[threadIdx.x + stride].beats(matches[threadIdx.x]))
      matches[threadIdx.x] = matches[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0 && matches[0].index != INT_MAX) {
    best = matches[0];
    int64_t donor = shared_slots[best.index];
    int count = min(k, best.continuation);
    for (int j = 0; j < k; ++j)
      tokens[row * k + j] = j < count ? history[donor * h + best.start + j] : 0;
    mask[row] = true;
  }
}

void check_device_metadata(torch::Tensor meta, torch::Tensor owners,
                           torch::Tensor pending, torch::Tensor pending_meta,
                           torch::Tensor slots, torch::Tensor generations,
                           torch::Tensor seq_lens) {
  auto device = meta.device();
  check_tensor(meta, torch::kInt32, device);
  check_tensor(pending, torch::kInt32, device);
  for (const auto &t : {owners, pending_meta, slots, generations, seq_lens})
    check_tensor(t, torch::kInt64, device);
  TORCH_CHECK(meta.dim() == 2 && meta.size(1) == Automaton::META_FIELDS &&
                  meta.size(0) < INT_MAX / Automaton::META_FIELDS &&
                  owners.dim() == 1 &&
                  owners.size(0) == meta.size(0) && pending.dim() == 2 &&
                  pending.size(0) == meta.size(0) && pending_meta.dim() == 2 &&
                  pending_meta.size(0) == meta.size(0) && pending_meta.size(1) == 3 &&
                  slots.dim() == 1 && generations.sizes() == slots.sizes() &&
                  seq_lens.sizes() == slots.sizes(),
              "invalid device feedback metadata");
}

void stage_feedback(torch::Tensor meta, torch::Tensor owners,
                    torch::Tensor pending, torch::Tensor pending_meta,
                    torch::Tensor slots, torch::Tensor generations,
                    torch::Tensor predict, torch::Tensor indices,
                    torch::Tensor lengths, torch::Tensor seq_lens,
                    bool has_anchor, int64_t vocab,
                    std::optional<torch::Tensor> use_retrieval,
                    std::optional<torch::Tensor> counts) {
  check_device_metadata(meta, owners, pending, pending_meta, slots, generations,
                        seq_lens);
  for (const auto &t : {predict, indices, lengths})
    check_tensor(t, torch::kInt32, meta.device());
  TORCH_CHECK(predict.dim() == 1 && indices.dim() == 2 &&
                  indices.size(0) == slots.size(0) && lengths.sizes() == slots.sizes() &&
                  vocab > 0 && vocab <= INT_MAX && predict.numel() < INT_MAX,
              "invalid verify feedback");
  TORCH_CHECK(use_retrieval.has_value() == counts.has_value(),
              "feedback metrics require both selection mask and counters");
  if (counts.has_value()) {
    check_tensor(*use_retrieval, torch::kBool, meta.device());
    check_tensor(*counts, torch::kInt64, meta.device());
    TORCH_CHECK(use_retrieval->sizes() == slots.sizes() && counts->dim() == 1 &&
                    counts->numel() == 6,
                "invalid feedback metrics shape");
  }
  c10::cuda::CUDAGuard guard(meta.device());
  if (slots.numel()) {
    feedback_kernel<<<slots.numel(), 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        meta.data_ptr<int>(), owners.data_ptr<int64_t>(), pending.data_ptr<int>(),
        pending_meta.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
        generations.data_ptr<int64_t>(), predict.data_ptr<int>(),
        indices.data_ptr<int>(), lengths.data_ptr<int>(), seq_lens.data_ptr<int64_t>(),
        meta.size(0), pending.size(1), indices.size(1), predict.numel(), has_anchor,
        vocab, use_retrieval ? use_retrieval->data_ptr<bool>() : nullptr,
        counts ? counts->data_ptr<int64_t>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void consume_and_query(
    torch::Tensor history, torch::Tensor states, torch::Tensor edges,
    torch::Tensor meta, torch::Tensor owners, torch::Tensor limits,
    torch::Tensor pending, torch::Tensor pending_meta, torch::Tensor slots,
    torch::Tensor generations, torch::Tensor seq_lens, torch::Tensor tokens,
    torch::Tensor mask, int64_t min_match, bool zero_partial,
    std::optional<torch::Tensor> shared_slots,
    std::optional<torch::Tensor> shared_generations,
    std::optional<torch::Tensor> shared_lengths, int64_t max_context) {
  check_storage(history, states, edges, meta);
  check_device_metadata(meta, owners, pending, pending_meta, slots, generations,
                        seq_lens);
  check_tensor(limits, torch::kInt64, meta.device());
  check_tensor(tokens, torch::kInt64, meta.device());
  check_tensor(mask, torch::kBool, meta.device());
  TORCH_CHECK(limits.sizes() == owners.sizes() && tokens.dim() == 2 &&
                  tokens.size(0) == slots.size(0) && tokens.size(1) > 0 &&
                  tokens.size(1) + 1 == pending.size(1) &&
                  mask.sizes() == slots.sizes() && min_match > 0 && min_match <= INT_MAX,
              "invalid device query outputs");
  c10::cuda::CUDAGuard guard(meta.device());
  TORCH_CHECK(shared_slots.has_value() == shared_generations.has_value() &&
                  shared_slots.has_value() == shared_lengths.has_value(),
              "shared slots, generations and lengths must be provided together");
  if (shared_slots) {
    check_tensor(*shared_slots, torch::kInt64, meta.device());
    check_tensor(*shared_generations, torch::kInt64, meta.device());
    check_tensor(*shared_lengths, torch::kInt64, meta.device());
    TORCH_CHECK(shared_slots->dim() == 1 &&
                    shared_slots->sizes() == shared_generations->sizes() &&
                    shared_slots->sizes() == shared_lengths->sizes() &&
                    shared_slots->numel() <= history.size(0) &&
                    max_context > 0 && max_context <= INT_MAX,
                "invalid shared history metadata");
  }
  if (slots.numel()) {
    if (shared_slots && shared_slots->numel()) {
      finalize_shared_kernel<<<shared_slots->numel(), 1, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
          history.data_ptr<int>(), states.data_ptr<int>(), edges.data_ptr<int>(),
          meta.data_ptr<int>(), owners.data_ptr<int64_t>(), pending.data_ptr<int>(),
          pending_meta.data_ptr<int64_t>(), shared_slots->data_ptr<int64_t>(),
          shared_generations->data_ptr<int64_t>(), shared_lengths->data_ptr<int64_t>(),
          history.size(0),
          history.size(1), tokens.size(1));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    consume_query_kernel<<<slots.numel(), 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        history.data_ptr<int>(), states.data_ptr<int>(), edges.data_ptr<int>(),
        meta.data_ptr<int>(), owners.data_ptr<int64_t>(), limits.data_ptr<int64_t>(),
        pending.data_ptr<int>(), pending_meta.data_ptr<int64_t>(),
        slots.data_ptr<int64_t>(), generations.data_ptr<int64_t>(),
        seq_lens.data_ptr<int64_t>(), tokens.data_ptr<int64_t>(), mask.data_ptr<bool>(),
        history.size(0), history.size(1), tokens.size(1), min_match, zero_partial);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (shared_slots && shared_slots->numel()) {
      int threads = 32;
      while (threads < shared_slots->numel() && threads < MAX_SHARED_QUERY_THREADS)
        threads *= 2;
      shared_query_kernel<<<slots.numel(), threads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
          history.data_ptr<int>(), states.data_ptr<int>(), edges.data_ptr<int>(),
          meta.data_ptr<int>(), owners.data_ptr<int64_t>(), limits.data_ptr<int64_t>(),
          slots.data_ptr<int64_t>(), generations.data_ptr<int64_t>(),
          seq_lens.data_ptr<int64_t>(), shared_slots->data_ptr<int64_t>(),
          shared_generations->data_ptr<int64_t>(), tokens.data_ptr<int64_t>(),
          mask.data_ptr<bool>(), history.size(0), history.size(1), tokens.size(1),
          shared_slots->numel(), max_context, min_match, zero_partial);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reset", &reset);
  m.def("append", &append);
  m.def("query", &query);
  m.def("stage_feedback", &stage_feedback);
  m.def("consume_and_query", &consume_and_query);
}
