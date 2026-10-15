#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <torch/extension.h>

#include "automaton.h"

__global__ void reset_kernel(int *states, int *meta, int slot, int h) {
  int *m = meta + slot * 5;
  m[0] = 0;
  m[1] = 1;
  m[2] = 0;
  m[3] = 0;
  m[4] = 1;
  int *root = states + static_cast<int64_t>(slot) * (2 * h + 1) * 4;
  root[0] = 0;
  root[1] = -1;
  root[2] = -1;
  root[3] = -1;
}

__global__ void append_kernel(int *history, int *states, int *edges, int *meta,
                              const int64_t *tokens, int count, int slot,
                              int version, int h) {
  Automaton a{history + static_cast<int64_t>(slot) * h,
              states + static_cast<int64_t>(slot) * (2 * h + 1) * 4,
              edges + static_cast<int64_t>(slot) * (3 * h + 1) * 3,
              meta + slot * 5, h};
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
  const int *m = meta + slot * 5;
  versions[row] = m[0];
  healthy[row] = m[4] != 0;
  if (!m[4] || m[0] == 0)
    return;
  const int *s = states + slot * (2 * h + 1) * 4;
  int suffix = s[4 * m[3] + 1];
  if (suffix <= 0)
    return;
  int start = s[4 * suffix + 2] + 1;
  int count = min(k, m[0] - start);
  matches[row] = s[4 * suffix];
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
                  history.size(1) > 0 && history.size(1) < INT_MAX / 12,
              "invalid history capacity");
  auto n = history.size(0), h = history.size(1);
  TORCH_CHECK(n < INT_MAX / 5 && states.dim() == 3 && states.size(0) == n &&
                  states.size(1) == 2 * h + 1 && states.size(2) == 4 &&
                  edges.dim() == 3 && edges.size(0) == n &&
                  edges.size(1) == 3 * h + 1 && edges.size(2) == 3 &&
                  meta.dim() == 2 && meta.size(0) == n && meta.size(1) == 5,
              "invalid automaton storage shape");
}

void reset(torch::Tensor states, torch::Tensor meta, int64_t slot) {
  check_tensor(states, torch::kInt32, states.device());
  check_tensor(meta, torch::kInt32, states.device());
  TORCH_CHECK(states.dim() == 3 && states.size(2) == 4 && states.size(1) >= 3 &&
                  states.size(1) < INT_MAX / 4 && states.size(1) % 2 == 1 &&
                  meta.dim() == 2 && meta.size(0) == states.size(0) &&
                  meta.size(1) == 5 && slot >= 0 && slot < states.size(0),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reset", &reset);
  m.def("append", &append);
  m.def("query", &query);
}
