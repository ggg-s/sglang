// Capture-only adapter: PyTorch owns the parent and all leaf graphs/pools.
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#if CUDART_VERSION < 12080
#error "conditional SWITCH requires CUDA toolkit >=12.8"
#endif

namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

__global__ void set_switch_handle(cudaGraphConditionalHandle handle,
                                  const int* count, unsigned long long* stats,
                                  const int* live_bs, unsigned int capacity) {
  unsigned int size = static_cast<unsigned int>(*count);
  cudaGraphSetConditional(handle, size);
  if (stats != nullptr && size <= capacity) {
    atomicAdd(stats + size, 1ULL);
    if (size > 0 && size < static_cast<unsigned int>(*live_bs)) {
      atomicAdd(stats + capacity + 1, 1ULL);
    }
  }
}

// Called once inside the parent's torch.cuda.graph capture, never during replay.
void capture_switch(uintptr_t count_ptr, const std::vector<uintptr_t>& leaves,
                    uintptr_t stream_ptr, uintptr_t stats_ptr,
                    uintptr_t live_bs_ptr) {
  if (leaves.empty() || count_ptr == 0) {
    throw std::invalid_argument("SWITCH needs a count and at least one tail");
  }
  if (stats_ptr != 0 && live_bs_ptr == 0) {
    throw std::invalid_argument("SWITCH statistics require the actual batch");
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  const cudaGraphNode_t* dependencies;
  const cudaGraphEdgeData* edge_data;
  size_t num_dependencies;
#if CUDART_VERSION >= 13000
  check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                &dependencies, &edge_data, &num_dependencies));
#else
  check(cudaStreamGetCaptureInfo_v3(stream, &status, nullptr, &graph,
                                   &dependencies, &edge_data, &num_dependencies));
#endif
  if (status != cudaStreamCaptureStatusActive) {
    throw std::runtime_error("SWITCH adapter requires active parent capture");
  }
  cudaGraphConditionalHandle handle;
  check(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  set_switch_handle<<<1, 1, 0, stream>>>(
      handle, reinterpret_cast<const int*>(count_ptr),
      reinterpret_cast<unsigned long long*>(stats_ptr),
      reinterpret_cast<const int*>(live_bs_ptr),
      static_cast<unsigned int>(leaves.size()));
  check(cudaGetLastError());
  // Refresh dependencies after the setter; no body may run before it.
#if CUDART_VERSION >= 13000
  check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                &dependencies, &edge_data, &num_dependencies));
#else
  check(cudaStreamGetCaptureInfo_v3(stream, &status, nullptr, &graph,
                                   &dependencies, &edge_data, &num_dependencies));
#endif
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeSwitch;
  // Case zero is empty. Cases 1..B preserve the exact compact tail sizes.
  params.conditional.size = static_cast<unsigned int>(leaves.size() + 1);
  cudaGraphNode_t node;
#if CUDART_VERSION >= 13000
  check(cudaGraphAddNode(&node, graph, dependencies, edge_data,
                         num_dependencies, &params));
#else
  check(cudaGraphAddNode_v2(&node, graph, dependencies, edge_data,
                            num_dependencies, &params));
#endif
  for (size_t i = 0; i < leaves.size(); ++i) {
    cudaGraphNode_t child;
    check(cudaGraphAddChildGraphNode(
        &child, params.conditional.phGraph_out[i + 1], nullptr, 0,
        reinterpret_cast<cudaGraph_t>(leaves[i])));
  }
  // All later parent work (statistics) must wait for the selected tail.
#if CUDART_VERSION >= 13000
  check(cudaStreamUpdateCaptureDependencies(
      stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
  check(cudaStreamUpdateCaptureDependencies_v2(
      stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
#endif
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("capture_switch", &capture_switch, pybind11::arg("count_ptr"),
             pybind11::arg("leaves"), pybind11::arg("stream_ptr"),
             pybind11::arg("stats_ptr") = 0, pybind11::arg("live_bs_ptr") = 0);
}
