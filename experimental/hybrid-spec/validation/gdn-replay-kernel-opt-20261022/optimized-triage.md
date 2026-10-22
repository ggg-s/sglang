Triage View
Mode: single-trace
Framework: SGLang
Input traces: /Users/gq/Desktop/workspces/sglang/experimental/hybrid-spec/validation/gdn-replay-kernel-opt-20261022/traces/sa16-replay-opt/DECODE-c1/DECODE-c1-TP-0.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| _fwd_kernel | other | 228.91 ms | 53.2% | 121 | unresolved | cudaGraphLaunch |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | gemm | 40.19 ms | 9.3% | 849 | unresolved | cudaGraphLaunch |
| void sglang::device::marlin_moe::Marlin<__half, 1125899907892224l, 1125899906910725l, 128, 1, 8, 4, true, 4, 8, false, false, false> | moe | 39.98 ms | 9.3% | 880 | unresolved | cudaGraphLaunch |
| gdn_replayssm_spec_circular_kernel | hybrid_linear | 23.89 ms | 5.6% | 330 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8> | gemm | 9.73 ms | 2.3% | 270 | unresolved | cudaGraphLaunch |
| ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_tn | gemm | 6.97 ms | 1.6% | 382 | unresolved | cudaGraphLaunch |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn | gemm | 6.64 ms | 1.5% | 9 | unresolved | cudaGraphLaunch |
| _causal_conv1d_update_kernel | hybrid_linear | 6.59 ms | 1.5% | 330 | unresolved | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |
