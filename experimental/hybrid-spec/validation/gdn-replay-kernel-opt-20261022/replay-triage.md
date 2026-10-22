Triage View
Mode: single-trace
Framework: SGLang
Input traces: /Users/gq/Desktop/workspces/sglang/experimental/hybrid-spec/validation/gdn-replay-kernel-opt-20261022/traces/sa16-replay/DECODE-c1/DECODE-c1-TP-0.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| _fwd_kernel | other | 227.55 ms | 42.0% | 121 | unresolved | cudaGraphLaunch |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | gemm | 87.89 ms | 16.2% | 561 | unresolved | cudaGraphLaunch |
| gdn_replayssm_spec_circular_kernel | hybrid_linear | 79.71 ms | 14.7% | 330 | unresolved | cudaGraphLaunch |
| void sglang::device::marlin_moe::Marlin<__half, 1125899907892224l, 1125899906910725l, 128, 1, 8, 4, true, 4, 8, false, false, false> | moe | 27.40 ms | 5.1% | 880 | unresolved | cudaGraphLaunch |
| _fwd_grouped_kernel_stage1 | other | 21.03 ms | 3.9% | 77 | unresolved | cudaGraphLaunch |
| sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas | gemm | 13.03 ms | 2.4% | 1320 | unresolved | cudaGraphLaunch |
| ampere_s16816gemm_fp16_64x64_ldg8_stages_64x5_tn | gemm | 12.39 ms | 2.3% | 330 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_f16_64x64_64x6_tn_align8> | gemm | 7.90 ms | 1.5% | 440 | unresolved | cudaGraphLaunch |
| _causal_conv1d_update_kernel | hybrid_linear | 6.53 ms | 1.2% | 330 | unresolved | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |
