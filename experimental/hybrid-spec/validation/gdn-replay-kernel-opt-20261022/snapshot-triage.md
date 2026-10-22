Triage View
Mode: single-trace
Framework: SGLang
Input traces: /Users/gq/Desktop/workspces/sglang/experimental/hybrid-spec/validation/gdn-replay-kernel-opt-20261022/traces/sa16-snapshot/DECODE-c1/DECODE-c1-TP-0.trace.json.gz

Kernel Table
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| _fwd_kernel | other | 227.90 ms | 47.9% | 121 | unresolved | cudaGraphLaunch |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | gemm | 87.84 ms | 18.5% | 561 | unresolved | cudaGraphLaunch |
| void sglang::device::marlin_moe::Marlin<__half, 1125899907892224l, 1125899906910725l, 128, 1, 8, 4, true, 4, 8, false, false, false> | moe | 27.39 ms | 5.8% | 880 | unresolved | cudaGraphLaunch |
| _fwd_grouped_kernel_stage1 | other | 20.96 ms | 4.4% | 77 | unresolved | cudaGraphLaunch |
| sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas | gemm | 13.23 ms | 2.8% | 1320 | unresolved | cudaGraphLaunch |
| fused_sigmoid_gating_delta_rule_update_kernel | activation | 12.73 ms | 2.7% | 330 | unresolved | cudaGraphLaunch |
| ampere_s16816gemm_fp16_64x64_ldg8_stages_64x5_tn | gemm | 12.43 ms | 2.6% | 330 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_f16_64x64_64x6_tn_align8> | gemm | 8.86 ms | 1.9% | 440 | unresolved | cudaGraphLaunch |
| _causal_conv1d_update_kernel | hybrid_linear | 6.53 ms | 1.4% | 330 | unresolved | cudaGraphLaunch |
| fused_moe_kernel | moe | 4.96 ms | 1.0% | 176 | unresolved | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Qwen-style shared-expert append into routed top-k output | Confirmed | 15.03 ms | 3.2% | fused_sigmoid_gating_delta_rule_update_kernel (2.7%) | unresolved | python/sglang/srt/models/qwen2_moe.py<br>python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py | Split kernels in this family take 3.2% of GPU time. This tree already has a matching path. Qwen-style shared experts can already be appended into routed top-k output in one Triton prep kernel before fused MoE execution. |
