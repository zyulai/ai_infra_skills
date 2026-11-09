# PR #21734: perf: optimize PCG inductor path for FP8 models

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jasperjiaguo |
| 合并 | 2026-04-14 |
| 分类 | Quantization |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/21734 |

## 一句话结论

优化 FP8 模型的 PCG inductor 编译路径,embedding 场景每请求 GPU 时间 -20%、kernel launch -24%。

## 动机

Piecewise CUDA graph(PCG)搭配 inductor 编译器时,FP8 模型的若干实现细节会阻碍 kernel 融合:QK norm 走不透明的 fused_inplace_qknorm 自定义内核、2D reshape 逼 inductor 生成 as_strided/clone/split_with_sizes 等独立内核、静态 per-tensor FP8 量化走 sgl_kernel 不透明内核,每个 linear 层都多出一次 kernel launch。

## 优化点

两处调用点局部修改,均只在 `piecewise_cuda_graph_compiler == "inductor"` 时生效,通过 `get_global_server_args()` 本地判断而非改全局 `SGLANG_ENABLE_TORCH_COMPILE` 环境变量,避免影响其他路径。其一,`apply_qk_norm`:inductor 下跳过 `fused_inplace_qknorm`,让 inductor 把 QK RMSNorm 与周围算子融合;同时把 `reshape(-1, head_dim)` 改成 `view(*shape[:-1], -1, head_dim)`——旧的 2D reshape 迫使 inductor 生成独立的 as_strided/clone/split_with_sizes 内核,新的 3D view 保持 stride、可与 RMSNorm 干净融合。其二,`apply_fp8_linear`:静态 per-tensor FP8 量化( input_scale 为标量)时改用纯 PyTorch 算子(reciprocal + clamp + cast),inductor 会将其与周围 RMSNorm、residual add 融合,消除每层 linear 的一次独立内核;eager PCG 与 decode 仍用更快的自定义 kernel。

## 关键改动

```text
python/sglang/srt/models/utils.py: apply_qk_norm 在 inductor 下跳过 fused 内核并改用 3D view
python/sglang/srt/layers/quantization/fp8_utils.py: apply_fp8_linear 静态 per-tensor 量化走纯 PyTorch 算子
```

## 性能声明

Embedding:Qwen3-0.6B embeddings、H200、PCG inductor、production traffic 分布、60 RPS——本 PR(FP8)34.38 items/s,对比 main(non-fp8,baseline)27.68 items/s,+24%。Generation:Qwen3-0.6B FP8、H200、PCG inductor、100 输入/30k 输出、10 prompts、request rate 2——本 PR 1025 output tok/s 对 main 1021,decode 无回退。Per-request GPU profile(FP8 embedding,7k tokens):GPU time/req 9.8ms 到 7.8ms(-20%);kernel launches/req 581 到 441(-24%)。消除的内核:fused_qknorm_warp 28 次、triton_poi_fused_as_strided_clone_split_with_sizes_view 54 次、triton_poi_fused_as_strided_view 27 次、per_tensor_quant_fp8_kernel 112 次。

## 验证与回归风险

PR 正文未提及新增测试,以上述 embedding/generation benchmark 与 GPU profile 为验证手段。回归面:改动以 inductor 编译器为开关分支,eager PCG 与 decode 路径显式保持原自定义内核;apply_qk_norm 的 view 改动作用于所有模型的 QK norm 形状变换(语义等价,但捕获模式下 alt_stream 分支同样受影响);fp8_utils 分支仅在静态 per-tensor input_scale 时触发,per-token 动态量化不受影响。

## 关联

同区域:FP8 量化路径(w8a8 block fp8、per-token group quant)与 PCG/torch compile 相关 PR;素材未给出具体编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
