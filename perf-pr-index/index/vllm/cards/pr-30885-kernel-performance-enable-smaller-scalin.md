# PR #30885: [Kernel][Performance] Enable smaller Scaling Factor tiling for NVFP4 small-batch decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-01-13 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready, ci/build, nvidia |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 10 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/30885 |

## 一句话结论

NVFP4 GEMM 小 batch 解码改用 8x4 scaling factor 布局,低并发输出吞吐较现有最优 NVFP4 后端高约 25-35%。

## 动机

NVFP4 量化 GEMM 的 scaling factor 布局受 Tensor Core 约束,最小 tile 为 128x4:激活先按 128 行 round-up pad 再 swizzle。decode 小 batch(如 bs=1)时 M 维远小于 128,padding 浪费与量化开销占比高,导致 NVFP4 在小 batch 下反而不如其他后端。该 PR 引入 opt-in 的 8x4 SF 布局变体,专门针对小并发 decode 场景;明确注明不推荐 medium/large batch 使用(大 batch 下 trtllm_8x4_sf 吞吐显著劣化)。

## 优化点

机制是替换 NVFP4 激活量化的 scale 布局:原路径 `scaled_fp4_quant` 将 scale pad 到 128x4 tile;新路径调用 FlashInfer `nvfp4_quantize` 以 `SfLayout.layout_8x4` 量化,只 pad 到 8 行,scale tile 显著缩小,小 M 下 GEMM 与量化都更高效。选择逻辑自动化:`scaled_fp4_quant` 新增 backend 参数,当 backend 含 "trtllm" 且行数 m ≤ 32 时自动启用 8x4;`flashinfer_scaled_fp4_mm` 同样在 `a.shape[0] <= 32` 时给 FlashInfer `mm_fp4` 传 `use_8x4_sf_layout=True`。量化入口注册为新 custom op `vllm::flashinfer_nvfp4_quantize`(含 fake impl 便于 torch.compile),线性层路径(`compressed_tensors_w4a4_nvfp4` 与 modelopt)把 backend 透传给量化调用;测试侧新增 `convert_swizzled_8x4_layout_to_linear` 用于把 8x4 swizzled scale 还原为 linear 布局做参考验证。启用方式:`export VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm` 且 batch size ≤ 32 时自动选中。

## 关键改动

```text
vllm/utils/flashinfer.py: mm_fp4 增加 use_8x4_sf_layout 参数;新增 flashinfer_nvfp4_quantize custom op(SfLayout.layout_8x4)
vllm/_custom_ops.py: scaled_fp4_quant 增加 backend 参数,trtllm 且 m<=32 时走 8x4 量化
vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py 与 modelopt.py: apply 时透传 backend
tests/kernels/quantization/nvfp4_utils.py: 新增 convert_swizzled_8x4_layout_to_linear 与布局 flag
tests/models/quantization/test_nvfp4.py: 新增三后端 e2e NVFP4 测试(Blackwell CI)
```

## 性能声明

微基准(FlashInfer 实现,全部开启 autotune,量化开销含入 FP4 数字):`meta-llama/Llama-3.1-8B-Instruct`(N=28672, K=4096,TFLOP/s)bs=1:trtllm_8x4_sf 20.42 vs cutlass-nvfp4 13.45、trtllm-nvfp4 8.32、cudnn-nvfp4 7.80、fbgemm 12.88、torch-bf16 5.97;bs=2/4/8/16 分别为 41.60/83.56/164.95/322.11,均为各列最优,bs=32 起 369.62 被反超,大 batch 严重劣化(bs=4096 仅 574.34)。`meta-llama/Llama-3.3-70B-Instruct`(N=8192, K=8192)bs=1:14.08 vs cutlass 7.92,bs≤16 最优。E2E(B200、TP=1、--max-concurrency=1):Llama-3.3-70B-NVFP4 输出吞吐 83.89 tok/s、TPOT 11.91 ms(flashinfer-cutlass 67.13/14.89,FP8 参考 64.59/15.48);Llama-3.1-8B-NVFP4 327.69 tok/s、TPOT 3.04 ms(cutlass 240.32/4.15,FP8 283.36)。总结:较当前最优 NVFP4 基线(flashinfer-cutlass)小 batch/低并发输出吞吐高约 25-35%,较 FP8 基线高约 15-30%。

## 验证与回归风险

测试:扩展 `tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py`(新增 (1,128,128) 与 pad 形状 (2,128,64)、(3,128,96);trtllm_8x4_sf 后端含 8x4 布局的参考反量化对照);新增 `tests/models/quantization/test_nvfp4.py::test_nvfp4` e2e 测试(Llama-3.1-8B-NVFP4,cudnn/trtllm/cutlass 三后端 x eager/graph,greedy 输出断言),加入 Blackwell CI(Blackwell Test 时长标注从 21 调到 23 分钟)。风险面:仅当 backend=flashinfer-trtllm 且 m<=32 时切换布局,其他后端与大 batch 路径不变;但 `scaled_fp4_quant` 签名变更和 `flashinfer_mm_fp4` 增加 use_8x4_sf_layout 参数影响所有调用点,且行为依赖 flashinfer 版本提供 `nvfp4_quantize`/`SfLayout`;bs=32 附近 trtllm_8x4_sf 已开始劣化,自动选择阈值固定为 32。

## 关联

同区域:NVFP4 GEMM 后端族 flashinfer-cudnn / flashinfer-trtllm / flashinfer-cutlass / cutlass(VLLM_NVFP4_GEMM_BACKEND 选择,envs.py);pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
