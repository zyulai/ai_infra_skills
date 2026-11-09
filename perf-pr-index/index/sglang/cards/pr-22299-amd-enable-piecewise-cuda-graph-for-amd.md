# PR #22299: [AMD] Enable Piecewise CUDA Graph for AMD GPUs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw |
| 合并 | 2026-06-07 |
| 分类 | Speculative Decoding (tier: major) |
| Label | amd, deepseek, run-ci, piecewise-cuda-graph, bypass-fastfail |
| Diff | 素材未提供 diffstat;diff 涉及 9 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/22299 |

## 一句话结论

在 ROCm/aiter 上启用 Piecewise CUDA Graph:DeepSeek-R1-MXFP4+EAGLE 输出吞吐 +63%,PCG 初始化快 3.6x。

## 动机

PCG(Piecewise CUDA Graph)此前只在 NVIDIA CUDA 路径可用,AMD/ROCm + aiter backend 无法受益;且当 batch 超过 `piecewise_cuda_graph_max_tokens` 捕获范围时,Dynamo 运行时重编译会触发 capture stream 缺失的断言崩溃。本 PR 在保持 CUDA 路径完全不变的前提下,让 PCG 在 AMD 上正确、快速地跑通。

## 优化点

四层修复:其一,`cuda_piecewise_backend` 在 HIP 上检测到 capture stream 未设置(即 capture 外的 Dynamo 运行时重编译)时回退 eager 执行,其他平台保留原断言;超捕获上限的 batch 在 `forward_extend` 中于 `enable_piecewise_cuda_graph()` + `set_forward_context()` 上下文内 eager 运行,保证 Dynamo guards 一致且 PCG 专有代码能取到 layer 对象。其二,aiter MXFP4 的四个 Triton 入口(gemm_afp4wfp4、pre_quant、dynamic_mxfp4_quant、fused_gemm_split_cat)用 `direct_register_custom_op` 包装为带 fake impl 的 custom op,使 Dynamo 可追踪;DeepSeek MLA 双 RadixAttention(attn_mqa/attn_mha)通过 `_pcg_mha_companion` 修正 backend 元数据;capture 前预热 aiter chip info(否则 CUDA graph capture 期间 `rocminfo` 子进程会永久挂起);HIP 上 warmup compile 从逐 token 数 N 次循环改为单次(原循环在 ROCm 上约 30 分钟)。其三,padded-token 正确性:aiter MoE kernel 无法安全处理 `topk_ids=-1`(负索引导致非法内存访问),改为对 padded 区域置零 MoE routing weights(`_zero_topk_weights_padded_region`)并清零 attention 输出的 padding 段,EAGLE verify 批用 `raw_num_tokens` 判定实际 token 数。其四,注册 AMD PCG 测试并修正 `_use_aiter` 初始化。

## 关键改动

```text
python/sglang/srt/compilation/cuda_piecewise_backend.py: HIP 上 capture stream 缺失时回退 eager,避免 Dynamo 重编译崩溃
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py: aiter MXFP4 算子注册为带 fake impl 的 custom op
python/sglang/srt/layers/moe/topk.py: 新增 _zero_topk_weights_padded_region,替代 HIP 上不安全的 topk_ids=-1 掩码
python/sglang/srt/layers/radix_attention.py: MLA MHA 路径 companion 查找 + PCG 回放后清零 padded 输出
python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py: 单次 warmup compile、aiter chip info 预热、replay 传递 raw_num_tokens
```

```python
if _is_hip:
    # On AMD HIP, the aiter MoE kernels do not handle topk_ids=-1 safely
    # (negative indices cause illegal memory access). Instead, zero the
    # routing weights for padded tokens so their MoE output contributes
    # nothing to the hidden state after the weighted sum.
    _zero_topk_weights_padded_region(topk_weights, num_token_non_padded)
```

## 性能声明

DeepSeek-R1-MXFP4(TP4/EP4、aiter、FP8 KV、EAGLE 3 steps/topk 1/4 draft tokens):基线无 PCG 精度 0.950、输出吞吐 1,642 tok/s;PCG + EAGLE(`--piecewise-cuda-graph-max-tokens 8192`)精度 0.954、输出吞吐 2,657-2,677 tok/s(**+63% throughput**);bench_serving(random 3500in/1500out、768 prompts、448 并发)输出 4,222 tok/s、总 14,070 tok/s、TPOT 78.74 ms。无 EAGLE 时 PCG(max-tokens 2048)吞吐 2,019 → 1,898 tok/s(轻微回退)。PCG 初始化(DS-R1-MXFP4)约 6 分钟 → 1 分 40 秒(**3.6x faster**)。Qwen3.5-397B-A17B-MXFP4 gfx950 A/B(TP2、aiter):8k/1k/cc=64 总吞吐 10,670.35 → 11,183.88 tok/s(**+4.81%**)、TPOT +5.93%;1k/1k/cc=8 总吞吐 +0.10%(mean TTFT +12.11%);1k/1k/cc=64 +0.32%;8k/1k/cc=4 总吞吐 -0.39%。精度:DS-R1-MXFP4 PCG 0.941、PCG+EAGLE 0.954、TP8 注册测试 0.946(GSM8K 5-shot 1,319 题,阈值 >= 0.94);Qwen2.5-VL-7B GSM8K 0.820、InternVL2.5-8B MGSM 0.704、Qwen2.5-VL-3B embedding diff 0.0。

## 验证与回归风险

注册测试:`test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py::TestDeepseekR1MXFP4`(加入 PCG flags,2 passed in 404.79s)与 `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py`(新增 AMD CI 注册,2 passed in 190.27s)。风险:改动集中在 HIP 分支,但 `topk.py` 的 HIP 置零逻辑、`quark_w4a4_mxfp4` 的 custom-op 包装对所有 HIP MoE/量化路径生效,若 aiter kernel 行为假设不成立会影响其他 AMD 模型;无 EAGLE 且 MLA 默认 cap 2048 时 PCG 只覆盖小 decode batch,MLA 模型建议调高 max-tokens 至 8192;pack 记录 CI:Base 与 Extra 均失败(:x:)。

## 关联

无正文记录的同区域 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
