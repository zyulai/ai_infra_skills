# PR #46340: [Kernel] TD operand loads for batched MoE GEMM (moe_mmk) on XPU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @oonyshch |
| 合并 | 2026-07-26 |
| 分类 | MoE (tier: major) |
| Label | intel-gpu, ready, v1, verified |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,10 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/46340 |

## 一句话结论

给 XPU batched MoE 专家 GEMM(moe_mmk)增加 Tensor-Descriptor 操作数加载:恢复 Xe XMX 2D 块读,单 kernel 延迟 -89~91%,EP 路径 e2e 吞吐 +116%。

## 动机

batched Triton MoE kernel(`moe_mmk`/`expert_triton_kernel`,激活格式 `[E_local, max_tokens, K]`)是低延迟 Expert-Parallel dispatch 使用的路径。在 XPU 上,带 mask 的 `tl.load` 喂给 `tl.dot` 会绕开 Intel Xe XMX 的 2D 块读通路,专家 GEMM 变成逐元素搬运;用 `tl.make_tensor_descriptor` 加载两个操作数可恢复该硬件通路。作者强调 e2e 效果必须在 EP 语境下以 TD-on vs TD-off 度量(此前单卡 GraniteMoE 数字有误导性,已被本 PR 的新数据取代)。

## 优化点

在 `moe_mmk` 增加 `USE_TD` constexpr 分支:操作数经 `tl.make_tensor_descriptor` 以 `[BLOCK_M, BLOCK_K]`/`[BLOCK_N, BLOCK_K]` tile 描述符加载(B 为 `[N, K]`,tile 经 `tl.trans` 转置后进 `tl.dot`),替代带 mask 的指针算术加载。launcher 侧仅在满足条件时启用 TD:triton 支持 tensor descriptor、A/B 的 K 维 stride 为 1(描述符要求末维 compile-time 1)、`K*element_size` 16 字节对齐、三个 BLOCK 为 2 的幂;启用时调用 `set_triton_allocator`。新增 `vllm/triton_utils/tensor_descriptor.py` 实现 `VLLM_TRITON_USE_TD` 三态门控(unset=XPU 自动开、CUDA sm90+ 可选;1/0 强制)。同时:(1) `BatchedTritonExperts` 设备支持扩到 XPU,`kernel config` 新增 `batched_triton` MoE backend,`all2all_utils` 在 XPU 上为此构建本地 `BatchedPrepareAndFinalize`(无 all-to-all);(2) 非量化路径的 dispatch(`BatchedPrepareAndFinalize.prepare`)与 combine(`topk_weight_and_reduce.apply`)从每专家 Python 循环改为向量化 hit-mask + cumsum slot + `index_add_`,把多次同步合并为单次 nonzero sync;(3) `_is_capturing_or_compiling` 修复 XPU 上不可用的 `torch.cuda.is_current_stream_capturing`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py: moe_mmk/expert_triton_kernel/batched_triton_kernel 增加 USE_TD 描述符加载分支与 launcher 启用条件
vllm/triton_utils/tensor_descriptor.py: 新增 use_tensor_descriptor(VLLM_TRITON_USE_TD 三态)
vllm/model_executor/layers/fused_moe/prepare_finalize/batched.py: 非量化 dispatch 向量化(单次 sync)
vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py: combine 向量化 scatter-add
vllm/model_executor/layers/fused_moe/all2all_utils.py, oracle/unquantized.py, config/kernel.py: XPU batched_triton 后端接线
```

关键代码(TD 分支):

```python
if USE_TD:
    a_desc = tl.make_tensor_descriptor(
        a_base_ptr, shape=[M, K], strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(
        b_base_ptr, shape=[N, K], strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K])
...
a = a_desc.load([0, k * BLOCK_K])
b = tl.trans(b_desc.load([0, k * BLOCK_K]))
```

## 性能声明

Per-kernel microbench:Arc B70(E=8, K=N=2048, bf16)TD 延迟 -89~91%;RTX 5060 与 H200(CUDA)-68~70%,收益随专家 GEMM 尺寸扩大;bit-exact。E2E batched 低延迟 EP 路径(2x Intel Arc B70,DeepSeek-V2-Lite,`--all2all-backend naive_low_latency`,bf16,`--enforce-eager`,seed 42,ShareGPT 500 prompts 饱和测试):output tok/s 134.2→290.6(+116%),median TTFT 27.1s→12.0s(-56%),median TPOT 797ms→334ms(-58%)。单流(max-concurrency 1):TTFT 1033→600ms(-42%),TPOT 353→270ms(-24%;no-EP 默认路径为 427ms/124ms)。CUDA 交叉验证(DeepEP low-latency,Qwen3-30B-A3B,4x H200):+72% req/s,-53% TPOT,per-kernel 1.3-6.1x。精度:GraniteMoE GSM8K(8-shot, 200 题)default 0.500 / TD0 0.480 / TD1 0.470,平坦在噪声内。

## 验证与回归风险

`tests/kernels/moe/test_batched_moe.py` 新增 TD vs plain 一致性(多形状)、zero-expert 处理、XPU 设备启用、backend mapping 用例。风险面:改动共享的 batched MoE kernel 与 prepare/finalize/combine 路径(量化路径保留旧循环,非量化路径切换为向量化);CUDA 上 TD 默认关闭(sm90+ 可选),XPU 默认开启;`_is_capturing_or_compiling` 影响所有 XPU 上该文件的 compile/cudagraph 判定。

## 关联

同区域:叠在可移植低延迟 all-to-all EP 后端 #46871 之上;MoE 后端体系(kernel config/oracle)相关 PR 见索引;PR 未注明其他直接前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
