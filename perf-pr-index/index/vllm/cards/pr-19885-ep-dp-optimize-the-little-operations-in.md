# PR #19885: [EP+DP] Optimize the little operations in the DeepGEMM + DeepEP low latency case

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tlrmchlsmth |
| 合并 | 2025-06-23 |
| 分类 | MoE |
| Label | performance, ready, qwen, deepseek |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/19885 |

## 一句话结论

优化 DeepGEMM + DeepEP low-latency 解码路径的小算子:fp8 dispatch + 融合 silu-mul-quant kernel,纯 decode 吞吐提升约 2.4x。

## 动机

P/D 分离的 decode worker 使用 DeepGEMM + DeepEP low-latency 组合时,大部分时间消耗在"小算子"上:per-token group fp8 量化、silu-and-mul 激活,以及把 scale 拷贝整理成 DeepGEMM 所需布局的 copy 操作。这些元素级操作夹在两个 GEMM 之间,且为了支持 CUDA graph 需要大量 padding,被 padding token 的无效计算进一步放大。

## 优化点

两块改动:其一,DeepEP LL 的 dispatch 开启 `use_fp8_dispatch`——量化交给 DeepEP 完成,scale 直接落在 DeepGEMM 需要的布局,同时降低 all2all 数据搬运量;启用条件为 quant_dtype 为 fp8 且 `act_quant_block_size[1] == DEEPEP_QUANT_BLOCK_SIZE`(assert 保证 block size 匹配)。其二,新增 Triton 融合核 `silu_mul_fp8_quant_deep_gemm`:输入 `(E, T, 2*H)` 激活,一次完成 silu 前半 × 后半、per-group(默认 128)amax 量化的 fp8 输出,`y_s` 直接以 DeepGEMM 要求的非连续 stride `(T*G, 1, T)` 分配,免去布局拷贝;kernel 以 `(E*G)` 为静态 grid、专家内循环 token,按 `tokens_per_expert` 只处理有效 token、跳过 padding。`BatchedDeepGemmExperts.apply` 中原来的 activation + per_token_group_quant_fp8 两步(含中间 workspace2)被这一个 kernel 替代。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py: 新增 silu_mul_fp8_quant_deep_gemm Triton kernel 并接入 apply
vllm/model_executor/layers/fused_moe/layer.py: DeepEP LL prepare/finalize 启用 use_fp8_dispatch
tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py: 融合核单元测试(新增)
```

```python
y_s = torch.empty_strided((E, T, G),
                          (stride_ys_e, stride_ys_t, stride_ys_g),  # (T*G, 1, T)
                          dtype=torch.float32, device=y.device)
```

## 性能声明

纯 decode serving 基准(`benchmark_serving.py`,deepseek-ai/DeepSeek-R1-0528,random-input-len 1 / output-len 1000,request-rate 512,512 prompts,ignore-eos;GPU 配置未注明):main 输出吞吐 1536.17 tok/s、mean TPOT 331.71 ms、总时长 333.30 s → 本 PR 3690.26 tok/s、TPOT 137.54 ms、138.74 s,即输出吞吐约 2.4x、TPOT -58.5%。质量:lm_eval gsm8k(Qwen/Qwen3-235B-A22B-FP8,`-dp 16 --enable-expert-parallel`,无 enforce-eager,limit 100)flexible 0.91 / strict 0.89(无前后对照)。

## 验证与回归风险

`tests/kernels/moe/test_deepep_deepgemm_moe.py` 保持 green;新增融合核单测按 (E,T,H,group_size) 参数化,与逐 token 参考实现对比 q/scales。风险:fp8 dispatch 改变量化发生位置(DeepEP 侧),要求 DeepEP 的量化 block size 与 DeepGEMM block shape 严格一致(assert);融合核跳过 padding token 的行为依赖 `expert_num_tokens` 正确性;仅作用于 DeepGEMM batched(EP low-latency)路径。

## 关联

同分类(MoE / EP):PR #19500(FP4 MoE 量化内核)、PR #29354(unpermute-aware fused MoE)。属 DeepEP low-latency + DeepGEMM decode 栈的基础优化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
