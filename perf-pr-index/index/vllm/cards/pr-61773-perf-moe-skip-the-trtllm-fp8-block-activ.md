# PR #61773: [Perf][MoE] Skip the TRTLLM fp8-block activation-scale transpose copy

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-10-15 |
| 分类 | MoE(tier: major) |
| Label | nvidia |
| Diff | +77/-14,5 files(PR 正文注明;diff 与之一致) |
| 链接 | https://github.com/vllm-project/vllm/pull/61773 |

## 一句话结论

让 per-token-group FP8 量化直接按 TRTLLM fp8-block MoE 期望的列主序布局输出 activation scale,消除每层每步一次 transpose 拷贝,TPOT 约降 2%。

## 动机

TRTLLM fp8-block MoE expert kernel 消费的 activation scale 是物理 `[K/128, M]`(scale-major)布局,而 MoE 的 per-token-group 量化输出 `[M, K/128]` 行主序,于是每层每个 decode 步都要在量化与 expert 之间做一次 `a1q_scale.t().contiguous()` 转置拷贝(实际是 `direct_copy_kernel`)。Qwen3.8-Flash-Next 有 24 个 MoE 层,即每步 24 次纯 plumbing 拷贝占用关键路径。

## 优化点

零内核改动,纯 Python plumbing:per-token-group 量化 kernel 本就有 `column_major_scales=True` 模式——把 scale buffer 直接按连续 `[G, M]` 分配、以 `[M, G]` 视图返回,此时 expert 侧的 `.t().contiguous()` 变成 no-op、返回同一张量。实现上给 `FusedMoEQuantConfig` 新增 `is_scale_transposed` 标志,由 fp8 oracle 仅在 `FLASHINFER_TRTLLM` 后端 + `[128,128]` block 组合时置 True,经 `no_dp_ep` 的 `_quantize_input` 传入 `moe_kernel_quantize_input` → `_fp8_quantize` → `per_token_group_quant_fp8(column_major_scales=...)`。TRTLLM expert 代码不动;scale 字节值完全相同,只是物理布局不同,专家输入 bitwise 一致。dispatch 路径(deepep、naive-dp、fi-nvl)因 scale 要跨 all2all 传递,保持行主序;其他 MoE 后端不受影响。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/oracle/fp8.py: oracle 仅对 FLASHINFER_TRTLLM+[128,128] 置 is_scale_transposed
vllm/model_executor/layers/fused_moe/utils.py: _fp8_quantize 透传 column_major_scales 给 per_token_group_quant_fp8
vllm/model_executor/layers/fused_moe/config.py: FusedMoEQuantConfig 新增 is_scale_transposed 字段
vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py: 把标志传入量化入口
```

## 性能声明

Qwen3.8-Flash-Next-FP8,TP4+EP,4×GB300,`vllm bench serve`,2048 input / 256 output、`--ignore-eos`,main 与 branch 顺序跑在同一节点、各两个 server 实例:并发 4 TPOT p50 5.11-5.12 ms → 4.98-5.00 ms(**-2.3%**),TPOT p99 5.67-6.04 → 5.57-5.58 ms;并发 1 ~0%(c=1 用交错 `vllm bench latency` bs=1、2048/128 验证:0.756/0.704 s vs 0.756/0.800 s,无回归)。Profiler 证据:每 decode 步 `direct_copy_kernel` 从 24 次降为 0。

## 验证与回归风险

`pytest tests/kernels/moe/test_flashinfer.py` 49 passed(+2 mxfp8 skips);新增 `test_trtllm_fp8_block_moe_deferred_finalize` 的 `transposed` 分支:走生产 oracle + `no_dp_ep` 路径构造 quant config,断言 `a1q_scale.t().is_contiguous()`、与行主序基线输出 bitwise 相等(atol=0, rtol=0),并断言 DEEPGEMM 等其他后端不置该标志。`column_major_scales` 模式本身无新测试(既有 CUTLASS/DeepGEMM/FlashInfer/PyTorch scaled-mm 每次 CUDA run 都在覆盖)。风险点:仅 no_dp_ep 路径生效;若未来其他 fp8-block 后端也需列主序需单独在 oracle 开启;无独立 accuracy eval(以 bitwise 一致性论证)。

## 关联

PR 自查过相邻 open PR:#59564(quant kernel 内 value-only amax reduction,正交)、#53345(deepep ep_scatter scale-stride bugfix)、#44034(XPU quant fallback),均不触碰 MoE activation-scale 布局。同区域 FP8 MoE 优化另见 #28124(DS sigmoid+bias fusion)。
