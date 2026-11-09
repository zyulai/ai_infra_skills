# PR #15347: Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @leejnau |
| 合并 | 2026-01-19 |
| 分类 | Attention / Kernel |
| Label | performance, deepseek, run-ci, nvidia |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 3 个文件(topk.py + 2 个测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/15347 |

## 一句话结论

DeepSeek 系 MoE 的 biased grouped topk 路由换用 flashinfer 的 `fused_topk_deepseek` kernel,kernel 本身约 4x,GSM8K 端到端输出吞吐翻倍。

## 动机

flashinfer 为 DeepSeek V3 提供了专门优化的 routing kernel(flashinfer PR #2099 引入,PR #2181 中 API 改名为 `fused_topk_deepseek`)。SGLang 此前在 `biased_grouped_topk_gpu` 中使用 sgl_kernel 的 `moe_fused_gate`,该 kernel 限制 `num_experts/num_expert_group <= MAX_VPT=32`,且不是 dsv3 场景的最快实现;路由 topk 位于 MoE 前向的必经路径,直接决定 decode TPOT。PR 正文未给出旧 kernel 的具体 profile 瓶颈,只以 profiling 截图对比二者耗时。

## 优化点

在 `biased_grouped_topk_gpu` 最前面插入 flashinfer `fused_topk_deepseek` 分支:满足 flashinfer 约束(`num_fused_shared_experts == 0`、`is_power_of_two(num_experts)`、`topk <= 8`、`topk_group <= num_expert_group` 且 `topk_group * num_expert_group >= topk`、分组时 `experts_per_group <= 32` 且 `experts_per_group * topk_group <= 128`,不分 组时 `num_experts <= 384`)即预分配 `topk_weights`/`topk_ids` 输出张量(flashinfer 原地写入)并调用;`routed_scaling_factor` 由 flashinfer 内部应用,传入 scaling_factor=1.0 或 routed_scaling_factor。不满足约束时保持原 `moe_fused_gate` 回退分支(条件改写为等价的 `experts_per_group <= 32`),`_use_aiter` 等其余路径不变;`expert_location_dispatch_info`/`num_token_non_padded` 场景走 `_biased_grouped_topk_postprocess` 后处理。

## 关键改动

```text
python/sglang/srt/layers/moe/topk.py: biased_grouped_topk_gpu 优先走 fused_topk_deepseek,保留 moe_fused_gate 回退
test/registered/kernels/test_fused_topk_deepseek.py: 新文件,nightly-1-gpu 注册,对比纯 PyTorch 参考实现
test/srt/test_deepseek_v3_mtp.py: 合并 AMD/非 AMD 的 accept-length 阈值为 2.8
```

关键机制(优先级分支 + 输出预分配):

```python
if (_is_cuda and fused_topk_deepseek is not None
        and num_fused_shared_experts == 0
        and is_power_of_two(num_experts)
        and topk <= 8 and topk_group <= num_expert_group ...):
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, ...)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, ...)
    fused_topk_deepseek(gating_output.to(dtype=torch.float32),
                        correction_bias, num_expert_group, topk_group,
                        topk, scaling_factor, topk_weights, topk_ids, True)
```

## 性能声明

PR 正文数据(nvidia/DeepSeek-R1-0528-FP4-V2,TP=8,EP=8,flashinfer_cutlass + modelopt_fp4):

- GSM8K(1316 题,parallel 1316):Output throughput 1009.301 → 2060.780 tok/s(约 **2x**),Latency 143.606 → 70.387 s,Accuracy 0.954 → 0.960。
- bench_serving(random 2048 prompts,in/out 各 1024,max-concurrency 2048):Total token throughput 13146.50 → 13416.34 tok/s(+2.1%),Mean TPOT 36.38 → 35.59 ms。
- kernel 本身:"The kernel itself is approximately 4x faster"(profiling 截图,未给出绝对耗时数值)。

## 验证与回归风险

新增 `test_fused_topk_deepseek.py`:seq_length 1–9 与 16–65536,3 组 expert 配置(128/256/64 experts,含 DeepSeek V3 的 256 experts 配置),对比 `biased_grouped_topk_impl` 参考实现,行和精确匹配 + scatter 允许平局(tie-breaking)差异(最多 `max(16, seq_length//500)` 个元素超差)。风险点:flashinfer 分支与 `moe_fused_gate` 的 tie-breaking 行为不同,测试显式容忍少量差异;`fused_topk_deepseek` 导入失败时静默回退(import try/except),依赖版本需 ≥ 对应 flashinfer;num_fused_shared_experts>0 模型(如带 shared experts 融合配置)不走新路径,无回归但收益为零。

## 关联

上游:flashinfer PR #2099(引入 kernel)、#2181(改名)。同区域 MoE 路由:其他 MoE runner backend PR #30541、#31470、#8036;DeepSeek 路由相关 #14133 等。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
