# PR #37338: [Perf] [Bugfix] Fix Triton autotuning in inference for Qwen3.5

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @arpera |
| 合并 | 2026-03-23 |
| 分类 | MoE(major) |
| Label | bug, ready, qwen |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(11 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/37338 |

## 一句话结论

修复 Qwen3.5 GDN Triton warmup 的 dummy tensor dtype 与真实推理不一致的问题,消除推理期 autotuning 重跑,benchmark 总时长快 4.0×。

## 动机

#36599 引入的 GDN Triton warmup(`_warmup_prefill_kernels()`)使用的 dummy tensor dtype 与真实推理不匹配,导致 warmup 生成的 Triton autotuner cache entry 在推理时 key 不匹配、从未被复用。服务 `Qwen/Qwen3.5-397B-A17B-FP8`(-dp 8 --enable-expert-parallel)时,第一个推理 batch 触发 6 个 FLA kernel 共 746 次 autotuning 事件,在线上造成长停顿。

## 优化点

三处 dtype/shape 对齐,使 warmup 产出与推理路径相同的 Triton specialization key:(1) `g`/`beta` 由 `torch.randn(bfloat16)` 改为调用 `fused_gdn_gating()` 生成(推理输出为 float32);(2) `cu_seqlens` 从 `torch.long`(int64)改为 `torch.int32`(与推理一致);(3) `output_final_state` 从 False 改为 True。每个 mismatch 都会产生不同的 kernel specialization key,对齐后全部 autotuning 在启动期完成、推理期为零。另将 10 个 FLA ops 文件中 `cu_seqlens` 的类型注解从 `torch.LongTensor` 更正为 `torch.Tensor`(无性能影响,仅一致性)。

## 关键改动

```text
vllm/model_executor/models/qwen3_next.py: _warmup_prefill_kernels 用 fused_gdn_gating/int32/output_final_state=True
vllm/model_executor/layers/fla/ops/*.py: cu_seqlens 注解 LongTensor -> Tensor(10 个文件)
```

## 性能声明

8× NVIDIA B200(每卡 183 GiB),`Qwen/Qwen3.5-397B-A17B-FP8`,tp1/dp8 + expert parallel,`vllm bench serve` random 8000 input / 1 output、128 prompts、max-concurrency 8、temperature 0,每次运行前清空 Triton cache:benchmark 总时长 109.73 s → 27.34 s(4.0× faster);总 token 吞吐 9,333 → 37,459 tok/s(4.0× higher);mean TTFT 6,821 → 1,670 ms(4.1× lower);median TTFT 929 → 999 ms(基本持平);0 failed requests。autotuning 事件分布:benchmark 期间 746 → 0,总计 1,908 → 1,162(全部在启动 warmup 完成)。

## 验证与回归风险

验证:`TRITON_PRINT_AUTOTUNING=1` 统计 server 启动完成前后的 Autotuning 行数做 A/B;性能表见上。回归风险:warmup 现在额外调用 `fused_gdn_gating`(需 `A_log`/`dt_bias`),`del` 列表相应变化;`output_final_state=True` 使 warmup kernel 写 final state,需 `state` 张量布局与推理一致;改动仅对齐 Qwen3-Next/GDN FLA kernel 的 warmup,Qwen3.5 以外的模型不受影响(注解清理无行为变化)。

## 关联

修复 #36599 引入的 warmup;同区域为 GDN/FLA Triton kernel 与 JIT warmup;pack 正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
