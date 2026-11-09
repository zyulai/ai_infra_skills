# PR #50932: buffer size insuffient Dspark sd for FlashInfer MNNVL allreduce

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @khushali9 |
| 合并 | 2026-08-27 |
| 分类 | Attention / Kernel (tier: major) |
| Label | bug, ready, nvidia, dflash |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/50932 |

## 一句话结论

修复 MNNVL FlashInfer allreduce 的 token 容量门控算错 3 倍的问题,消除 spec decode 下 49–146 token 张量在内核中 abort 的崩溃。

## 动机

`FlashInferAllReduce.should_use_fi_ar` 用整个 MNNVL workspace 分配(`max_workspace_size // (hidden_dim * element_size)`,TP8 时为 2 MB,折算约 146 token)做门控。但 MNNVL 后端是 Lamport-based,轮转使用 3 个 buffer(`NUM_LAMPORT_BUFFERS=3`),单次 allreduce 实际只有约 1/3 预算可用(per-buffer 698,368 字节,真实容量约 48 token)。任何 49–146 token 的张量都会通过门控后在内核内 abort。spec decode(dspark draft)的 drafter 每步产生 B × (1 + num_speculative_tokens) 个 token——例如 14 请求 × 4 = 56 token——恰好落入该区间;无 drafter 时 decode 每步仅 B 个 token,低于 48,问题只在开 spec decode 时暴露(fixes #50877)。

## 优化点

保留原有预算上界作为廉价预过滤,之后改为直接调用 workspace 抽象基类自带的 `is_buffer_size_sufficient(tp_size, num_tokens, hidden_dim, dtype)`,由 workspace 自己按后端真实 buffer 容量拒绝内核放不下的尺寸,而不是在 vLLM 侧重新实现 Lamport 算术。该方法在 flashinfer-python==0.6.15.post1(pinned 版本)的抽象基类上,对 mnnvl 与 trtllm 两种后端都正确,因而对后端内部 buffer 策略变化是版本稳健的。同一缺陷也存在于 `fused_allreduce_gemma_rms_norm._can_use_flashinfer`(其 docstring 自述 "mirrors FlashInferAllReduce"),一并修复。

## 关键改动

```text
vllm/distributed/device_communicators/flashinfer_all_reduce.py: should_use_fi_ar 末尾追加 workspace.is_buffer_size_sufficient 检查
vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py: _can_use_flashinfer 同样追加 is_buffer_size_sufficient 检查
```

关键摘录(flashinfer_all_reduce.py 注释说明根因):

```python
# The token bound above uses the full allocation budget, but mnnvl's
# Lamport buffers rotate through three slots, so only ~1/3 is usable per
# call. Asking the workspace directly to reject sizes the kernel can't fit.
return workspace.is_buffer_size_sufficient(
    tp_size=self.world_size,
    num_tokens=num_tokens,
    hidden_dim=hidden_dim,
    dtype=input_tensor.dtype,
)
```

## 性能声明

无性能数字声明;本 PR 为崩溃/正确性修复。PR 未注明测试条件下的吞吐数据。

## 验证与回归风险

作者新增了测试文件并运行通过;同时跑了已有测试 `pytest tests/compile/passes/distributed/test_fusion_all_reduce.py -v`。fix 在 GB300 上由 @janbernloehr 验证;GB200 上作者无硬件未实测,给出复现命令(deepseek-v4 + dspark draft,TP8 起 api_server),请 issue 报告者验证。明确未修复:`AllReduceFusionPass` 存在同样的过松上界,但其中 `max_token_num` 同时驱动 workspace 分配请求,需拆分两个关注点并在 GB200 上验证 one-shot/two-shot 切分,留作 follow-up。可用 `VLLM_ALLREDUCE_USE_FLASHINFER=0` 绕开 eager 的 FlashInferAllReduce 路径(编译 fusion pass 不受影响)。影响路径:MNNVL/TRT-LLM 后端下 TP8 allreduce 与 fused allreduce+Gemma RMSNorm 融合的准入判定;被拒绝的尺寸会回退到普通 allreduce 路径。

## 关联

fixes issue #50877;follow-up 方向为 AllReduceFusionPass 的同类上界问题。本卡所在批次素材中未列出其他同区域 PR,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
