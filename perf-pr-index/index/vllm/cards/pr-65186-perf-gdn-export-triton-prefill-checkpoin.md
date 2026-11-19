# PR #65186: [Perf][GDN] Export Triton prefill checkpoints via per-sequence buffers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @specture724 |
| 合并 | 2026-11-17 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, performance, ready, ci/build |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py、vllm/third_party/flash_linear_attention/ops/{chunk.py,chunk_delta_h.py}、tests/kernels/fla/test_chunk_gated_delta_rule.py(新增)、tests/v1/e2e/general/test_mamba_prefix_cache.py 等 |
| 链接 | https://github.com/vllm-project/vllm/pull/65186 |

## 一句话结论

让 Qwen GDN 的 Triton/FLA prefill 导出内部 prefix-cache checkpoint(每序列 buffer + 既有 Mamba checkpoint exporter),align-mode prefill 得以在一个前向中越过最后一个 cache 边界,免去调度器拆分;本 PR 明确不声明量化 serving 加速。

## 动机

align-mode prefix caching 要求在 prefill 内部(cache block 边界)留存状态;此前只有 FlashInfer GDN prefill 能发中间状态,Triton/FLA prefill 依赖调度器在边界处拆分 prefill,多一次调度与前向。设计来自 ZJY0516/vllm#30(mentor fork);与 #60659(AITER FlyDSL,不同 prefill 执行路径)互补,不提供本 FLA kernel 实现。

## 优化点

1. 稠密 `chunk_gated_delta_rule` 只增加可选 `checkpoint_state`(`[N, H, V, K]` 连续 buffer,行 n 存序列 n 在 `checkpoint_offsets[n]` token 后的状态)与 `checkpoint_offsets`;保留原有 initial/final state 契约,kernel 不接触 cache block id。
2. Triton kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 新增 `STORE_CHECKPOINT` constexpr:在 64-token chunk 边界处(仅严格位于序列内部的网格偏移)把 FP32 累加器快照写入 cache-dtype buffer;未传 checkpoint 张量时该 store 编译期消除。
3. `qwen_gdn_linear_attn`:Triton backend 在 CUDA 上宣告内部 checkpoint 能力(`FLA_CHUNK_SIZE` 对齐);checkpoint workspace 为 `max_num_seqs` 行、recurrent-cache dtype,走共享 workspace manager;非对齐/不支持情形保留调度器拆分。
4. 新增 `_forward_triton_prefill_with_checkpoints`:kernel 写每序列连续行,exporter 连同 conv 窗口一起拷入 checkpoint block;`do_not_specialize_on_alignment` 保证前导 decode 行不引入新 kernel 变体。

## 关键改动

```python
# vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
+    if STORE_CHECKPOINT:
+        # The state after o_c tokens is the one entering chunk o_c / BT. Only
+        # offsets on a chunk boundary strictly inside the sequence are stored.
+        o_c = tl.load(checkpoint_offsets + i_n).to(tl.int32)
+        i_c = tl.where((o_c > 0) & (o_c % BT == 0) & (o_c < T), o_c // BT, -1)
+        checkpoint_state = checkpoint_state + i_nh.to(tl.int64) * V * K
```

```python
+        if STORE_CHECKPOINT:
+            if i_t == i_c:
+                # From the fp32 accumulator, unlike h.
+                p_c = checkpoint_state + o_v[:, None] * K + o_k1[None, :]
+                tl.store(p_c, b_h1.to(p_c.dtype.element_ty), mask=m_h1)
```

## 性能声明

PR 正文未提供量化的 serving/kernel 性能数字(perf claims 检测为 none);作者明示:"No real-weight accuracy evaluation or end-to-end TTFT measurement was run for this scope. No quantified serving speedup is claimed." 收益机制为:未拆分的 align-mode prefill 单次前向越过最后一个 cache 边界,减少调度拆分;代价是新增状态存储与 recurrent checkpoint 拷贝流量,"An unsplit prefill can delay other requests' decode steps"。

## 验证与回归风险

- 环境:H20-3e, CUDA, PyTorch 2.14.0+cu132, FlashInfer 0.7.4;基于 upstream `2f70c32d07af8063b632cb67fa160b3fd19ea90b`,测试于 `65e10ceda758f889a5d87a68f20cdbfb6ad26cfe`。
- Kernel/int64 测试:6 passed(checkpoint 内容与切片 reference 精确相等、输出/final state 不变、非法偏移不动 buffer 行、非连续 buffer 拒绝、偏移元数据视图复用已预热 kernel)。
- 既有 MRV2 引擎测试 3 passed(`VLLM_GPU_SYNC_CHECK=error`:同步 Triton/MTP、异步 Triton/MTP、异步 FlashInfer),检查 cache-block 生命周期、prefix-hit 续跑与导出的 conv/recurrent 状态;修改后的异步测试在原始 main 上失败(分配 4 个 block,checkpoint 路径需要 5 个),同运行时同配置下 PR 通过。
- 回归风险:checkpoint workspace 按 `max_num_seqs` 预留带来额外显存与拷贝;未拆分 prefill 可能延迟其他请求的 decode 步;ROCm/XPU 保持既有集成行为,Blackwell 本地不可用未测;仅 kernel 与引擎级测试,无真权重精度评测。

## 关联

设计来源 ZJY0516/vllm#30;相关 #60659(AITER FlyDSL 路线);Buildkite engine step 依赖新增 `vllm/model_executor/layers/mamba/gdn/` 与 `vllm/third_party/flash_linear_attention/ops/`。索引内同区域:GDN/Mamba prefix cache 卡片(pack 未提供具体编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
