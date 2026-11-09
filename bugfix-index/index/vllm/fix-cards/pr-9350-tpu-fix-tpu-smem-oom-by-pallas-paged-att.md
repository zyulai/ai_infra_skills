# PR #9350: [TPU] Fix TPU SMEM OOM by Pallas paged attention kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9350](https://github.com/vllm-project/vllm/pull/9350) |
| 作者 | WoosukKwon |
| 合并日期 | 2024-10-14 |
| labels | tpu |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
max_model_len 大时 Pallas paged attention 块表撑爆 SMEM;把 batch 维切子批多次调用内核作临时规避。

## 问题现象
当 `max_model_len` 较大时,Pallas paged attention 内核导致 SMEM OOM(解码阶段触发)。正文未附具体报错日志,未描述影响面。

## 根因
正文只说明是 Pallas paged attention 内核在 `max_model_len` 大时引发的 SMEM OOM,未披露更深层机制。diff 内注释补充了机制:PagedAttention Pallas 内核会把整个 block table 存进 SMEM,块表过大时内核编译会失败(由 diff 推断)。

## 修复方案
`vllm/attention/backends/pallas.py` 解码路径:按 `MAX_SMEM_USAGE = 512 * 1024` 与每序列块表占用(4 字节 × 块表列数)算出 `max_num_seq`;batch 不超限时单次调用,超限则把 batch 维切成大小不超过 `max_num_seq`(且向下取偶)的子批,循环多次调用并把结果写回;新增辅助函数 `paged_attention` 封装 megacore_mode 的 None 处理。`vllm/worker/tpu_model_runner.py` 在块表占用 ≥ 512KB 时打日志,建议调小 `--max-model-len`。正文自述此法会降低性能并增加编译时间,是对 SMEM OOM 的临时 workaround。

```diff
+            MAX_SMEM_USAGE = 512 * 1024
+            size_per_seq = 4 * attn_metadata.block_tables.shape[1]
+            max_num_seq = MAX_SMEM_USAGE // size_per_seq
+
+            if batch_size <= max_num_seq:
                 output = paged_attention(
                     query,
                     key_cache,
                     value_cache,
                     attn_metadata.context_lens,
                     attn_metadata.block_tables,
                     pages_per_compute_block,
                     self.megacore_mode,
                 )
+            else:
+                chunk_size = max_num_seq
```

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:TPU 解码注意力执行路径——大 batch 时分块多次调用内核(正文自述性能下降、编译时间增加);megacore_mode 判定逻辑移入新辅助函数;启动时新增块表过大告警。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
