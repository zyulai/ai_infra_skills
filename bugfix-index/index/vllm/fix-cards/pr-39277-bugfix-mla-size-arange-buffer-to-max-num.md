# PR #39277: [Bugfix][MLA] Size arange_buffer to max_num_batched_tokens to prevent CUDA IMA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39277](https://github.com/vllm-project/vllm/pull/39277) |
| 作者 | UranusSeven |
| 合并日期 | 2026-04-29 |
| labels | DSv4, bug, nvidia, v1, verified |
| 热度 | 👍 5 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
MLA indexer 的 arange_buffer 分配过小,CUDAGraph dummy run 下越界写引发 CUDA IMA;改为按 max_num_batched_tokens 取较大值。

## 问题现象
正文描述:在数据并行(DP)+ 投机解码(MTP)场景下,MLA indexer 的 decode flattening 路径发生 CUDA illegal memory access(IMA)崩溃,IMA 在下一个同步点异步浮出。复现配置(GLM-5-FP8,8× H200):`--data-parallel-size 4 --tensor-parallel-size 2 --speculative-config.method mtp --speculative-config.num_speculative_tokens 3 --kv-cache-dtype fp8 --gpu-memory-utilization 0.85`。测试结果:修复前服务器在负载下(1000 请求、平均 10K prompt tokens)很快崩溃;修复后同一基准负载稳定运行 85+ 分钟、零错误。

## 根因
正文 Why 节披露完整链条:
1. `execute_dummy_batch` 创建 1 个请求的 dummy batch,但 `_build_attention_metadata` 使用 `num_reqs_padded`(CUDAGraph bucket 大小);
2. `query_start_loc` 只为这 1 个真实请求填充,padded 索引保留上一真实批次的陈旧值;
3. `split_decodes_and_prefills` 把所有 padded 请求判为 decode(投机解码下 `max_query_len <= decode_threshold`);
4. flattening 路径对全部 padded 请求求和,`actual_expanded = sum(decode_lens)` 远超 buffer 大小;
5. `self.arange_buffer[:actual_expanded]` 越界写 → CUDA IMA。
背景:`arange_buffer` 按 `max_num_seqs * next_n` 分配,而同路径的 `expanded_seq_lens_buffer`、`expanded_block_table_buffer` 本就正确使用 `max_num_batched_tokens`。

## 修复方案
改 `vllm/v1/attention/backends/mla/indexer.py` 中 `DeepseekV32IndexerMetadataBuilder.__init__` 的 `arange_buffer` 分配大小,与另两个 flattening buffer 对齐:

```diff
-            scheduler_config.max_num_seqs * next_n,
+            max(
+                scheduler_config.max_num_seqs * next_n,
+                scheduler_config.max_num_batched_tokens,
+            ),
```

## 影响与回归风险
- 回归面:按 diff 实际改动,仅 `DeepseekV32IndexerMetadataBuilder.__init__` 中 `arange_buffer` 的分配尺寸;flattening 与索引计算逻辑未改。buffer 取 `max(...)` 后不小于原值,最多多占少量 GPU 显存,对非 DP/非 MTP 场景行为不变。

## 关联
正文未列出 issue/PR 链接(仅链接作者团队站点 novita.ai)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
