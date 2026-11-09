# PR #9532: [Bugfix] Fix `illegal memory access` error with chunked prefill, prefix caching, block manager v2 and xformers enabled together

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9532](https://github.com/vllm-project/vllm/pull/9532) |
| 作者 | sasha0552 |
| 合并日期 | 2024-10-31 |
| labels | ready, ci/build |
| 热度 | 👍 1 · 💬 21 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
chunked prefill + prefix caching + block manager v2 + XFormers 同开时,特定请求序列触发 CUDA illegal memory access;修复 prefix cache 命中时传给 kernel 的 block table 构造。

## 问题现象
对特定的请求序列,同时开启 1) chunked prefill、2) prefix caching、3) block manager V2、4) XFormers 的 vLLM 会以 `CUDA error: an illegal memory access was encountered` 崩溃(有时发生在 prefix caching kernel 里)。崩溃栈显示错误从 `xformers.py` 的 `PagedAttention.forward_prefix` → Triton `prefix_prefill.py` 的 `context_attention_fwd` kernel 抛出;作者最初在 P40 上观察到,后希望在其他(受支持的)GPU 上复现。正文给出了完整的多进程 worker 崩溃日志。

## 根因
正文未披露根因——最初描述已被作者删除(strike-through),且 pack 中正文在 20000 字符处被截断,后续部分未见到。从 diff 看,`_add_seq_group` 中 prefix cache 命中时传给 attention kernel 的 `block_table` 取的是 `computed_block_nums`,而非该序列实际的 `block_tables[seq_id]`;同时对非滑窗配置(`curr_sliding_window_block == 0`)也在做 `[-curr_sliding_window_block:]` 负切片。prefix cache 命中与 chunked prefill 组合时两者与 slot mapping/context 长度的计算不一致,可能造成越界访存(由 diff 推断)。

## 修复方案
改 `vllm/attention/backends/utils.py` 的 `_add_seq_group`:
```python
-                block_table = computed_block_nums
+                block_table = block_tables[seq_id]
```
prefix cache 命中时改用完整 block table;非滑窗(`curr_sliding_window_block == 0`)时取整个 `block_tables[seq_id]` 而不做负切片,滑窗行为保持 `[-curr_sliding_window_block:]` 不变。同时在 `tests/prefix_caching/test_prefix_caching.py` 新增 `test_unstable_prompt_sequence`(5 条长 prompt 序列,FLASH_ATTN/FLASHINFER/XFORMERS 三后 × Qwen2.5-0.5B-Instruct,开 chunked prefill + prefix caching)作为回归测试。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:所有走 `AttentionMetadata` 构造的 v0 attention 后端的 block table 装配(prefix cache 命中、滑窗与非滑窗路径);prefix cache 命中时 kernel 现在能看到完整 block table,对 FlashAttention/FlashInfer/XFormers 同样生效(新增测试即按三后端参数化)。

## 关联
正文未列出关联 issue/PR

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
