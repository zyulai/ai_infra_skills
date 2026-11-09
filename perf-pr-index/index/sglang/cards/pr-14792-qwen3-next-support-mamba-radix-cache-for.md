# PR #14792: [Qwen3-next] support mamba radix cache for overlap scheduler

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hanming-lu |
| 合并 | 2025-12-15 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | documentation, npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/14792 |

## 一句话结论

Qwen3-next(GDN 混合线性)接入 mamba radix cache:overlap 调度、page_size>1 与分支点缓存三项能力补齐。

## 动机

Qwen3-next 此前不支持三样东西:1) overlap scheduler——线性注意力的 recurrent state 无法像 KV cache 一样做前缀缓存与分支复用,GDN 模型被排除在 overlap 调度的吞吐收益之外;2) page_size > 1;3) branching point caching(speculative decoding 的树分支点状态缓存)。这三项缺失使 GDN 混合模型的调度路径与 attention 路径不匹配,是纯性能/能力型短板。

## 优化点

核心机制是给 mamba/GDN 状态补一套与 forward 对齐的 track(状态拷贝到持久缓存槽)管道,使 radix cache 可以在 overlap 调度下缓存分支点状态。decode 侧新增 Triton kernel `track_mamba_state_if_needed_kernel`:按 `mamba_track_mask` 把 conv/ssm 状态从工作槽整行拷贝到 track 槽,替代原来的 Python 逐批循环拷贝。extend 侧利用 FLA chunk 结构做 chunk 边界对齐的状态追踪:`chunk_gated_delta_rule` 额外返回中间态 `h`;`_init_track_conv_indices` 只追踪到最后一个完整 FLA chunk 边界(conv state 取窗口尾部);`_init_track_ssm_indices` 按 seq len 是否对齐 FLA_CHUNK_SIZE 分两路——对齐用 `last_recurrent_state`、未对齐从中间态 `h` 的对应 chunk 索引取,避免缓存含未对齐位置的污染状态。spec 侧解耦:conv/ssm 的中间状态 buffer 引入独立的 `intermediate_state_indices`(kernel 直接按其索引写),使 spec decoding 中间状态尺寸与 max running requests 关联而不是与总 ssm 状态池关联(内存分配改进第 6 条);`update_mamba_state_after_mtp_verify` 扩展为同时 scatter 到工作槽与 track 槽。通过 `--mamba-scheduler-strategy extra_buffer` 启用;PD decode 侧暂不启用 extra buffer(TODO)。

## 关键改动

```text
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:
  track_mamba_state_if_needed_kernel(decode 状态追踪);
  _init_track_conv_indices/_init_track_ssm_indices(FLA chunk 边界索引);
  _track_mamba_state_decode/_track_mamba_state_extend 挂入 GDN forward
python/sglang/srt/layers/attention/fla/{chunk,chunk_delta_h}.py:
  chunk_gated_delta_rule 返回中间态 h;CHUNK_SIZE 常量化
python/sglang/srt/layers/attention/fla/fused_recurrent.py,
  mamba/causal_conv1d_triton.py: 新增 intermediate_state_indices 解耦索引
python/sglang/srt/managers/(schedule_batch 等调度面,24 文件未展开):
  mamba_track_mask/indices、radix cache、overlap 调度接线
```

```python
# decode:mask 为真的请求把 conv/ssm 状态整行拷到 track 槽(替代 Python 循环)
src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices
```

## 性能声明

PR 正文未提供 benchmark 数字(Benchmarking 小节为空)。性能意义为能力解锁:overlap scheduler 支持使 Qwen3-next 可以进入 overlap 调度路径(此前完全被排除),分支点缓存使 spec decoding 的树分支可复用缓存状态;spec 中间状态内存从"与总 ssm 状态池耦合"改为"与 max running requests 耦合"直接减少内存占用。具体提升幅度 PR 未注明。

## 验证与回归风险

新增 mamba radix cache KL 测试覆盖 prefill 与 decode,两种策略(`--mamba-scheduler-strategy extra_buffer` 与 `no_buffer`)。矩阵测试:(ps=1, ps>1) × (非 spec、sd topk1、sd topk>1) 全部通过,仅 ps>1 + sd topk>1 组合不支持(main 尚未支持)。回归风险:conv state 追踪断言 `conv_states_shape[-1] < FLA_CHUNK_SIZE`,conv 窗口更大的模型会 assert;`_init_track_ssm_indices` 在 host 侧做 mask 过滤(CPU 往返),只换来了减少 kernel launch 的取舍;PD 分离 decode 侧 `enable_mamba_extra_buffer=False` 硬编码,PD + mamba radix cache 组合未打通;ps>1 + sd topk>1 需等 main 支持。

## 关联

同区域(Mamba/GDN + 调度):本批次 #29792(overlap 调度下 Mamba track-boundary bookkeeping 修复,可视为本 PR 打开的 overlap 路径的后续正确性收敛)、#32219(hybrid-linear MTP host-seam)、#43171(GDN prefill 后端选择)。pack 未列出其他直接关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
