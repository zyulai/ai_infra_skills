# PR #52789: [Perf] Support internal prefill checkpoints for Mamba prefix caching, 9%~25% TTFT improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-08-22 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | ready, ci/build, kimi, k3, scheduler, kv-cache-manager |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动约 11 个文件(另有 2 个文件因超长省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/52789 |

## 一句话结论

Mamba prefix caching 支持预填内部 checkpoint,8K prefill 不再为块对齐多做一次完整模型前向,TTFT 改善 9%~25%。

## 动机

Mamba/线性注意力层的状态是序列相关的,prefix caching 只能在块边界保存状态。此前 mamba_cache_mode="align" 下,prefill 被切分到最后一个块边界:8K 输入要先 forward 7680 tokens、FlashKDA、存 checkpoint,再对剩余 320 tokens 完整跑一遍模型前向 + FlashKDA。第二次前向要重新穿过 attention、MoE、routing 和 TP collectives,纯属对齐开销。

## 优化点

机制是"单次前向 + 内部 checkpoint":FlashKDA prefill kernel 新增 `checkpoint_state`/`checkpoint_offsets` 参数,允许在 prefill 内部按块对齐位置(须满足 `FLASHKDA_CHUNK_SIZE=16` 对齐)导出 recurrent 状态;新增 Triton kernel `_store_cache_checkpoints_kernel` 把 checkpoint 的 conv state 与 recurrent state 写进对应 mamba cache block。调度器 `_mamba_block_aligned_split` 在启用 internal checkpoint 时对块对齐起点取消尾部二次切分(`last_cache_position=0`);`KimiK3KDAMetadataBuilder` 计算 per-request 的 checkpoint offset/state_indices(`KDACheckpointMetadata`),非 FlashKDA 对齐或非对齐起点则回退 `NULL_BLOCK_ID` 不启用。`MambaSpec` 新增 `num_prefill_checkpoint_blocks`(KDA 层设为 1),max_mamba_memory 相应多计一个块;block 分配器为 checkpoint 块预留/豁免 null 填充。FlashKDA 子模块随之 bump。

## 关键改动

```text
vllm/models/kimi_k3/nvidia/kda.py: _flashkda_prefill 传 checkpoint 参数 + _store_cache_checkpoints_kernel
vllm/models/kimi_k3/nvidia/kda_metadata.py: builder 计算 checkpoint offsets/state_indices
vllm/v1/core/sched/scheduler.py: aligned split 在 internal checkpoint 下不切尾块
vllm/v1/core/single_type_kv_cache_manager.py: checkpoint 块的分配/预留逻辑
vllm/v1/kv_cache_interface.py: MambaSpec.num_prefill_checkpoint_blocks 与内存计量
```

## 性能声明

Kimi-K3,`-tp 8`,guidellm,synthetic_text prompt_tokens=8000 / output_tokens=1000(reasoning_effort max,temp 1.0,top_p 0.95),TP8 硬件型号未注明。TTFT:并发 1 435.28→394.19 ms(**9.4%**),并发 4 1450.94→1080.78 ms(**25.5%**),并发 16 2648.48→2099.69 ms(**20.7%**)。其他指标:prompt tokens/s 分别 +4.0%/+1.4%/+2.4%,request latency -3.8%/-3.4%/-0.2%。

## 验证与回归风险

`tests/models/kimi_k3/test_kda.py`:flashkda correctness 测试扩展,checkpoint 状态与 naive_recurrent_kda 参考比对;`test_kda_metadata.py` 新增对齐边界取最后一个 aligned boundary、非对齐起点跳过 checkpoint 两个用例;`tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` 新增 partial hit + internal checkpoint 时 mamba 块互不重叠、起点非块对齐不产 checkpoint 的用例;`test_mamba_align_chunk_split.py` 参数化扩展。风险点:spec decode 暂不支持(`use_eagle` 直接禁用);仅覆盖 flashkda 后端的 KDA 层,Mamba2 等其他线性层未启用;checkpoint 块占用额外 1 块 mamba 内存;非 16 对齐 offset 自动降级回原路径。

## 关联

同区域:后续任务 PR 明言待做 spec decoding 支持与 flashKDA 更新(避免 FlashKDA 两次 kernel 调用);mamba/混合模型优化见 #20839(Mamba2 RMSNormGated kernel)、#28587(hybrid mamba2 torch compile)。
