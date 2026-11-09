# PR #53614: [Kimi K3] Support internal prefix checkpoints with partial prefix caching and spec-decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZeldaHuang |
| 合并 | 2026-09-06 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, kv-connector, kimi, k3, scheduler, kv-cache-manager |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),另有 2 个文件因超长从 diff 中省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/53614 |

## 一句话结论

把 Kimi-K3 的 prefill 内部 checkpoint 机制扩展到推测解码与 partial prefix caching,增量请求 P50 TTFT 改善 25.57%、有效吞吐提升 28.08%。

## 动机

PR #52789 引入的 Kimi-K3 prefill checkpoint(Mamba/KDA 状态导出为可复用的 cache 条目)此前不支持三种生产场景:Eagle/DSpark 推测解码的 block rewind、`prefix_match_unit` 小于 Mamba block size 的部分前缀命中、以及 checkpoint block 经 KV connector(MooncakeStore)恢复。多轮递增式长 prompt(如每轮 +2K token)在非 partial 模式下无法命中部分边界,导致大量状态重算。

## 优化点

三处协同改动,共享统一的 checkpoint 合法性规则(`is_mamba_prefill_checkpoint_valid` / `get_mamba_prefill_checkpoint_position` 提升到 `kv_cache_interface.py`)。(1) 推测解码:cache manager 与 FlashKDA worker 用同一套规则判断 checkpoint 是否可导出;Eagle block drop 开启时 checkpoint 位置回退一个 hash block,使 Mamba checkpoint 对齐 Eagle 的 attention replay 边界(`MambaManager.drop_eagle_checkpoint_block`,由 coordinator 在 `use_eagle` 时置位);调度器 `_mamba_block_aligned_split` 的内部 checkpoint 判定改用共享函数。(2) partial prefix caching:MambaSpec 新增 `prefill_checkpoint_alignment`(FlashKDA 为 16);对齐的 partial checkpoint 把请求自有的 checkpoint block 从临时满块边界 re-key 到 FlashKDA 实际导出的 state 位置(`block_pool.cache_partial_block` 新增 `replace_existing_hashes` 参数,先移除该 block 的旧 hash 条目再注册新条目,消除 stale hash),`_checkpoint_positions` 取代原来的 checkpoint block 计数。(3) KV connector:`cache_blocks` 提供 boundary offload 时改用 `block.block_hash_num_tokens` 精确边界,使 MooncakeStore 等连接器能按导出状态保存/恢复 checkpoint block。

## 关键改动

```text
vllm/v1/kv_cache_interface.py: MambaSpec 增加 prefill_checkpoint_alignment;新增 get_mamba_prefill_checkpoint_position 与 is_mamba_prefill_checkpoint_valid 共享规则
vllm/v1/core/single_type_kv_cache_manager.py: MambaManager 以 checkpoint_position 驱动分配,re-key 导出边界,Eagle drop 语义
vllm/v1/core/block_pool.py: cache_partial_block 支持 replace_existing_hashes(内容被替换时清理旧 hash)
vllm/v1/core/sched/scheduler.py: _mamba_block_aligned_split 用共享合法性函数判定内部 checkpoint 并取消内部 checkpoint 时的边界 stop
vllm/models/kimi_k3/nvidia/kda_metadata.py: checkpoint metadata 按 checkpoint_position/alignment 构建,offset = checkpoint_position - query_start
```

关键合法性规则:

```python
return (
    query_start % hash_block_size == 0
    and checkpoint_col > initial_state_col
    and query_start + hash_block_size <= checkpoint_position
    and query_start < checkpoint_position < query_end
    and (checkpoint_position - query_start) % checkpoint_alignment == 0
)
```

## 性能声明

部署:Kimi-K3 TP8/EP8(两个 4-GPU 节点),FlashKDA(DS conv-state layout),prefix caching,`--prefix-match-unit 128`,`Inferact/Kimi-K3-DSpark` 推测解码(DSpark,num_speculative_tokens 7),enforce-eager。负载:精确 token-ID prompt 发往 `/v1/completions`,8 流从 120K 到 164K 每轮 +2K,组间 barrier,每变体 1 次预热 + 3 次测量,每变体 528 个增量请求全部成功:D-Spark-7 non-partial 增量 P50 TTFT 1,526.59 ms、有效 req/s 4.639、平均重算 4,803.64 tokens;D-Spark-7 Partial128 为 1,136.22 ms、5.941 req/s、平均重算 2,409.45 tokens。即 partial 命中多缓存 2,394.18 tokens/请求、重算减少 49.84%、增量 P50 TTFT -25.57%、有效吞吐 +28.08%;冷 TTFT 基本不变(-0.44%)。精度:OCR-Bench 全量 1,000 样本跑 3 次(D-Spark-7 + Partial128),得分 0.881/0.889/0.886(均值 0.8853),零请求错误、零可疑输出;MooncakeStore + NIXL PD 分离部署下三跑均值 0.888。PR 未注明 GPU 型号(仅给出 nightly 镜像标签)。

## 验证与回归风险

focused 单测 113 passed;新增/改写的测试覆盖:Eagle rewind 下 checkpoint 对齐 replay 边界(`disable_eagle_block_drop` x `prefix_match_unit` x 期望 offset 参数化,含 checkpoint 为 None 的负例)、partial hash 生命周期(state@112 替换临时 state@96 键、replay 时 num_computed==112)、Eagle 对齐 checkpoint 覆盖更新 hash、hash 对齐 query 结尾走常规 partial tail 而非内部 checkpoint。aligned re-key 回归覆盖 B=H=32、prompt 长 104、Eagle rewind 场景。风险面:scheduler 目前取第一个 MambaSpec 的 alignment(TODO 注明多 Mamba spec 需扩展);re-key 会主动删除旧 hash 条目,依赖与 worker 导出状态严格一致,否则可能出现可用条目缺失;行为变化仅限 Mamba align 模式 + checkpoint blocks > 0 的部署。

## 关联

同区域:PR #52789(前序,引入 prefill checkpoint 与 block 生命周期,本 PR 保持其分配/驱逐/释放语义);KV connector 恢复见 pr-60695(Mooncake save eagle attention)、pr-42296(simple connector DSV4);Kimi-K3 同模型优化见 pr-53396(DS conv-state layout 融合 KDA)、pr-50578(ROCm MLA asm decode);DSpark 见 pr-46995。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
