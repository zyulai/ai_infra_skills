# PR #24539: [Spec Decode] Efficient padded speculation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2025-09-18 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, v1 |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 5 个文件（speculative.py、eagle.py、gpu_input_batch.py、gpu_model_runner.py、test_eagle.py），1 个文件被剔除 |
| 链接 | https://github.com/vllm-project/vllm/pull/24539 |

## 一句话结论

实现 padded speculative decoding（EAGLE 默认开启）：draft 阶段用 GPU 采样结果 + 全 padding batch 消除 CPU bookkeeping 阻塞，Llama 3.1 8B EAGLE3 在 B200 上 TTFT/TPOT 各提速 2%。

## 动机

V1 EAGLE 投机解码此前要等 target 模型的 bookkeeping（CPU 侧整理 `valid_sampled_token_ids`）完成后才能开始 draft，draft 模型每步被 CPU-GPU 同步卡住。RFC #21984 提出 padded speculation：把被拒绝的 token 保留在 batch 中作为 padding、后续由 `token_indices_to_sample` 过滤，让 draft 直接消费 GPU 上的采样张量。本卡是 #22684/#20078 实现的清理版与最新 main 的兼容版，只做公共核心（不含新 attention backend 支持）。

## 优化点

(1) `execute_model` 中 EAGLE 路径在 Bookkeep 之前直接用 GPU 上的 `sampler_output.sampled_token_ids` 调 `propose_draft_token_ids`，draft 与 bookkeeping 并行；ngram/medusa 仍走 CPU 路径。(2) 新增 `prepare_next_token_ids_padded`：全 device 操作处理 sampled token——被丢弃请求的索引 `index_fill_(-1)`、构造 valid mask、`torch.gather` 取每行最右合法 token、无效位置用预计算的 `backup_next_token_ids`（CPU-GPU pinned buffer）兜底，返回 `next_token_ids` 与 `valid_sampled_tokens_count`，无阻塞同步。(3) 新增 `prepare_inputs_padded`：不改 query 长度（被拒 token 保留为 padding），构造 `spec_common_attn_metadata`（slot_mapping 按全部 token 索引切片）并返回 `token_indices_to_sample = query_start_loc[1:] - 1 - num_rejected_tokens_gpu`，propose 阶段改用 `last_token_indices` 取 hidden states。(4) draft 循环重写：多步 drafting 中改为更新 `common_attn_metadata`（含 CPU 侧 `seq_lens_cpu`/`query_start_loc_cpu`）并每步经 `build_for_drafting` 重建 per-backend attention metadata，不再原地改旧 metadata；`num_speculative_tokens==1` 的 early exit 提前到第一次 logits 计算后。(5) `disable_padded_drafter_batch` 配置开关保留旧行为（默认 False 即新行为开启）。附带：`get_token_id` 越界返回 -1、discard request 索引移到 `_prepare_inputs` 中用 numpy 预计算。

## 关键改动

```text
vllm/v1/spec_decode/eagle.py: 新增 prepare_next_token_ids_padded/prepare_inputs_padded，propose 支持 last_token_indices，draft 循环按步重建 attn metadata
vllm/v1/worker/gpu_model_runner.py: EAGLE 在 bookkeeping 前用 GPU sampled_token_ids 起 draft；discard 索引预计算
vllm/config/speculative.py: 新增 disable_padded_drafter_batch 开关
vllm/v1/worker/gpu_input_batch.py: CachedRequestState.get_token_id 越界返回 -1
tests/v1/spec_decode/test_eagle.py: 新增 test_prepare_next_token_ids / test_prepare_inputs_padded
```

无同步取 token 的核心：

```python
next_token_ids = torch.where(
    last_valid_indices != -1, selected_tokens,
    self.backup_next_token_ids.gpu[:batch_size])
```

## 性能声明

测试条件：Llama 3.1 8B-Instruct + EAGLE3（`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`，num_speculative_tokens=4）、单张 B200、`VLLM_ATTENTION_BACKEND=FLASH_ATTN`、max-model-len 2048、prefix caching 关。结果：TTFT 与 TPOT 均有约 **2% speedup**。Profile 确认 EAGLE drafting 阶段无 GPU→CPU sync；收益被两个抵消因素压低：draft 模型的 CUDA graph 尚不可用（#23679）、bookkeeping 为全 GPU 化新增了大量小算子。作者预期新 attention backend 支持、draft CUDA graph 恢复、bookkeeping 融合 kernel（#20078）落地后性能继续提升。`disable_padded_batch` 开启时性能与行为不变。

## 验证与回归风险

新增 `test_prepare_next_token_ids`（CPU/padded 两条路径对拍，覆盖 accepted/rejected/skipped/discarded 四种请求状态）与 `test_prepare_inputs_padded`（token_indices 与 token_indices_to_sample 期望值校验），`test_prepare_inputs` 更新为新签名。风险面：(1) padded batch 要求 attention metadata 构建支持 padding token，未适配的 backend 会因 `allowed_attn_types` 断言失败（旧 assert 改成了带提示的 ValueError）；(2) `prepare_next_token_ids_padded` 中 backup token 仍走逐请求 Python 循环（numpy），大 batch 下有 CPU 开销；(3) `token_indices_to_sample` 依赖 rejection 计数正确，partial prefill 与 discard 交互路径（`get_token_id` 返回 -1）需被下游正确处理；(4) 本卡未支持新 backend，#21984/#24045 方向的 backend 适配留待后续。

## 关联

前序：#21984（RFC）、#22684、#20078（实现来源与融合 kernel 方向）；相关：#23679（draft model CUDA graph 不可用问题）、#24045（backend 支持）。同分类：#14702（spec decode + structured outputs）、#38047（GDN build 去 sync）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
