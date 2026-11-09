# PR #24934: DeepSeek V4 MTP Support CP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Paiiiiiiiiiiiiii |
| 合并 | 2026-05-19 |
| 分类 | Speculative Decoding(classify.py 给出,tier: major) |
| Label | high priority, deepseek |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 2 个文件(NextN 模型 + registered 测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/24934 |

## 一句话结论

让 DeepSeek-V4 的 MTP/NextN draft 模型在 NSA prefill context parallel(CP)下正确工作,MTP 可以叠加 CP 预填充部署。

## 动机

DeepSeek-V4 支持 `--enable-nsa-prefill-context-parallel`(NSA 预填充上下文并行,round-robin-split 模式),但 NextN/MTP draft 模型的前向没有适配 CP 数据切分:draft 的 decoder 前向拿到的 hidden_states/positions 是全局(未按 CP 分片)的,而 CP 模式下 target 侧按 rank 分片计算。PR 正文无 Motivation 文字,以上来自 diff 内容与测试配置的推断边界内描述——draft 需要在 prefill CP 下做 split/rebuild 与 all-gather 才能输出正确 logits。

## 优化点

- `DeepseekV4ModelNextN.forward`:`nsa_use_prefill_cp(forward_batch)` 为真时,decoder 前向前对 hidden_states 与 positions 执行 `cp_split_and_rebuild_data` / `cp_split_and_rebuild_position`,decoder 前向后用 `cp_all_gather_rerange_output` 把各 CP rank 的分片结果重组回全局序列,保证 hc_head 与 logits 处理在完整序列上进行。
- `DeepseekV4ForCausalLMNextN.forward`(draft 入口):CP 启用时若 `can_nsa_cp_split` 允许切分,先 `prepare_context_parallel_metadata` 构造 `attn_cp_metadata`;round-robin-split 模式下对 core attention metadata 执行 `apply_cp_reindex()` 与 `init_flashmla_related()`,并重建 indexer metadata(`init_forward_metadata_indexer`),使 target 侧已按 CP 重排的 KV/索引视图与 draft 前向对齐。
- 模型初始化时缓存 `nsa_enable_prefill_cp`、`cp_rank`、`cp_size`,非 CP 部署路径完全不受影响。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4_nextn.py: NextN draft 的 CP 切分/重组与 metadata 准备
test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py: 新增 CP + MTP 组合的 registered 测试
```

```python
if nsa_use_prefill_cp(forward_batch):
    hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
    positions = cp_split_and_rebuild_position(forward_batch, positions)
hidden_states = self.decoder(positions=positions, hidden_states=hidden_states, ...)
if nsa_use_prefill_cp(forward_batch):
    hidden_states = cp_all_gather_rerange_output(
        hidden_states, self.cp_size, forward_batch, torch.cuda.current_stream())
```

## 性能声明

PR 正文未提供性能对比数据(Speed Tests 章节为空)。给出的是功能验证:8 GPU TP8 + DeepEP + NSA prefill CP(round-robin-split)+ EAGLE MTP(steps=1/topk=1/draft=2)下跑 GSM8K(20-shot、1319 题、parallel 1319)的截图佐证;未注明硬件型号与准确率数值(正文仅附截图)。

## 验证与回归风险

新增 registered 测试 `TestDSV4FlashFP4B200Balanced_CP`:TP=4、attn-cp-size=4、DP attention、DeepEP、EAGLE 1-step spec,跑 `_gsm8k_check`,使 CP+MTP 组合进入 CI 回归。风险:NSA/CP 元数据路径上对 `core_attn_metadata` 做 in-place `apply_cp_reindex`/`init_flashmla_related`,与 target 侧 metadata 状态共享,round-robin 之外的 CP 模式未覆盖;CI 快照显示 Base 与 Extra 均失败(以合并时为准);Checklist 未勾选。

## 关联

同区域:DeepSeek-V4 MTP/online compress(PR #26471)、NSA prefill context parallel 相关 CP 基础设施(`cp_utils`、`nsa/utils`);依赖 DeepSeek-V4 NSA indexer 优化线(如 PR #32755)。PR 正文未列出具体编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
