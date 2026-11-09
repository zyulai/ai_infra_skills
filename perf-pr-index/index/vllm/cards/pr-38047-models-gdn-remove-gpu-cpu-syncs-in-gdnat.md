# PR #38047: [Models][GDN] Remove GPU/CPU syncs in `GDNAttentionMetadata.build` during speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lgeiger |
| 合并 | 2026-04-06 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, v1 |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 1 个文件（gdn_attn.py，约 +10/-9 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/38047 |

## 一句话结论

把 GDN attention metadata 构建中的布尔 mask 移到 CPU 并修正 `repeat_interleave` 的 `output_size`，消除投机解码路径上的 GPU/CPU 同步，吞吐 +7%、TTFT −20%。

## 动机

`GDNAttentionMetadata.build`（GDN/Qwen3.5 类混合模型的 attention backend）在投机解码期间用 GPU 上的布尔 mask 做张量索引与 `repeat_interleave`。GPU 布尔 mask 索引无法异步执行——PyTorch 需要把 mask 拷回 CPU 并同步确定输出大小，形成 GPU/CPU sync，打断 CUDA 流水。自 pytorch/pytorch#156384 起，mask 在 CPU 上时布尔索引可以异步完成。

## 优化点

两处机制修改：(1) 引入 `spec_sequence_masks_cpu`，所有布尔索引（`block_table_tensor[spec_sequence_masks_cpu]`、`~spec_sequence_masks_cpu`、`query_lens[spec_sequence_masks_cpu]`、`num_accepted_tokens[mask]`、`has_initial_state[...]`）均改用 CPU mask，索引操作变为可异步；(2) `torch.repeat_interleave(spec_sequence_masks, query_lens)` 显式传 `output_size=query_start_loc_cpu[-1].item()`——这里 `.item()` 读取的是已在 CPU 上的 `query_start_loc_cpu`，不产生新的 GPU 同步，而 `repeat_interleave` 在未指定 `output_size` 时必须同步读取 GPU 上的 `query_lens` 求和来确定输出形状。其余逻辑（argsort 排序、cumsum、chunk metadata）不变。

## 关键改动

```text
vllm/v1/attention/backends/gdn_attn.py: GDNAttentionMetadata.build 布尔索引全部改用 CPU mask；repeat_interleave 显式指定 output_size
```

消除同步的关键一行：

```python
spec_token_masks = torch.repeat_interleave(
    spec_sequence_masks,
    query_lens,
    output_size=query_start_loc_cpu[-1].item(),
)
```

## 性能声明

测试条件：Qwen/Qwen3.5-2B、MTP 投机解码（`num_speculative_tokens: 1`）、`vllm bench serve` sharegpt 1000 prompts、video 关闭。结果：total token throughput 20073.91 → 21452.04 tok/s（**+7%**，作者表述为 7% improved throughput），output token throughput 9149.19 → 9838.98 tok/s，request throughput 48.61 → 51.67 req/s；Mean TTFT 4265.28 → 3397.06 ms（**−20%**），Median TTFT 4177.40 → 3261.89 ms，P99 TTFT 6718.88 → 5761.40 ms；Mean TPOT 61.07 → 57.88 ms。投机指标不变（acceptance rate 55.32% → 55.01%，acceptance length 1.55），确认不改变解码行为。torch profile 确认 sync 消失。

## 验证与回归风险

验证手段为 torch profile（sync 消失）+ 上述 e2e benchmark；素材未列出新增单测。风险面：(1) CPU mask 要求 `spec_sequence_masks` 的真值在 CPU 侧可得，若上游生成路径变化（mask 只在 GPU 上）需要额外拷贝；(2) `output_size` 取自 `query_start_loc_cpu[-1]`，依赖该 CPU 张量与 GPU `query_lens` 的一致性——投机解码元数据构建约定变化时需同步；(3) 仅覆盖 GDN backend 的 `build` 路径，其他 backend 的同类同步模式不受影响。

## 关联

同分类（Speculative Decoding）其他 PR：#14702（spec decode + structured outputs）、#24539（padded speculation）、#38047 同区域前序未在素材中列出。依赖 PyTorch 上游 pytorch/pytorch#156384（CPU 布尔索引异步化）。GDN/Qwen3.5 投机解码线：素材未给出更多编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
