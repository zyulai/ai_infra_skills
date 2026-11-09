# PR #29105: [NPU] perf: precompute mamba conv-state track indices once per batch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AndyLi429 |
| 合并 | 2026-06-25 |
| 分类 | Speculative Decoding (tier: major) |
| Label | npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/29105 |

## 一句话结论

Ascend GDN backend 把 mamba conv-state 追踪索引从每层重算改为每 batch 预计算一次,消除 prefill 关键路径上每层的 `nonzero()` device->host 同步,P90/P95 TTFT 降约 14%,总吞吐 +4.8%。

## 动机

Ascend GDN(Gated Delta Net)attention backend 的 prefill/extend 路径原来在 `forward_extend` 内部、对每个 linear-attention 层重算一次 mamba conv-state 追踪索引(`mamba_track_mask.nonzero()` 加一次对 `mamba_track_indices` 的 gather)。这批索引是 batch 级数据,跨层不变,却被重复计算 K 层。在 NPU 上 `nonzero()` 产生数据依赖的动态 shape,强制一次 device->host sync,因此每层重复执行在 prefill 关键路径上引入可避免的开销,拖慢 TTFT。

## 优化点

新增 `AscendGDNAttnBackend._prepare_mamba_track_metadata()`:当 `forward_metadata.has_mamba_track_mask` 为真时,预计算 `mamba_track_mask_indices = mamba_track_mask.nonzero(as_tuple=True)[0]` 与 `conv_states_mask_indices = mamba_track_indices[mamba_track_mask_indices]`,存到 `ForwardMetadata` 上;该函数从 `init_forward_metadata` 与 `init_forward_metadata_out_graph` 两个入口调用,即每个 batch 恰好执行一次。`forward_extend` 里删除逐层的 `nonzero()` + gather 与内联的 `mamba_track_mask is not None and .any()` 守卫,改为读取预计算的 `forward_metadata.conv_states_mask_indices`。净效果:K 层 hybrid 模型中动态 shape 的 `nonzero()` 从每次 prefill K 次降到每 batch 1 次;decode 路径不动。PR 声明这是行为保持的改动:守卫定义 `bool(mask is not None and mask.any())` 与原内联守卫完全一致,索引值等于原 `mamba_track_indices[mask_indices]`;每个 batch 重建新的 `ForwardMetadata`,不存在跨 batch 的脏索引。附带一个 reviewer 提示:`mamba_track_mask_indices` 目前是只写字段(仅内部用于计算 conv_states_mask_indices),非阻塞。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py: 新增 _prepare_mamba_track_metadata;init_forward_metadata(_out_graph) 调用;forward_extend 改读预计算索引
```

关键片段(forward_extend 替换后):

```python
if forward_metadata.has_mamba_track_mask:
    mixed_qkv_to_track = mixed_qkv[
        :, forward_metadata.track_conv_indices
    ].transpose(0, 1)
    conv_states.transpose(1, 2)[
        forward_metadata.conv_states_mask_indices
    ] = mixed_qkv_to_track
```

## 性能声明

负载:同一 workload,80 requests @ concurrency 20,MTP on(accept length 约 3.7),Qwen3.6-27B,tp-size 2,chunked-prefill 32768,`--speculative-algorithm NEXTN` 等配置见正文。硬件一栏 PR 原文写作 `<Ascend 910x — confirm SKU>`(占位符,PR 内未最终确认具体 SKU):

- Benchmark duration 360.15 s -> 343.59 s(-4.6%)
- Total token throughput 15776.60 -> 16537.05 tok/s(+4.8%);Output/Input token throughput 同为 +4.8%
- Mean TTFT 31962.90 -> 30373.23 ms(-5.0%);P90 TTFT 55360.60 -> 47856.74 ms(-13.6%);P95 TTFT 55727.35 -> 47961.14 ms(-13.9%);P99 TTFT -8.6%
- Median E2E 82039.43 -> 77467.83 ms(-5.6%)
- Mean TPOT 48.98 -> 49.57 ms(+1.2%,PR 自注 within noise)

PR 总结:收益集中在 TTFT/prefill(P90-P95 降约 14%)与总吞吐(+4.8%),与移除 prefill 路径冗余 per-layer `nonzero()` 一致;decode(TPOT)持平,符合 decode 路径未改动的预期。

## 验证与回归风险

准确性:PR 称改动行为保持,accuracy 应与改动前一致;用 evalscope 跑 GPQA-Diamond(Qwen3.6-27B,198 题)得 mean_acc 0.8586,在预期范围内,确认无回退。未注明新增单元测试;Checklist 未勾选;CI 状态显示 PR Test (Extra) 为失败标记。回归面:仅 Ascend NPU 的 GDN backend prefill/extend 路径,decode 与 CUDA 路径不受影响;语义依赖 `has_mamba_track_mask` 与 `conv_states_mask_indices` 在 ForwardMetadata 上按 batch 重建的约定。

## 关联

同区域为 Ascend NPU 上 GDN/mamba hybrid 模型(带 NEXTN MTP)推理优化;pack 素材未列出同分类其他 PR 编号,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
