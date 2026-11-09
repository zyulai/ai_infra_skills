# PR #21088: [v1] Add Whisper model support (encoder-decoder)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @russellb |
| 合并 | 2025-09-10 |
| 分类 | KV Cache / Memory(major) |
| Label | documentation, speculative-decoding, ready, ci/build, v1, multi-modality |
| Diff | diffstat 缺失(diff 全文已提供但裁剪了 8 个文件,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/21088 |

## 一句话结论

V1 引擎新增 encoder-decoder 架构与 cross-attention KV cache 管理,使 Whisper(V0-only)可在新引擎上转录,补齐 V0 弃用(#18571)前的能力缺口。

## 动机

V1 引擎此前不支持 encoder-decoder 模型:`processor.py` 对 encoder 输入直接 `raise NotImplementedError`,Whisper 带 `SupportsV0Only`。V0 已进入弃用流程(关联 #18571 与 2025 Q3 roadmap #20336,closes #12761),不补齐该架构则 Whisper 用户无法迁移;需要一套 cross-attention KV cache 的分配/管理机制与调度器配合。

## 优化点

新增 `vllm/attention/layers/cross_attention.py`:`CrossAttention` 层按 KV cache 规格挂接 `CrossAttentionSpec`(独立于 decoder 自注意力的 KV cache 组),并通过 `subclass_attention_backend` 动态生成 `CrossAttention_*` 后端——builder 在 `build` 时把 `causal` 置 False、`seq_lens` 填为 encoder 最大长度,按 `encoder_seq_lens` 计算 cross slot mapping(仅遍历非零 encoder 长度的请求)。调度器:encoder-decoder 模型禁用 chunked prefill 与 prefix caching,`max_num_encoder_input_tokens` 设为 encoder 长度;encoder 输入不在当前 step 去重(暂不使用 encoder cache),且一旦生成首个 token 即释放 encoder 输入(cross-attention KV 已写入缓存)。`gpu_model_runner` 增加 `max_encoder_len`、`encoder_seq_lens` 传递与 InputBatch 长度上限取 max。`whisper.py` 移除 `SupportsV0Only`,encoder 层注意力按类型分派(ENCODER 用支持 2D 输入的 `WhisperEncoderAttention`,ENCODER_DECODER 用 `CrossAttention`,DECODER 用普通 `Attention`);Voxtral 的 standalone encoder 路径同步去掉 `is_standalone_encoder`。各 V1 attention backend 的 MetadataBuilder 统一改为调用基类 `__init__`(补齐 `layer_names/vllm_config/device` 字段)。

## 关键改动

```text
vllm/attention/layers/cross_attention.py: CrossAttention 层 + cross-attention 后端子类与 slot mapping 计算
vllm/v1/core/sched/scheduler.py: enc-dec 调度(禁 chunked prefill/prefix caching;首 token 后释放 encoder 输入)
vllm/v1/worker/gpu_model_runner.py: max_encoder_len、encoder_seq_lens、InputBatch 上限
vllm/model_executor/models/whisper.py: 三类注意力按 AttentionType 分派;移除 SupportsV0Only
vllm/v1/attention/backends/utils.py: AttentionMetadataBuilder 基类 __init__ 补全;CommonAttentionMetadata 增 encoder_seq_lens
(其余 8 个文件在素材 diff 中被裁剪)
```

cross slot mapping 节选:

```python
active_indices = np.nonzero(encoder_seq_lens)[0]
for req_index in active_indices:
    num_blocks_needed = cdiv(encoder_seq_len, block_size)
    slot_mapping = block_numbers * block_size + block_offsets
```

## 性能声明

PR 正文未提供 benchmark 数字(无性能声明);这是能力补齐型 PR。TODO 项注明尚未利用 encoder cache(每步重复调度 encoder 输入),留有后续性能空间。

## 验证与回归风险

`tests/models/multimodal/generation/test_whisper.py` 收窄为 whisper-large-v3-turbo(移除 whisper-small);BART 测试标注 skip("bart not supported in V1",Whisper-only 支持);`test_initialization.py` 对 Whisper 设 `VLLM_WORKER_MULTIPROC_METHOD=spawn`,Florence2 跳过。风险面:所有 encoder-decoder 模型被禁用 chunked prefill 与 prefix caching(config `__post_init__` 全局判定);Whisper 已知 fork worker 问题(非 spawn 时告警);mistral audio config 显式 `is_encoder_decoder=False` 覆盖以防误判。

## 关联

Closes #12761;Related:#18571(V0 deprecation)、#20336(roadmap)。KV cache / 调度区域其他 PR 编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
