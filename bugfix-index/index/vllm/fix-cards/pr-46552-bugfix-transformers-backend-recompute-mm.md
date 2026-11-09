# PR #46552: [Bugfix] Transformers backend: recompute `mm_token_type_ids` per request for M-RoPE

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46552](https://github.com/vllm-project/vllm/pull/46552) |
| 作者 | decarpentierg |
| 合并日期 | 2026-06-28 |
| labels | bug, multi-modality, ready |
| 热度 | 👍 8 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
Transformers 后端把按图片缓存的全序列 `mm_token_type_ids` 复用给不同长度请求,`get_rope_index` 形状不匹配崩溃;改为每请求从 placeholder 范围重算。

## 问题现象
两个请求共用同一张图(相同 `mm_hash`,命中 multimodal-processor 缓存)但 prompt 长度不同时,第二个请求触发 `get_rope_index` 崩溃:

```
RuntimeError: The expanded size of the tensor (N) must match the existing size (M)
at non-singleton dimension 1.  Target sizes: [3, N].  Tensor sizes: [3, M]
```

V1 下该错误会杀死该 rank 的 engine core。影响所有经 Transformers fallback(`model_impl="transformers"`)的 M-RoPE VLM(如 Qwen2.5-VL、GLM-4.1V),只要 `mm_processor_cache_gb > 0`(默认)。作者的最小复现(Qwen/Qwen2.5-VL-3B-Instruct,dummy 权重,同图不同文本长度)修复前 target 请求崩溃(`[3,174] vs [3,167]`),修复后两个请求都成功;对照组 `mm_processor_cache_gb=0` 不崩溃,证实缓存是触发条件。

## 根因
正文完整披露:`mm_token_type_ids` 是全序列张量(每个 prompt token 一个条目),但 Transformers 后端把它随多模态特征按 `mm_hash`(图片内容)逐 item 缓存,而不是按请求缓存。两个请求图片相同但 prompt 长度不同时,第二个请求复用第一个请求的 `mm_token_type_ids`,其长度与当前 `input_ids` 不匹配,`get_rope_index` 即崩溃。

## 修复方案
改 `vllm/model_executor/models/transformers/multimodal.py`:`get_mrope_input_positions` 不再从缓存特征读取 `mm_token_type_ids`,改为每请求用每个 `mm_feature.mm_position` 的 `offset/length` 与模态映射(`image=1, video=2, audio=3`)重算:

```python
            mm_token_type_ids = torch.zeros(len(input_tokens), dtype=torch.int)
            for feature in mm_features:
                position = feature.mm_position
                offset, length = position.offset, position.length
                mm_token_type_id = _MODALITY_TO_TOKEN_TYPE_ID[feature.modality]
                mm_token_type_ids[offset : offset + length] = mm_token_type_id
            kwargs["mm_token_type_ids"] = mm_token_type_ids.unsqueeze(0)
```

同时:`apply` 中 `mm_token_type_ids`/`token_type_ids` 改为 `pop` 处理;`embed_multimodal` 删除对 `mm_token_type_ids` 的丢弃注释行;`get_mrope_input_positions` 的 kwargs 白名单相应调整。作者在 vLLM 0.23.0、transformers 5.9.0、torch 2.11.0、单张 NVIDIA L4 上验证。

## 影响与回归风险
- 影响窗口:正文验证于 vLLM 0.23.0;受影响区间为引入该缓存读取行为至本 PR 之间,`mm_processor_cache_gb > 0`(默认)即可能触发。
- 回归面:Transformers 后端多模态路径的 `apply`/`embed_multimodal`/`get_mrope_input_positions`;M-RoPE 模型的 `mm_token_type_ids` 改为按请求重算,video/audio 的 token type 值(2/3)由本 PR 的映射表定义,仅支持 images 的约束检查保持。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
