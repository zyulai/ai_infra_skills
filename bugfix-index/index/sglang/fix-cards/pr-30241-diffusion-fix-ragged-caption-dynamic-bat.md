# PR #30241: [diffusion] Fix ragged-caption dynamic-batching accuracy bug in Ernie-Image

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30241](https://github.com/sgl-project/sglang/pull/30241) |
| 作者 | mickqian |
| 合并日期 | 2026-07-07 |
| labels | diffusion, run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
动态批合并不同真实长度的 caption 时,Ernie-Image/Wan T2V 会让 cross-attention 看到共享 padded 文本尾部而静默污染输出;接入 require_text_seq_lens/varlen 掩码修复。

## 问题现象
这是 #29742 Z-Image 精度修复的跟进:审阅者问及「动态批在请求真实 caption 长度不同时静默破坏输出」这类 bug 是否 Z-Image 独有,答案是否定——`ernie_image` 与 `wan`(T2V)存在同样的缺口。影响场景是动态批处理把真实长度不同的请求合并进一个共享、被 padding 的 `encoder_hidden_states` 张量:cross-attention 会 attend 到填充尾部,输出被静默污染。正文未附错误输出截图,准确率/速度测试未运行(作者环境无 GPU,显式标注留待有 GPU 的人做 mixed-caption-length batch vs singleton 对比)。

## 根因
正文明确披露两处:
- **Ernie-Image**:`_prepare_cond_kwargs` 硬编码 `txt_seq_lens = [prompt_embeds[0].shape[1]]`(无视真实 batch size 的单元素列表),且 DiT `forward()` 完全没有 attention-mask 参数——没有任何机制阻止 cross-attention 看到共享 padded 张量的填充尾部。
- **Wan T2V**:`WanT2VCrossAttention.forward()`/`WanI2VCrossAttention.forward()` 声明了正为此设计的 `context_lens` 参数,但所有调用点都硬编码 `context_lens=None`(死代码);每个请求的真实长度其实在 `t5_postprocess_text` 里已算出,随后被丢弃。
正文同时确认 SD3 存在同类 bug 但有意不在本 PR 范围内:其 DiT 直接包装 diffusers 的 `JointTransformerBlock`/`JointAttnProcessor2_0`,后者接受 `attention_mask` 参数却从不使用(作者读已安装的 diffusers 源码确认),真正修复需自定义 AttnProcessor,风险/范围更大。

## 修复方案
**Ernie-Image**(`configs/pipeline_configs/ernie_image.py`、`runtime/models/dits/ernie_image.py`):
- tokenizer padding 由 `False` 改 `"longest"`,不同长度 prompt 才能合批 tokenize;
- `ernie_image_postprocess_text` 按 tokenizer attention mask 抽取每个请求的真实(未填充)token 区间再经 `pad_text_embeddings_with_mask` 重填充,返回 `TextConditioningOutput`:
```diff
-    return hidden_states
+    prompt_mask = _text_inputs.attention_mask.to(hidden_states.device).bool()
+    split_hidden_states = [
+        hidden_states[idx][prompt_mask[idx]] for idx in range(hidden_states.shape[0])
+    ]
+    return pad_text_embeddings_with_mask(split_hidden_states)
```
- `_prepare_cond_kwargs` 用 `require_text_seq_lens` 读真实 per-request 长度(替代 batch-size-1 假设),新增 `_prepare_encoder_hidden_states_mask` 构造 `[batch, text_seq_len]` 真实 token 掩码(等长时返回 None);
- DiT `forward()` 接受 `encoder_hidden_states_mask`,拼出 `[image, text]` joint 掩码并 `build_varlen_mask_meta`,沿 block/自注意力调用链传入既有 `USPAttention`。
**Wan T2V**(`configs/pipeline_configs/wan.py`、`runtime/models/dits/wanvideo.py`,据正文;这两个文件与 `test_wan_pipeline_config.py` 的改动在 pack 的 diff 中未展示,内容以正文描述为准):`t5_postprocess_text` 返回 `TextConditioningOutput`;`prepare_pos/neg_cond_kwargs` 经 `require_text_seq_lens` 构造 `encoder_hidden_states_mask`;`WanTransformer3DModel.forward()` 把掩码穿到 cross-attention 的 `context_lens`(原先恒为 None)并经 `build_varlen_mask_meta` 掩蔽 padding;`WanI2VCrossAttention` 同样接线(I2V 目前不在动态批准入白名单,今日为 inert)。

## 影响与回归风险
- 影响窗口:正文未说明(该缺口自 Ernie-Image/Wan T2V 支持引入起即存在)。
- 回归面:Ernie-Image 与 Wan T2V 的文本编码后处理、条件 kwargs 构造、DiT 注意力调用链;非等长 batch 新增掩码与 varlen 元数据路径,等长 batch 行为不变(掩码为 None);SD3 未修(同类 bug 仍在)。单测为 CPU-only,作者注明本地 pytest 因环境既有 torch/huggingface_hub 版本不匹配无法直接运行(对既有 `test_zimage_pipeline_config.py` 亦复现,非本改动引入)。

## 关联
- [PR #29742](https://github.com/sgl-project/sglang/pull/29742)(Z-Image 同类修复)
- [#29742 审阅评论](https://github.com/sgl-project/sglang/pull/29742#pullrequestreview-4600915803)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
