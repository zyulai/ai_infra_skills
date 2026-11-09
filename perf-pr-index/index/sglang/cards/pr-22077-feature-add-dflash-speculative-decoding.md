# PR #22077: [Feature] Add DFLASH speculative decoding support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @dcw02 |
| 合并 | 2026-04-07 |
| 分类 | Speculative Decoding (tier: major) |
| Label | high priority, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用但超长截断(9 个文件未展示) |
| 链接 | https://github.com/sgl-project/sglang/pull/22077 |

## 一句话结论

新增 DFLASH 投机解码:轻量 draft 模型吃目标模型中间层上下文特征、在草稿块上做非因果注意力,整体接入 CUDA graph 与 KV 预算;PR 正文未附性能数字。

## 动机

PR 正文仅一句"moving PR from https://github.com/sgl-project/sglang/pull/16818 to here",未写动机与数据。从实现看,DFLASH 是一种上下文特征式投机解码:目标模型在指定中间层捕获 aux hidden states,由小型 draft 模型(`fc` + `hidden_norm` 投影后)对每个位置连续预测草稿块,不再独立维护 embedding/LM head,也不做逐 token 自回归 draft。

## 优化点

机制上分四块。(1) 新模型 `models/dflash.py`:`DFlashDraftModel` 不含 token embedding 与 LM head,直接消费目标模型 embedding(`input_embeds` 必填);其 attention 用 `AttentionType.ENCODER_ONLY` 的非因果 RadixAttention 对草稿块整体打分,并提供 `kv_proj_only`(未量化权重时切片融合 QKV 权重做单次 GEMM、跳过 Q)与仅 K 的 norm/RoPE 快路径,把 ctx token 物化进 draft KV cache。(2) 目标侧:`model_runner` 解析 draft 配置(`parse_dflash_draft_config`),校验 draft 与 target 层数关系、`resolve_target_layer_ids` 选定捕获层,`set_dflash_layers_to_capture` 挂到 llama 系模型;KV cell size 按 draft 层数用 `scale_kv_cell_size_per_token_for_dflash` 缩放预算。(3) CUDA graph:`speculative_algorithm.is_speculative()` 统一 eagle/standalone/ngram/dflash 判定,draft worker 复用 TARGET_VERIFY 捕获(此前禁止),replay 支持 `input_embeds` 缓冲;`get_spec_info` 按 attention 后端决定是否启用 custom mask(能走内建 causal 路径的后端免 custom mask,避免 FlashInfer 因 `custom_mask_buf` 已初始化而误用零掩码)。(4) 调度:`--speculative-algorithm DFLASH` 校验(不支持 DP attention、仅 pp_size=1 等,截断部分不可见),请求侧禁 return_logprob 与 grammar 约束。

## 关键改动

```text
python/sglang/srt/models/dflash.py: 新增 DFlashDraftModel(无 embedding/lm_head,ENCODER_ONLY 非因果注意力,kv_proj_only)
python/sglang/srt/speculative/dflash_info.py / dflash_utils.py: DFlashVerifyInput、draft 配置解析与 verify mask 策略(截断未展示)
python/sglang/srt/model_executor/cuda_graph_runner.py: draft worker 复用 TARGET_VERIFY capture、input_embeds replay
python/sglang/srt/model_executor/model_runner.py: dflash aux 层捕获与 KV cell size 缩放
python/sglang/srt/server_args.py: DFLASH 算法参数(speculative_dflash_block_size/draft_window_size)与校验
```

```python
+        # DFlash uses non-causal attention over the draft block.
+        self.attn = RadixAttention(
+            ..., attn_type=AttentionType.ENCODER_ONLY,
+        )
```

## 性能声明

PR 正文与 pack 正文均无 benchmark 数字或测试条件;性能声明一节:PR 未注明。接受率/加速需参考其原始 PR #16818 或后续集成 PR(#46182 一类 DFlash 系列工作)。

## 验证与回归风险

- pack 中未展示新增测试文件(9 个文件未从 diff 截断中保留,测试情况无法从此素材确认)。
- 回归面:新投机算法默认不影响现有路径,但 `is_speculative()` 统一判定、flashinfer verify capture 的 custom_mask 条件化与 ragged prefill 的 causal 判定(`ENCODER_ONLY` 不 causal)触及共享代码;请求侧 DFLASH 明确不支持 return_logprob、JSON schema/regex/EBNF/structural tag 约束;不支持 DP attention、仅 pp_size=1(截断部分可见的校验)。

## 关联

- 前序:#16818(本 PR 的原始出处)。
- 同分类:#11398(EAGLE beta overlap);DFlash 系列 component:#46184(mamba conv verify 窗口 tile 写,component 1)、#46182(adaptive-verification 集成)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
