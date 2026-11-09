# PR #6830: [Speculative Decoding] EAGLE Implementation with Top-1 proposer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @abhigoyal1997 |
| 合并 | 2024-08-22 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 部分可用——因篇幅丢弃了 5 个文件,可见部分含约 12 个文件(EAGLE 模型、HiddenStates、spec_decode worker 等) |
| 链接 | https://github.com/vllm-project/vllm/pull/6830 |

## 一句话结论

为 vLLM 接入 EAGLE draft 模型(Top-1 proposer):target 隐状态传递给 draft、prefill 阶段 scorer 先于 proposer 执行、可选 token_map 截断词表。

## 动机

vLLM 当时的投机采样 proposer 仅支持 Medusa、MLP speculator 与 n-gram,缺少 EAGLE(论文 https://arxiv.org/pdf/2401.15077)这种以 target 模型隐状态为条件生成 draft 的方案;同时框架里 hidden states 只是 `SamplerOutput` 上的一个可选张量,不足以表达 EAGLE 所需的 prefill/decode 两阶段、bonus token 展开等状态传递语义(顺带修复 SafeAILab/EAGLE#43)。

## 优化点

三层改动。模型层:新增 `EAGLE` draft 模型与 `EAGLEConfig`,`forward` 把 `fc(concat(tok_embeds, previous_hidden_states))` 作为输入嵌入(position 0 处置零掩码),decoder 结构通过 `ModelRegistry.resolve_model_cls` 复用目标模型架构(与参考实现的差异:首层也有 input_layernorm、允许任意 decoder layer);可选 `token_map` 把 draft 词表截断为高频 token(`truncated_vocab_size < vocab_size` 时生效)以降低采样开销,`compute_logits` 中以 -inf 填充再 scatter 回原词表。状态层:`HiddenStates` 重构为 scorer 隐状态 + `second_last_token_hidden_states`(bonus token 场景),新增 `expand_with_bonus_tokens` 与 `_expand_execute_model_request` 对齐展开、`SamplerOutput.prefill_hidden_states` 携带 prefill 全 token 隐状态;`prepare_prefill_hidden_states` 用 `roll(1)` 把第 n-1 个隐状态对齐到第 n 个 token。调度层:`no_spec`(prefill/无请求/自动禁用)标志随 broadcast 传给非 driver worker,保证 prefill 时 scorer -> proposer、decode 时 proposer -> scorer 的执行顺序一致(EAGLE 需要 scorer 状态作输入);`draft_model_runner` 增加 `previous_hidden_states` 形参并在 CUDA graph capture 时 pad 到 graph batch size;model_runner 按 forward 签名探测是否需要 dummy `previous_hidden_states`;TP > 1 显式 `raise NotImplementedError`。

## 关键改动

```text
vllm/model_executor/models/eagle.py: 新增 EAGLE draft 模型(fc 融合 embedding+hidden、token_map 截断词表、权重加载)
vllm/sequence.py: HiddenStates 重构(second_last_token_hidden_states、expand_with_bonus_tokens)、SamplerOutput.prefill_hidden_states
vllm/spec_decode/spec_decode_worker.py: prefill 时 scorer 先于 proposer、prepare_prefill_hidden_states、TP>1 不支持
vllm/spec_decode/draft_model_runner.py: previous_hidden_states 传递与 CUDA graph pad
vllm/worker/model_runner.py: capture dummy previous_hidden_states、prefill_hidden_states 输出
vllm/transformers_utils/configs/eagle.py: 新增 EAGLEConfig
```

draft 前向(节选):

```python
tok_embeds = self.model.model.embed_tokens(input_ids)
inputs_embeds = self.fc(
    torch.cat([tok_embeds, previous_hidden_states], dim=-1))
inputs_embeds[positions == 0] = 0  # masking inputs at position=0
hidden_states = self.model.model(
    input_ids=None, inputs_embeds=inputs_embeds, ...)
```

## 性能声明

PR 正文无 benchmark 数字(素材正则未检出性能声明,正文仅说明功能与修复的 issue)。`token_map` 的文档注释声称可"减少采样开销、带来额外加速且基本不影响接受率",但未给出数字与测试条件。

## 验证与回归风险

新增单元测试 `test_expand_execute_model_request_sync_with_expand_hidden_states`:验证 seq_group 展开与 HiddenStates 的 bonus token 展开逻辑保持同步(k=5、batch=16、含 bonus token 的序列)。回归风险面:所有投机采样路径的 hidden states 传递语义被重写(`HiddenStates` 字段顺序与构造签名变更、`_run_no_spec` 中 scorer/proposer 顺序对调),影响 Medusa/EAGLE 及非 driver worker 的执行顺序;EAGLE 权重加载与参考实现/上游 checkpoint 格式存在兼容性要求(作者提供了转换脚本与兼容版 checkpoint);CUDA graph capture 路径新增 dummy hidden states buffer。

## 关联

同区域投机采样 proposer:Medusa(本 PR 顺带给 `Medusa` 类补充文档与 token_map 支持)。EAGLE 后续在 v1 的演化(DFlash/DSpark 等)见 PR #59511 一线。PR 正文仅引用 SafeAILab/EAGLE#43。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
