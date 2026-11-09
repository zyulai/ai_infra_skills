# PR #4947: [Model] MLPSpeculator speculative decoding support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JRosenkranz |
| 合并 | 2024-06-21 |
| 分类 | Speculative Decoding（tier: major） |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/4947 |

## 一句话结论

接入 IBM fms 的 MLPSpeculator 复用 spec_decode 框架，离线 batch_size=1 示例 86.03→133.83 toks/s。

## 动机

vLLM 已有 speculative decoding 框架，但缺少一个"用户几乎无需改动即可显著提升推理性能"的 speculator。本 PR 把 IBM fms 的 MLPSpeculator 纳入：架构类似 medusa，关键差别是每个 head 的输出以前一个 head 为条件，从而生成更成形的 n-gram、提高 acceptance rate（技术报告 arXiv:2404.19124）。

## 优化点

复用 `spec_decode_worker`，以 `MultiStepWorker`（proposer，背后是 MLPSpeculator）配 `HiddenStatesWorker`（scorer）。`MLPSpeculator` 作为独立模型，每次 forward 只跑一层 head，靠 head counter 递增选择当前 head，并把 hidden states 存为下一步输入；`HiddenStatesWorker` 覆写 `execute_model` 保存 base model hidden states 并注入 MLPSpeculator；`SingleStepSpeculativeModelRunner` 单步跑 speculator 且只计算一次输入。因 base model hidden states 未剪枝，生成新候选前需按每请求 accepted tokens 数（由前后 context length 之差得到）剪枝。

## 关键改动

```text
vllm/model_executor/models/mlp_speculator.py: 新增 MLPSpeculator 模型与 generate_proposals
vllm/spec_decode/mlp_speculator_worker.py: 新增 MultiStepWorker 用的 speculator worker
vllm/worker/worker.py + model_runner.py: HiddenStatesWorker 保存/注入 hidden states
vllm/config.py + transformers_utils/configs/mlp_speculator.py: 配置与模型注册
examples/offline_inference_mlpspeculator.py: 离线推理示例
```

```python
        for head_index in range(num_predict_tokens):
            z = self.emb[head_index](last_tokens)  # b k d
            states = self.proj[head_index](previous_hidden_states)
            states.add_(z, alpha=self.emb_weight / self.state_weight)
            states = self.activation(self.ln[head_index](states))  # b k d
            previous_hidden_states = states
            logits = self.logits_processor(self.head[head_index].weight,
                                           states, sampling_metadata)
            output = self.sampler(logits.flatten(0, 1), sampling_metadata)
            last_tokens = output.sampled_token_ids
            next_tokens.append(output)
```

## 性能声明

PR 正文 offline example（batch_size=1）：Non-Speculative `1.62it/s, Generation Speed: 86.03 toks/s`；Speculative `2.52it/s, Generation Speed: 133.83 toks/s`。PR 正文未注明测试条件（模型/硬件/精度/workload）；`examples/offline_inference_mlpspeculator.py` 使用 `meta-llama/Llama-2-13b-chat-hf`、`temperature=0.0`、`max_tokens=200`。除该示例外 PR 正文未给出其他定量性能数据。

## 验证与回归风险

PR 正文未列专门测试命令；diff 含 `tests/spec_decode/test_spec_decode_worker.py`、`tests/spec_decode/test_utils.py` 与离线示例。风险面：新增 HiddenStatesWorker 需正确保存/重置 base model hidden states、head counter 每轮归零，并在下一次 `get_spec_proposals` 前按 accepted tokens 剪枝 hidden states；复用 `MultiStepWorker`/`spec_decode_worker` 与 `models/__init__.py`、`transformers_utils/configs` 注册，可能波及既有 spec decode 路径。PR 自述"improve inference performance significantly"但未附更多 quantitative 评测。

## 关联

无（PR 正文仅引用 IBM fms、技术报告 arXiv:2404.19124 与若干 accelerator 模型，未列关联 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
