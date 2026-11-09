# PR #20859: [Feature] limit thinking tokens (hard limit)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @llsj14 |
| 合并 | 2026-03-24 |
| 分类 | Other |
| Label | documentation, performance, new-model, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/20859 |

## 一句话结论

为推理模型加 `thinking_token_budget` 硬预算:超限即在 logit 层强制输出 think 结束 token,截断失控长推理以降低无效输出 token。

## 动机

此前控制 thinking token 要么依赖模型自带 soft limiting(如 gpt-oss 的 `reasoning_level`),要么发两次 API 调用先推理后作答;两次调用在没有 prefix-aware routing 时无法保证路由到同一节点,且基于 prompt 的软限制常导致重复推理内容,拉低输出质量并浪费 token(服务团队反馈的痛点)。关联 issue:#15418、#17887、#25714、#26091。

## 优化点

新增内置 `ThinkingTokenBudgetLogitsProcessor`:为每个带 budget 的请求维护状态(`in_think`/`in_end`/`think_count` 等),增量扫描新产出 token 检测 think start/end 序列(支持多 token 边界);`think_count` 达到 budget 后进入 `in_end`,在 `apply` 中对 end token id 的 logit 加 1e9 强制采样该 token。检测逻辑先在 CPU 上判断是否需要改 logits,避免 GPU sync。配置侧新增 `ReasoningConfig`(`think_start_str/think_end_str` 经 tokenizer 转为 token id,`VllmConfig.__post_init__` 初始化),`SamplingParams` 与 OpenAI chat 请求透传 `thinking_token_budget`;未配置 `--reasoning-config` 时 processor 完全不生效。

## 关键改动

```text
vllm/v1/sample/logits_processor/builtin.py: ThinkingTokenBudgetLogitsProcessor(状态跟踪 + 强制 end token)
vllm/config/reasoning.py: 新增 ReasoningConfig,边界字符串转 token id
vllm/sampling_params.py, vllm/entrypoints/openai/chat_completion/protocol.py: 透传 thinking_token_budget
vllm/v1/worker/gpu_model_runner.py: reasoning_config 非空时 logitsprocs_need_output_token_ids=True
```

## 性能声明

开销测试:DeepSeek-R1-Distill-Qwen-1.5B,max_tokens=512、min_tokens=512、thinking_token_budget=20,中位数开销"几乎为零"(PR 附图);budget 更大时(硬限叠在软限之上)开销更小。未给出 e2e 吞吐/时延对比数字。

## 验证与回归风险

新增 `tests/v1/entrypoints/openai/test_thinking_token_budget.py`(Qwen3-0.6B e2e,验证 budget 内 token 数精确等于设定值)并扩展 `tests/v1/logits_processors/test_correctness.py`(与其他 processor 隔离测试)。风险点:processor 声明 `is_argmax_invariant()=False`,会改变贪心采样结果;`logitsprocs_need_output_token_ids` 在配置 reasoning 后恒为 True,使 async scheduling 保守地维护 output token ids,可能影响相关优化路径。

## 关联

同区域:Nemotron 3 的 budget control(多次 API 调用方案,见 PR 引用);gpt-oss soft limiting(对照)。speculative decoding label 存在但 diff 未涉及 spec decode 逻辑。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
