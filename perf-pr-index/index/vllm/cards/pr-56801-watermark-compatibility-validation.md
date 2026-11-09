# PR #56801: [watermark] compatibility validation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @TQCB |
| 合并 | 2026-10-05 |
| 分类 | Other \| tier: major |
| Label | documentation, performance, frontend, ready, ci/build, mrv2, cohere |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 16 file(s) dropped from diff for size (see PR for full diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/56801 |

## 一句话结论

把 engine/request watermarking 兼容性策略集中到 `VllmConfig._check_supports_watermarking`，不兼容时告警一次并回退无水印生成。

## 动机

正文 Summary 给出的问题面：把 engine 与 request 的兼容性策略集中到 `VllmConfig._check_supports_watermarking(config) -> bool`；省略或 `true` 的 request watermarking 视为默认启用、仅 `false` 为显式退出；当 engine 或 request 模式无法应用水印时告警一次并回退到无水印生成；覆盖引擎缺少配置、greedy decoding、trace replay、beam search；请求侧解析集中到 engine admission 且只作用于 request-local 副本，避免复用的 caller parameters 把决策跨请求/跨引擎携带；校验支持的 speculative-decoding 方法并拒绝不兼容的 draft/rejection/parallel-drafting/custom-sampler 配置；保留显式 seed 与 `n>1` 并行采样；内部簿记请求（language detection、generative scoring、KV-transfer cleanup、mixed prefill/decode warmup）显式退出。

## 优化点

把原先 `_check_watermarking_unsupported(...)`（抛异常/仅校验）改成返回 bool 的 `_check_supports_watermarking(config)`：无 `watermark_config` 时若请求未显式 `false` 则 `logger.warning_once` 并返回 False；speculative_config 的 draft_sample_method 非 probabilistic 等不兼容情形仍校验；`BeamSearchParams` 请求改为告警并回退而非 `raise ValueError`；custom sampler 仍拒绝。`InputProcessor` 新增 `resolve_watermarking(params)` 调该 config 检查、`apply_watermarking(params, watermarking)` 把结果写到 request-local params 并记录 `_watermarking_skipped`；`SamplingParams` 与 OpenAI/Anthropic/Cohere/speech/realtime/batch/beam-search 等入口保留可选 request 值直到集中 admission 解析，`async_llm.py`、`llm_engine.py`、`gumbel.py`、`warmup.py` 与 benchmarks 脚本同步适配。

## 关键改动

```text
vllm/config/vllm.py: _check_watermarking_unsupported 改为返回 bool 的 _check_supports_watermarking(config)，无配置/beam search 等改为告警回退。
vllm/v1/engine/input_processor.py: 新增 resolve_watermarking / apply_watermarking，在 admission 集中解析并只写 request-local params。
vllm/sampling_params.py: watermarking 可选 request 字段（None/True/False）与 _watermarking_skipped 标记。
vllm/v1/engine/async_llm.py、vllm/v1/engine/llm_engine.py、vllm/v1/watermarking/gumbel.py、vllm/v1/worker/gpu/warmup.py: 引擎/worker 侧适配。
tests/watermarking/test_engine_compatibility.py、tests/v1/engine/test_input_processor_trace_replay.py 等: CPU 契约测试与 accelerator smoke 测试。
```

```python
    def resolve_watermarking(self, params: SamplingParams | BeamSearchParams) -> bool:
        return self.vllm_config._check_supports_watermarking(params)

    def apply_watermarking(self, params: SamplingParams, watermarking: bool) -> None:
        # Sticky, so resolving an already resolved request keeps the flag.
        params._watermarking_skipped = params._watermarking_skipped or (
            self.vllm_config.watermark_config is not None
            and params.watermarking is not False
            and not watermarking
        )
        params.watermarking = watermarking
```

## 性能声明

PR 正文未给出定量性能数据（无吞吐/延迟/benchmark 数字）；Test plan 仅给出测试计数：`139 passed, 15 skipped, 412 deselected`。

## 验证与回归风险

Test plan：focused impacted suite（排除本机未安装的可选 Cohere SDK）`139 passed, 15 skipped, 412 deselected`；所有 pre-commit hooks（Ruff、markdownlint、mypy）通过；Cohere 转换测试因可选 `cohere` SDK 未安装未在本地收集；accelerator smoke 模块可成功收集，CUDA/GPU 执行仍 pending。可能影响路径：OpenAI / Anthropic / Cohere / speech-to-text / realtime / batch / beam-search / 直接 engine 各入口的 watermarking 参数传递，sampling params、trace replay、KV-transfer rejection、mixed warmup gate、prefix caching、grammars、tool calling、TP/PP/DP、speculative decoding 组合，以及 benchmarks 脚本与 `.buildkite/test_areas/lm_eval.yaml`。pack 末尾注明 16 file(s) dropped from diff for size (see PR for full diff)。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
