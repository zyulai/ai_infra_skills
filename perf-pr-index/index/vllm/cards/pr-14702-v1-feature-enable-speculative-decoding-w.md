# PR #14702: [V1][Feature] Enable Speculative Decoding with Structured Outputs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2025-04-30 |
| 分类 | Speculative Decoding（tier: major） |
| Label | structured-output, ready, ci/build, v1 |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 8 个文件（scheduler、StructuredOutputManager、xgrammar/guidance 后端、gpu_model_runner、测试与 benchmark 脚本） |
| 链接 | https://github.com/vllm-project/vllm/pull/14702 |

## 一句话结论

打通 V1 结构化输出与投机解码：grammar 校验 draft token、按投机位置生成 bitmask 并应用到 logits，使带 grammar 约束的请求也能吃投机加速。

## 动机

V1 中 structured output 与 speculative decoding 互斥：draft token 未经过 grammar 校验，bonus token 的采样也没按 grammar 约束 mask，启用投机解码会破坏结构化输出的合法性。此前 gpu_model_runner 中留有 "TODO: compatibility with spec decode"，xgrammar 的 `GrammarMatcher` 也未配置 `max_rollback_tokens`。

## 优化点

三段机制：(1) **draft 校验**——scheduler 在 `update_from_output` 中对新 draft token 调 `grammar.validate_tokens()`（新增抽象方法：逐 token 试接受后 rollback，不改 FSM 状态），只保留合法前缀作为 `request.spec_token_ids`，且在正式 `accept_tokens(new_token_ids)` 之后执行以保证状态正确；(2) **多位置 bitmask**——`grammar_bitmask()` 签名从 `batch_len` 改为接收 `scheduled_spec_decode_tokens`，对每个结构化输出请求按"每个投机位置 + 1 个 bonus 位置"逐位置生成 mask（每前进一个投机位置就 `accept_tokens` 一次，最后统一 `rollback(state_advancements)`），xgrammar 侧 `GrammarMatcher` 以 `max_rollback_tokens=num_speculative_tokens` 编译；bitmask 按"压缩 + 投机位置交织"的方式打包进 `max_batch_size * (1 + max_num_spec_tokens)` 行的缓冲区，非结构化请求被跳过；(3) **GPU 侧解包**——`apply_grammar_bitmask` 按 batch 顺序重排 bitmask 行并计算每个投机位置的 logit 偏移，一次性 `apply_token_bitmask_inplace` 应用到所有相关 logit 行。兼容 xGrammar 与 Guidance 两个后端（guidance 侧新增 `validate_tokens`/`rollback`）。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: draft token 经 validate_tokens 过滤后再挂到 request
vllm/v1/structured_output/__init__.py: grammar_bitmask 按投机位置逐位生成并压缩交织打包，FSM 前进后统一 rollback
vllm/v1/structured_output/backend_xgrammar.py: GrammarMatcher 开启 max_rollback_tokens；新增 validate_tokens/rollback
vllm/v1/structured_output/backend_types.py: StructuredOutputGrammar 抽象类新增 validate_tokens/rollback 抽象方法
vllm/v1/worker/gpu_model_runner.py: apply_grammar_bitmask 重排 bitmask 并按投机位置偏移应用
```

scheduler 侧 draft 校验：

```python
if request.use_structured_output:
    metadata = request.structured_output_request
    assert metadata is not None and metadata.grammar is not None
    # Needs to happen after new_token_ids are accepted.
    request.spec_token_ids = metadata.grammar.validate_tokens(
        spec_token_ids[req_index])
```

## 性能声明

PR 正文未给出量化性能数字（正则亦未检出）。功能性收益：修复 `benchmark_serving_structured_output.py` 后，投机解码开启时 benchmark 达到 100% correctness（temperature==0 时原 prompt 会诱导无限重复）。作者注明 bitmask 外层循环可并行化以优化大 batch 下的生成开销，属已知未做优化。

## 验证与回归风险

`tests/v1/entrypoints/llm/test_struct_output_generate.py` 参数矩阵从 4 个组合扩到 7 个：新增 ngram 投机 + xgrammar/guidance、EAGLE（`yuhuili/EAGLE-LLaMA3.1-Instruct-8B`）+ xgrammar 三组，覆盖 JSON schema 测试全流程。风险面：(1) validate_tokens 对 xgrammar 的实现是"逐 token accept 再整段 rollback"，高频调用会增加 FSM 操作次数（作者已注明外层循环待并行化）；(2) bitmask 缓冲区大小变为 `max_batch_size * (1 + max_num_spec_tokens)`，显存/序列化体积随投机长度线性增长；(3) gpu runner 的 bitmask 解包在投机未启用时也要执行（compact 格式跳过非结构化请求），非投机路径有少量额外开销；(4) ngram 与 EAGLE 两条投机路径均进入测试，guidance 后端依赖 `ll_matcher.validate_tokens`/`rollback` 原生能力。

## 关联

同分类：#13365（V1 n-gram matching 优化）、#14930（V1 rejection sampling 优化）、#16937（EAGLE-3 support）均为 V1 投机解码线；本卡补齐其与 structured output 的交集。素材未列出更直接的关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
