# PR #17731: [V0][Bugfix] Fix parallel sampling performance regression when guided decoding is enabled

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#17731](https://github.com/vllm-project/vllm/pull/17731) |
| 作者 | shadeMe |
| 合并日期 | 2025-05-23 |
| labels | structured-output, ready |
| 热度 | 👍 4 · 💬 11 |
| 分类 | Sampler / Detokenization |
| bug 类型 | regression |

## 一句话结论

并行采样按 n 次深拷贝 SamplingParams,引导解码 logits 处理器拷贝开销随时随 n 线性放大成热点;改用 clone 并为三个后端补齐 clone 方法。

## 问题现象

开启 guided decoding 时,并行采样(`SamplingParams.n > 1`)出现性能回归。基准环境:Nvidia L4、Phi-1.5、`n=3`、异步设置,180 秒 profile(预热后、含在途请求)中 `ParallelSampleSequenceGroup.add_request` 一调用就占总运行时间的 60%-86%(随 guided decoding 后端不同),改动后降至 0.01%-0.6%,瓶颈基本消除。

## 根因

正文披露:`ParallelSampleSequenceGroup.add_request` 需要按请求的样本数 n 复制原始 `SamplingParams`,当前用 `copy.deepcopy` 实现;而 `logits_processors` 字段可能携带带昂贵拷贝状态 的任意 Python 对象,现有 guided decoding logits 处理器正是如此,拷贝开销随 `SamplingParams.n` 线性增长,成为热路径瓶颈。此前针对类似问题已引入 `SamplingParams.clone` 规避——它尝试对每个 logits 处理器调用 `clone` 方法,让类用浅拷贝尽量降低开销——但并非所有现存处理器都实现了该方法,且 `ParallelSampleSequenceGroup` 也没有使用 `SamplingParams.clone`。

## 修复方案

- `vllm/sequence.py`:`ParallelSampleSequenceGroup.add_request` 中把 `params = copy.deepcopy(original_params)` 改为 `params = params.clone()`。
- `guidance_logits_processors.py`:为 `GuidanceLogitsProcessor` 新增 `clone()`——`copy.copy` 浅拷贝后,若已初始化则重建 `llguidance.LLMatcher` 并重新分配 token bitmask。
- `outlines_logits_processors.py`:为 `BaseLogitsProcessor` 与 `CFGLogitsProcessor` 新增 `clone()`,浅拷贝自身、复制 `_guide`、深拷贝 `_fsm_state`。
- `xgrammar_decoding.py`:完善已有 `clone()`,传入共享的 `tokenizer_info`(相应地 `__post_init__` 仅在 `tokenizer_info is None` 时才从 config 构建),并共享编译好的 grammar ctx、分离可变状态。
- `lm-format-enforcer` 后端未改,因其 logits 处理器实现位于 vLLM 之外。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围,属 V0 引擎并行采样 + guided decoding 路径。
- 回归面:影响所有 `n>1` 并行采样请求的 SamplingParams 复制路径:三个后端的处理器从 deepcopy 语义变为 clone 语义(guide/grammar 等不可变部分改为共享引用,仅可变状态独立),若某后端 clone 实现遗漏了本应隔离的可变状态,可能出现跨子请求的状态串扰;`lm-format-enforcer` 后端仍走 deepcopy,行为不变;xgrammar `__post_init__` 的条件构建也影响非 clone 的常规构造路径(仅当外部已传入 tokenizer_info 时行为不同)。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
