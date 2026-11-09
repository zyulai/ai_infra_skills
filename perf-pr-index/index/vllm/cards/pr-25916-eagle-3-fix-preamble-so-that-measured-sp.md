# PR #25916: EAGLE 3: Fix preamble so that measured speedup over Eagle 1 becomes 32% instead of 5% on MTBench

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ekagra-ranjan |
| 合并 | 2025-10-02 |
| 分类 | Speculative Decoding (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 仅含 `vllm/benchmarks/datasets.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/25916 |

## 一句话结论

修 benchmark 基准数据的前置 token 缺陷:MTBench 上 EAGLE3 相对 EAGLE1 的实测加速从 5.6% 修正为 32.9%。

## 动机

issue #20780:E3 的 acceptance length 比 E1 高 20%,但 e2e TOPS 只高 4%,远低于预期。排查发现 offline 推理在 MTBench 上 E3 的 AL 为 2.79,而 online(绕过 prometheus 指标 reset 显示整体 AL)只有约 2.2。根因是数据构造:offline 与 online 用同一数据集,但 offline 设 `add_special_tokens=False`,online 对 llama 3.1 按 model config 设 True——online 的 prompt 开头出现两个 `<|begin_of_text|>`(chat template 一个 + tokenizer.encode 一个)。这个微小差异足以让 E3 的 AL 骤降,而 E1 不受影响,所以 E1 ablation 期间从未暴露。

## 优化点

这本质是基准方法论修复而非内核优化。正确做法是数据集 builder 跳过 chat template,用 `/v1/chat/completions` 端点(服务端按需应用模板),避免 tokenizer 端再叠加 special token。实现上:`vllm bench serve` 新增通用 `--skip-chat-template` 参数(替代只作用于 CustomDataset 的 `--custom-skip-chat-template`),并把它传给 HF 数据集、ShareGPT 类数据集等各 sample 路径;移除 ShareGPT `sample()` 中强制 `kwargs["skip_chat_template"] = False` 的覆盖;各数据集的 `sample` 在 `not skip_chat_template` 时才 `apply_chat_template`。

## 关键改动

```text
vllm/benchmarks/datasets.py: 新增通用 --skip-chat-template;HFDataset/
  ShareGPT 等各 sample() 按开关跳过 apply_chat_template;移除强制覆盖
```

关键片段(各数据集 sample 的模式):

```python
if not skip_chat_template:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
```

## 性能声明

TPOT on MTBench BS4 on H100,EAGLE3 drafter `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`,num_speculative_tokens 3,Llama-3.1-8B-Instruct,V1 引擎:`vllm bench serve` 用 philschmid/mt-bench、80 prompts、max-concurrency 4、temperature 0、openai-chat backend、`--skip-chat-template`。E1:4.32 ms;E3 修复前:4.09 ms(比 E1 快 5.6%);E3 修复后:3.25 ms(比 E1 快 32.9%)。

## 验证与回归风险

验证方式是前后对比同一 benchmark 命令,并确认 online/offline AL 一致(2.79)。风险:该参数改变各数据集 prompt 的默认构造可选行为——默认 `skip_chat_template=False` 不影响既有用法;但 `--custom-skip-chat-template` 被移除改名,依赖旧参数名的脚本会破坏。此 PR 不改引擎/解码路径本身,无推理正确性风险。

## 关联

修复 issue #20780。EAGLE 系列同分类:#22872、#24505、#5765、#13365。属于"基准正确性影响性能结论"的方法论案例,与 #22872 引入 MMStar 多模态基准数据集同在 `vllm/benchmarks/datasets.py`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
