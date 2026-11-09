# PR #37246: [Bugfix] dtype mismatch in ngram gpu propose

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37246](https://github.com/vllm-project/vllm/pull/37246) |
| 作者 | PatchouliTIS |
| 合并日期 | 2026-03-17 |
| labels | bug, ready, speculative-decoding, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
GPU ngram 提案器在 VLLM_COMPILE 下因 int32/int64 混算复用旧编译内核、错位读取数据,产出错误 draft token;统一转 int32。

## 问题现象
GPU ngram proposer 在 VLLM_COMPILE 模式(即 torch.compile 开启 drop_guards)下产生错误的 draft tokens。关联 issue #37150。正文未贴具体报错日志,错误表现为 draft token 内容不正确(后缀提取位置错误导致 ngram 匹配错误)。

## 根因
正文完整披露:dtype 不匹配发生在 warmup/tracing 与实际运行时之间——
1. warmup 阶段 `NgramGPUKernel.forward()` 被 trace 时 `num_tokens` 为 int32(`_generate_dummy_data` 第 354 行生成);
2. 运行时 `num_tokens_tmp = num_tokens_no_spec + valid_sampled_tokens_count`,其中 `num_tokens_no_spec` 是 int32(来自 gpu_model_runner.py:528),而 `valid_sampled_tokens_count` 是 int64(PyTorch 中 bool 张量 `.sum()` 默认返回 int64),按类型提升规则 int32 + int64 = int64;
3. drop_guards 模式下输入 dtype 从 int32 变 int64 不会触发 Inductor 重编译,复用按 int32 编译的内核处理 int64 数据;
4. int64 元素占 8 字节而编译内核按 4 字节(int32)stride 读取,交错读取命中的是每个 int64 的高 4 字节(小数值时全 0),得到后缀起始位置 `0 - ngram_len = -2`,后缀提取从此读错位置,导致 ngram 匹配与 draft token 错误。

## 修复方案
改动 `vllm/v1/spec_decode/ngram_proposer_gpu.py` 两处,显式把结果转回 int32:

```python
num_tokens_tmp = (num_tokens_no_spec + valid_sampled_tokens_count).to(
    torch.int32
)
...
valid_sampled_tokens_count = valid_mask.sum(dim=1).to(torch.int32)
```

使运行时张量 dtype 与 warmup 编译图保持一致,drop_guards 复用内核时不再错位读取。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响 `ngram_proposer_gpu.py` 的 `propose` 与 `update_token_ids_ngram` 两条路径的张量 dtype;count 值被截断为 int32,若单请求 token 数超出 int32 范围会溢出(由 diff 推断,实际场景远达不到该范围)。非 VLLM_COMPILE 模式行为不变。

## 关联
- https://github.com/vllm-project/vllm/issues/37150

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
