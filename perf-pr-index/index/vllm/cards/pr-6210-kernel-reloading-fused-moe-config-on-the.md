# PR #6210: [Kernel] reloading fused_moe config on the last chunk

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @avshalomman |
| 合并 | 2024-07-08 |
| 分类 | Attention / Kernel \| tier: major |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/6210 |

## 一句话结论

fused_moe 最后一个 chunk 按实际 token 数重载 kernel config，平均约 5% 加速。

## 动机

这是 #6029 引入的 `fused_moe` chunking 机制的 follow-up PR。正文指出：当最后一个 chunk 小于 chunk size 时，按最后一个 chunk 的大小重新加载 kernel configuration 会有帮助。正文以一张性能对比截图（plot）示意差异，未给出逐项数值或更多背景，也未进一步描述此前的具体瓶颈或性能分解。

## 优化点

把原先内联在 `fused_experts` 中的配置选择逻辑抽成 `try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M, override_config)`：优先用 `override_config`，否则读 `get_moe_configs(E, N, dtype)` 并以 `min(configs.keys(), key=lambda x: abs(x - M))` 取最接近 M 的配置，无配置文件时回退 `get_default_config(M, E, N, w1_shape[2], top_k, dtype)`。`fused_experts` 用 `functools.partial` 绑定形状与 dtype 构造 `get_config_func`，首 chunk 调 `get_config_func(M)`，之后在每个 chunk 缩小 `intermediate_cache1/2/3` 后按 `tokens_in_chunk` 重新调用，使较小的 last chunk 用上与其实际大小匹配的 config。kernel 本身不变，只改配置选择的时机与位置。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: 新增 try_get_optimal_moe_config，抽取并复用原 config 选择逻辑
vllm/model_executor/layers/fused_moe/fused_moe.py: fused_experts 用 functools.partial 构造 get_config_func，并在 last chunk 处按 tokens_in_chunk 重载 config
```

## 性能声明

正文称"In our experiments, this can yield a ~5% speedup on average"，并附一张性能对比截图（plot）说明差异；pack 的 regex 性能声认为 `5% speedup`。PR 未注明测试条件（未给出模型、硬件、精度、workload 的具体信息），也未给出可核对的逐项数值。

## 验证与回归风险

PR 正文未注明测试计划、测试结果或回归风险。改动集中在 `fused_experts` 的配置选择时机，可能影响所有走 `fused_moe` chunking 的路径；正文未列出具体受影响的模型、硬件或精度范围。相关 chunking 机制来自前序 PR #6029。

## 关联

正文明确提到 #6029（引入 `fused_moe` chunking 机制的前序 PR）。除此之外无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
