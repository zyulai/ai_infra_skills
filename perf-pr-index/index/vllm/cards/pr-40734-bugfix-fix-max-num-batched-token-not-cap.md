# PR #40734: [Bugfix] Fix max_num_batched_token not captured in cuda graph

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-04-29 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | bug, ready, nvidia |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/40734 |

## 一句话结论

把 `max_num_batched_tokens` 补进 CUDA graph capture sizes,避免最大 batch 落在非 stride 步长时回退 eager 造成约 3 倍性能损失。

## 动机

CUDA graph capture sizes 只包含落在 8 或 16 倍数步长上的 batch size。当 `max_num_batched_tokens` 不是 stride 的倍数时,该尺寸不会被 capture;模型跑到最大 batch 时只能回退 eager 模式,性能急剧下降。作者用 `max-cudagraph-capture-size: 2048` + `max-num-batched-tokens: 700` 的组合复现了这一骤降。

## 优化点

在 `_set_cudagraph_sizes` 中,若 `max_num_tokens`(即 max_num_batched_tokens)不超过 `max_cudagraph_capture_size` 且尚未在 capture 列表中,则将其追加进 `cudagraph_capture_sizes`,再去重排序。这样即使最大 token 数落在 16-stride 步长之间,也会被单独 capture 一张图,消除峰值 batch 的 eager 回退。

## 关键改动

```text
vllm/config/vllm.py: _set_cudagraph_sizes 追加 max_num_tokens 到 capture sizes
tests/compile/test_config.py: 新增 (2048, ..., 257) 用例验证 off-stride 也被 capture
```

## 性能声明

`nvidia/MiniMax-M2.5-NVFP4`,DP2 + EP,`max-cudagraph-capture-size 2048`,4096 prompts、random 1024/1024、max-concurrency 2048 的 `vllm bench serve`:`max-num-batched-tokens 700`(off-stride,eager 回退)与 `768`(on-stride)对比——duration 776.71s vs 252.07s,request throughput 5.27 vs 16.25 req/s,output token throughput 5400.07 vs 16639.32 tok/s,mean TTFT 210700.29 vs 60547.79 ms,mean TPOT 122.18 vs 42.23 ms。作者结论:"The difference in max-num-batched-tokens from 700 to 768 causes a 3x performance drop"。注意这是修复前 off-stride 配置的损失量级,PR 正文未给出修复后同配置(700)的对比数字。

## 验证与回归风险

测试:`tests/compile/test_config.py` 新增参数化用例(FULL_AND_PIECEWISE 下 2048/257 组合应 capture 到 257)。回归风险:capture sizes 列表可能新增一个非步长条目,增加少量 graph capture 数量与显存占用;仅影响 sizes 逻辑,不改变 kernel 行为;对依赖步长假设的下游(如 capture size 相关断言)需留意。

## 关联

同区域为 CUDA graph capture 尺寸策略;pack 正文未列出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
