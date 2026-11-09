# PR #51318: [Bugfix][DSv4] Revert adaptive C128A metadata packing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51318](https://github.com/vllm-project/vllm/pull/51318) |
| 作者 | tobymao |
| 合并日期 | 2026-08-16 |
| labels | bug, ready, verified |
| 热度 | 👍 4 · 💬 11 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#50004 的自适应 C128A top-k 宽度/打包缓冲视图与 CUDA graph 捕获时的行跨度不一致,后续行可能读到陈旧偏移;整体回退恢复容量跨度布局。

## 问题现象
DSv4 稀疏解码路径下,运行时元数据行与被捕获的消费者读取布局错位,导致输出错误(issue #52448)。作者用 #52448 的精确请求在 2x NVIDIA GB300、`deepseek-ai/DeepSeek-V4-Flash-0731`(revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`)、TP=2 + 专家并行、DSpark-7、`FULL_AND_PIECEWISE`、`max_model_len=70000`、并发 32、`max_tokens=16384` 验证回退:256 条 priming + 256 条测量请求,issue 特征命中 0/256,0 请求错误、0 `length` 结束,253 `tool_calls` + 3 `stop` 结束,中位时延 8.445 s。

## 根因
正文披露:C128A 元数据构建器在 FULL CUDA graph replay 之前运行,而稀疏解码消费者是被捕获的,会复用捕获时的行布局。#50004 让运行时元数据按随 batch 变化的 `active_topk_width` 写打包行,但被捕获的消费者仍保留捕获时的行跨度——第 0 行仍从偏移 0 开始,后续行却可能从陈旧偏移读取。

## 修复方案
回退 #50004 的打包布局,改 `vllm/models/deepseek_v4/sparse_mla.py`:
- `_build_c128a_metadata` 删除 `active_topk_width` 计算,`max_compressed_tokens` 恢复为 `self.c128a_max_compressed`。
- `build_c128a_topk_metadata` 恢复容量跨度切片,不再对缓冲区做 `view(-1)[...]` 打包:

```python
global_decode = global_decode_buffer[:num_decode_tokens]
decode_lens = decode_lens_buffer[:num_decode_tokens]
prefill_local = prefill_buffer[:num_prefill_tokens]
```

- Triton kernel 的行宽参数恢复用各持久缓冲的 `stride(0)`,而非 `max_compressed_tokens`。
- 删除断言打包布局的测试 `test_deepseek_v4_c128a_dynamic_topk_packed_buffers`。

## 影响与回归风险
- 影响窗口:受 #50004 之后到本回退之间的提交影响(正文明确引入 PR 为 #50004)。
- 回归面:仅触及 DeepSeek-V4 的 C128A 稀疏 MLA 元数据构建与对应测试文件;恢复为容量跨度布局意味着放弃 #50004 试图带来的按 batch 自适应宽度(若有性能收益则一并放弃)。验证为 GB300 上 #52448 场景全量复跑(见上),pre-commit 各钩子通过。

## 关联
- Fixes #52448
- #50004(被回退的引入 PR)
- #48805、#41836、#41834(关键词相关,但分别处理 warmup、prefill、模型启用,非本回退范围)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
