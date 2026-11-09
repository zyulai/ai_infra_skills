# RFC #20098: [RFC]: Lazy CUDA Graph capture

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20098](https://github.com/vllm-project/vllm/issues/20098) |
| 作者 | lionelvillard |
| 创建日期 | 2025-06-25 |
| 状态 | ⚪ closed(关闭于 2025-12-26) |
| labels | RFC, stale, startup-ux |
| 热度 | 👍 4 · 💬 20 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
把 CUDA Graph 捕获从引擎初始化改为按需惰性执行,缩短 vLLM 启动时间、避免无用捕获。

## 提案动机
目前 vLLM 在引擎初始化阶段捕获 cudagraphs,显著拖慢启动时间。默认会捕获 67 个图,依模型大小与 GPU 类型不同,这一步可能耗时超过 10 秒,UX 不佳(细节见 #19824)。此外,这 67 个图不太可能全部真正用到,既浪费时间也浪费空间。

## 提出的方案
把 cudagraphs 改为惰性捕获(lazy capture):不再在引擎初始化阶段执行 dummy runs,而是把这些 dummy runs 移到 CUDA piecewise backend 中进行,并且只在当前运行时 shape 尚未缓存时才捕获对应图。正文明言「Exact implementation needs to be worked out.」——具体实现尚待细化。

## 优化点与收益
收益方向为启动时间与资源:省去初始化时对 67 个图的批量捕获(现状成本照抄:依模型大小与 GPU 类型「can take more than 10s」),并避免为用不到的 shape 捕获图而浪费时间和空间。正文未给出改进后的量化数字。

## 现状与落地
已关闭(2025-12-26),labels 含 stale;正文极短且未含实现细节,看不出是否已实现或被替代,落地情况需查关联 PR。

## 关联
- [#19824](https://github.com/vllm-project/vllm/issues/19824)(启动 UX 问题的细节出处)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
