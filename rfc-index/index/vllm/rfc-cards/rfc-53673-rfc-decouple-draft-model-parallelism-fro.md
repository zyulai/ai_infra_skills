# RFC #53673: [RFC]: Decouple draft-model parallelism from the target model (PCP/DCP + speculative decoding)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53673](https://github.com/vllm-project/vllm/issues/53673) |
| 作者 | LucasWilkinson |
| 创建日期 | 2026-08-25 |
| 状态 | 🟢 open |
| labels | RFC, k3, kimi, speculative-decoding |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
提议给 draft 模型独立的并行配置 `draft_parallel_config`,使其不再隐式继承 target 的 PCP/DCP 映射。

## 提案动机
vLLM 已支持 target 模型的 context parallelism(PCP #25749 与 DCP),但投机解码假设 drafter 与 target 用同一并行映射。两个在途 PR 路线不同:#53427 让全复制 drafter 在每个 PCP rank 上跑 global batch,简单通用,但每个 rank 重复 long-prefill draft 计算与 KV 写;#53653 让 MTP draft prefill 保持 PCP 分片、只在采样边界恢复全局序,长 prefill 性能更好且支持 GLM-5.2 sparse-MLA 路径,但每种投机方法都要单独适配。两者最终都可能需要:未实现 PCP/DCP 的 drafter(ngram、EAGLE 等)消费 global batch,有分片实现的(MTP)分片运行省去冗余计算。

## 提出的方案
- `SpeculativeConfig` 增加 `draft_parallel_config`(收编现有 `draft_tensor_parallel_size`),各并行度初始限 1 或 target 值;默认 per-method(MTP 继承 target 的 PCP size,无分片实现的方法默认 1),可经 `--speculative-config` 覆盖。
- drafter 启动时校验拿到的配置,无法执行则报清晰错误。
- 不新建 process group:drafter 仍在 target 的 ranks 上,size=1 即视为对该组复制(与今天 draft TP=1 under target TP=N 相同)。
- 已否决的替代:仅复制、仅分片、任意 drafter 并行(需要新分布式组与 rank placement,改动过大且无需求)。

## 优化点与收益
正文未给出量化数字;收益是长 prefill 的 draft 性能、对新/小众投机方法的通用 CP 支持,并避免在 model runner 里堆积 per-method 特例。

## 现状与落地
open,Feedback Period 约 2 周,尚无评论;留有 4 个开放问题:CUDA graph 支持承诺、global/sharded 是否应动态切换、sparse-MLA indexer 区分放 attention backend 还是 PCP manager、scheduler 是否需要知道 drafter 有效 CP 度。

## 关联
#25749、#53427、#53653、#53515

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
