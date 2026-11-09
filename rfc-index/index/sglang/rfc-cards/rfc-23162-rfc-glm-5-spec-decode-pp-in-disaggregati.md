# RFC #23162: [RFC] GLM-5 Spec. Decode + PP in Disaggregation Decode Mode

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23162](https://github.com/sgl-project/sglang/issues/23162) |
| 作者 | MMuzzammil1 |
| 创建日期 | 2026-04-19 |
| 状态 | ⚪ closed (2026-06-24) |
| labels | (none) |
| 热度 | 👍 4 · 💬 4 |
| 分类 | Disaggregated Serving |

## 一句话结论
提出在 PD 分离 decode 模式下让投机解码(EAGLE/MTP)与流水线并行 PP 共同工作的权重布局与执行流程设计,征求可行性反馈。

## 提案动机
作者计划支持 Speculative Decoding + PP in Disagg Decode Mode,分享行动方案并征求社区建议;正文明确提出的待反馈点包括:跨 PP rank 传递增量 KV 值的可行性,以及该实现的性能是否可观。

## 提出的方案
- 权重布局:投机模型(MTP 层)权重在 `first_pp_rank` 与 `last_pp_rank` 重复加载(即两处各有重复 `lm_head` 与 `embed`);target 模型按常规 PP 设置
- Draft 阶段在 first_pp_rank 执行,`draft_tokens` / `scores` 经已实现的 `PPProxyTensors` 发往各 PP rank
- 各 PP rank 各自 materialize `build_tree_kernel_efficient()` 得到 `EagleVerifyInput`;last_pp_rank 运行 `verify()` 得到 `accepted_tokens`,将 `verify_metadata` 回传 first_pp_rank
- last_pp_rank 另运行 `forward_draft_extend_after_decode()`;first_pp_rank 的 `_prep_batch_result` 调 `post_process_after_verify` 更新请求状态(finished/unfinished)并释放未用 KV slots,`verify_metadata` 再沿 PP-1..PP-2 逐级传递以调整各 rank 请求状态
- 计划扩展 `EagleWorker`(PP 本就不支持 overlap 模式,无需改用 `EagleWorkerV2`)
- 关键顾虑:`draft()` 与 `draft_extend_after_decode()` 需作用在同一 `batch` 上(两者分别在 first/last PP rank 执行);增量 KV 跨 rank 传递的可行性与整体性能待反馈

## 优化点与收益
正文未给出量化数字;目标为功能组合(投机解码 + PP + 分离 decode)的正确落地,性能被列为开放问题而非已验证收益。

## 现状与落地
已关闭(2026-06-24)。正文为设计分享与征询意见;落地情况需查关联 PR。

## 关联
正文未列出外部链接;正文提及相关文件 `python/sglang/srt/speculative/eagle_worker.py`、`python/sglang/srt/managers/scheduler_pp_mixin.py`、`python/sglang/srt/models/deepseek_nextn.py` 等。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
