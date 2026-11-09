# RFC #50438: [RFC]: Lookahead-aware prefix-cache hashing for EAGLE-style draft models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50438](https://github.com/vllm-project/vllm/issues/50438) |
| 作者 | skajre |
| 创建日期 | 2026-07-30 |
| 状态 | 🟢 open |
| labels | RFC, stale |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Speculative Decoding |

## 一句话结论
把 EAGLE 式草稿模型的边界依赖 token 纳入 prefix-cache 哈希键,命中即证明边界一致,从而免去保守的末块丢弃重算。

## 提案动机
vLLM 将消费 target 模型隐藏状态的投机方法归入 `SpeculativeConfig.use_eagle()`。前缀缓存查找时,这些 EAGLE 式方法保守地丢弃最后匹配单元并重算:当前缓存键只证明块内 token 相等,不能证明紧随边界的 target token 相等,而它正是草稿输入的一部分(`x1 x2 x3 x4` 的移位输入)。对 hybrid 全注意力 + Mamba/GDN 模型代价尤其大——对齐缓存单元可含数百上千 token,丢一个单元会抹掉大部分前缀缓存收益。已有 #38182、#44858 两份报告。

## 提出的方案
1. 每个 EAGLE 式块哈希纳入一个后随 token:`H_i_eagle = hash(H_(i-1)_eagle, token_ids[s:e+1], extra_keys[s:e+1])`。lookahead token 是 hash 边界后第一个 finalized 的 prompt/target token,不是未验证的草稿 token,与 `num_speculative_tokens` 无关;现有 cache salt、LoRA 身份、多模态 extra-key 语义保持不变。
2. 延迟发布:块需 B+1 个相关 token 才 hash-ready;请求恰好在边界结束则末块保持 pending,直到 target finalize 下一个 token 且对应 draft-cache 写入完成后才可发布;缓存管理器不得发布超出可用块哈希数的条目。
3. 命中即免 pop:命中 `H_i_eagle` 证明前缀、target 派生状态、边界 token 均相同,scheduler/KV coordinator 可跳过 EAGLE 式末块丢弃;后随 token 未知或不同时保持现有保守行为(退回更早的缓存命中)。
4. 移除对应的 Mamba align-mode 一块回退——它只是为与 speculative cache pop 保持一致而存在的调度器回退,不改变 Mamba 状态的物化与保留方式。
5. 首版启用条件:prefix caching 开启 AND `use_eagle()` 为真 AND 无 kv_transfer_config AND KV 事件禁用;覆盖 eagle/eagle3/mtp/dflash/dspark 五种方法,逐个验证 one-token 依赖与缓存发布顺序后才启用 no-pop 路径;P/D 分离、KV 连接器、可续传请求明确排除在首版之外。正确性不变量:只有哈希键证明块内全部 target 派生与边界 token 输入相等且 draft-cache 写入先于发布,才能免 pop 复用,否则保留现有 pop 回退。

## 优化点与收益
照抄正文数字:一份 Qwen3.6 报告(vLLM 0.20.1,#38182 评论)测得前缀缓存命中率从 87.3% 降至 69.9%,每请求少命中 1,056 个 token——约一个 1,072-token 的对齐块;正文注明这是 vLLM 0.20.1 的历史证据,current main 已含部分前缀命中改动,剩余重算成本需重新测量。预期收益:边界 token 匹配时恢复一个投机丢弃单元、不匹配时无命中率回退;hybrid 模型上避免一次丢弃可能同时避免更大规模的 Mamba 对齐重算。CPU 哈希开销约为 token 负载的 `1 / B` 量级,预计相对避免模型重算可忽略,但仍需实测。

## 现状与落地
仍在讨论/等待评审(open,反馈期至少一周)。已有本地原型与 CPU 回归测试(前缀缓存、partial-hit、哈希、流式调度器测试均通过);GPU/模型评估与性能测量按计划在提交实现 PR 之前完成。备选方案中基于 prefill/decode 阶段跳过 pop 的 #44986 与本方案的关系(相位判断 vs 内容证明)已提请维护者反馈。

## 关联
- [#38182](https://github.com/vllm-project/vllm/issues/38182)(MTP 前缀缓存命中率回退,动机证据)
- [#44858](https://github.com/vllm-project/vllm/issues/44858)(prefill 阶段 pop 优化讨论)
- [#44986](https://github.com/vllm-project/vllm/pull/44986)(基于阶段的 skip-drop PR,备选方案)
- [#43996](https://github.com/vllm-project/vllm/issues/43996)(P/D lookahead 分配与裁剪失败案例)
- [#43559](https://github.com/vllm-project/vllm/issues/43559)(hybrid MTP 正确性跟踪)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
