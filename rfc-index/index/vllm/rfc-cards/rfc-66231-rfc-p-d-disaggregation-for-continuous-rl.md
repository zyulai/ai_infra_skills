# RFC #66231: [RFC]: P/D disaggregation for continuous RL rollouts (S1): order-free, seconds-long weight-update pauses

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66231](https://github.com/vllm-project/vllm/issues/66231) |
| 作者 | BeichenHuang |
| 创建日期 | 2026-11-07 |
| 状态 | 🟢 open |
| labels | kv-connector |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Disaggregated Serving |

## 一句话结论
为 P/D 分离下的连续(全异步)RL rollout 定义 order-free、秒级完成的权重更新暂停契约,并给出三个调度器侧结构。

## 提案动机
全异步 RL 框架(如 prime-rl,生产环境用 GLM-5.x + NIXL、xPyD、DP/EP)在 rollout 仍在生成时更新策略权重:P/D 引擎 `pause(mode="keep", clear_cache=False)` → 广播权重 → `update_weight_version` → `resume`。但在 main 上:①P 的 keep 暂停要等 D 读完所有导出(或 lease 过期);②暂停的 D "go dark",停止续租(NIXL/Mooncake 的续租都走 `build_connector_meta`,只在 step 内运行);③传输中 abort 会杀死 EngineCore 或让 D 的 pull 悬空。prime-rl 目前靠"先停所有 P 再停 D"与 monkey-patch 绕过。

## 提出的方案
- 契约四条:Order-free(任意顺序 pause/resume,包括并发)、Bounded(pause 在一个 engine step + device sync 内完成)、Safe(任何时长暂停下 KV 块不被释放/复用,resume 后无请求读到失效 KV)、Isolated and accounted(新 `cache_salt` 对旧 KV 零命中)。
- P1 传输账本 `vllm/v1/core/sched/kv_transfer_ledger.py`:按 (request, direction) 记 EXPORT/IMPORT 条目,块仅在无条目引用时释放,迟到 completion 按构造容忍(#46240)。
- P2 按模式键控的 quiescence 表 `Scheduler.quiesced(mode, clear_cache)` 取代分散分支;resume 使挂起 pause 失败。
- P3 账本驱动步进:`ledger.needs_driving()` 时引擎零计算继续 step,暂停中的 D 续租、P 观察完成,任一 connector 都不因暂停过期 lease,无需改 connector 代码。
- 分为独立 PR(step 0–6 + Mooncake 修复 M),附 1P1D GPU 测试矩阵、S2/S3 探针与 stale-read copy-task 探针。

## 优化点与收益
条件:单 H20 节点、1P1D(各一卡)、Qwen3-8B、RoCE、D 限 `--max-num-seqs 32`、lease 30 s、600 请求并发 96。A(结构)方案 S1 矩阵在 Mooncake 与 NIXL 均 8/8,P 暂停 0.9–1.7 s、D 暂停 0.03 s、0 次 lease 过期(含 45 s 暂停);main+NIXL 仅 4/8(P 暂停 31 s、19 个 lease 过期);B(胶水)需 `kv_unclaimed_grace=300` 才 8/8。无暂停负载三者均 56.5–57.1 s。Mooncake `dpause` 陈旧读探针:main 12 errors,A 12/12 correct。另引 prime-rl #4059:仅排序 workaround 就把最慢暂停从 30 s 降到 14.3 s、"Potentially invalid KV blocks" 从 354 降到 0。

## 现状与落地
仍在讨论/已有原型:open 状态,代码分支 `s1-ledger` 与测试 harness 已就绪,计划拆成多个 PR 落地;正文如实列出待评审项(NIXL abort 后 stale-read 安全只被探针证明未解释、一次 `resume_during_wait_D` 卡死未 root-cause、D-only 长暂停会占满 P 的 KV 等)。注:pack 正文在 20k 字符处截断,"What the other #62150 scenarios reuse" 表之后的内容未收录,本卡基于现有内容成卡。

## 关联
- [s1-ledger 分支](https://github.com/BeichenHuang/vllm/compare/5cdeea9b13670c9da832d8ea72d799c8bfbb4552...s1-ledger)
- #62150(三种 RL 暂停场景的总追踪)
- #46240、#59670(相关缺陷/行为)
- #64865 / #64900(S3 前置)、#65672 / #65787(Mooncake 相关)
- prime-rl #4059(排序 workaround 数据来源)
(以上编号正文均未附链接)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
