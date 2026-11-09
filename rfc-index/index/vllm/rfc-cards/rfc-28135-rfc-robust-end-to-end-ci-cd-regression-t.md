# RFC #28135: [RFC]: Robust End-to-End CI/CD Regression Testing for Speculative Decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28135](https://github.com/vllm-project/vllm/issues/28135) |
| 作者 | rahul-tuli |
| 创建日期 | 2025-11-05 |
| 状态 | ⚪ closed (关闭于 2026-03-11) |
| labels | RFC, stale |
| 热度 | 👍 6 · 💬 11 |
| 分类 | Speculative Decoding |

## 一句话结论
为投机解码(Eagle3 起步)建立端到端 CI/CD 回归测试:用版本化基线跟踪 acceptance rate 与加速比,把回退发现时间从数天缩到数小时。

## 提案动机
投机解码用 draft model 提前预测 token 并行验证,可将 LLM latency 提升 2-3x;但小的代码改动可能在正确性测试不失败的情况下破坏 acceptance rate 与 speedup。现有缺口:测试只查 66-80% match 阈值、不跟踪 acceptance rate 与实际加速比;nightly benchmark 无基线、无告警;MTP 变体覆盖不全;没有文档化的性能预期。

## 提出的方案
- 三层测试金字塔:PR smoke(每 PR,Eagle3+Llama-3.1-8B 单配置,5 min,L4)、Nightly(每日,4 核心变体×代表模型,45 min,L4+A100)、Weekly Deep Dive(每周,10 变体×5 模型全硬件矩阵,3 hours,L4+A100+H100)
- 三类测试:correctness(投机输出与非投机输出的 match rate)、quality(acceptance rate、avg drafts per step)、performance(TTFT/TPOT/E2E,p50/p99)
- 成功标准(每变体):match rate ≥ baseline;acceptance rate ±10%;TTFT/TPOT ±15%;无崩溃/OOM
- 基线管理:版本控制的 JSON 基线(人类可读),按 variant+model+hardware 组织,更新需 PR review,且只允许有意变更
- CI 资源优化:按 source_file_dependencies 智能触发、Buildkite parallelism 分片、渐进覆盖、未改动代码路径的结果缓存
- 分五阶段实施:Phase 0 基础(Eagle3 基线,Llama-3.1-8B 10-20 runs)→ Phase 1 Eagle3 全覆盖(3+ 模型+nightly CI)→ Phase 2 EAGLE/n-gram → Phase 3 MTP 变体与多模态等高级场景 → Phase 4 优化(<10% false positives、<15min nightly)

## 优化点与收益
照抄数字:目标是把回退发现从 "days" 缩短到 "hours";投机解码 latency 提升 2-3x;Eagle3 基线示例(L4,sharegpt):match_rate 0.75、acceptance_rate 0.73、ttft_p50_ms 44.0、speedup 2.1(容差 10-15%)。

## 现状与落地
已关闭,落地情况需查关联 PR。

> 注:pack 素材正文在 20k 字符处截断(Future Expansion 及之后内容缺失),本卡基于现有内容生成。

## 关联
正文未列出外部 RFC/PR 链接(引用的是现有测试、CI 配置与 benchmark 文件路径,如 `tests/v1/e2e/test_spec_decode.py`、`.buildkite/test-pipeline.yaml`)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
