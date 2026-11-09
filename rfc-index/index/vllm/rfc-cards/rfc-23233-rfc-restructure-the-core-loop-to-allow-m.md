# RFC #23233: [RFC]: Restructure the core loop to allow more asynchrony

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23233](https://github.com/vllm-project/vllm/issues/23233) |
| 作者 | WoosukKwon |
| 创建日期 | 2025-08-20 |
| 状态 | ⚪ closed (关闭于 2025-11-22) |
| labels | RFC, stale |
| 热度 | 👍 10 · 💬 10 |
| 分类 | Architecture / Engine |

## 一句话结论
提议把采样从 execute_model 中拆出,让核心循环在 GPU 执行模型期间重叠 CPU 侧工作(采样元数据与 grammar bitmask)。

## 提案动机
当前核心循环是 schedule → execute_model → update_from_output 的串行结构;虽然简单,但 GPU 在 execute_model 中执行模型时,CPU cycles 得不到利用。

## 提出的方案
- 把 sampling 阶段从 execute_model 中拆出:execute_model 变为非阻塞调用,结尾不再做 GPU→CPU 同步,只执行到最后的 hidden states
- GPU 执行期间,scheduler 侧构造 grammar bitmask(使用结构化输出时,否则为 None);worker 侧可重叠构造 sampling metadata(与 logits processor)
- 新增 `sample(bitmask)` 阻塞调用,准备采样元数据并采样下一步 token ids,产出 model_runner_output,再由 scheduler.update_from_output 生成输出
- async scheduling 版核心循环:先做 initial step(schedule → prepare_inputs → execute_model → bitmask);循环内 schedule → prepare_inputs → sample(上一步 bitmask)→ execute_model → 新 bitmask,并用 prev_scheduler_output 更新输出,形成流水

## 优化点与收益
正文未给出量化数字。两条定性收益:(worker 侧)把 sampling metadata(与 logits processor)的构造与 execute_model 重叠;(scheduler 侧)把 bitmask 构造与 execute_model 重叠。

## 现状与落地
已关闭。正文的反馈期与 CC 均为「No response」,未记录讨论结论,落地情况需查关联 PR。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
