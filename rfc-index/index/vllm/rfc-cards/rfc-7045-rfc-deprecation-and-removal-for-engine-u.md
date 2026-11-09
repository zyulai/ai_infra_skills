# RFC #7045: [RFC]: Deprecation and removal for `--engine-use-ray`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7045](https://github.com/vllm-project/vllm/issues/7045) |
| 作者 | simon-mo |
| 创建日期 | 2024-08-01 |
| 状态 | ⚪ closed(关闭于 2024-08-14) |
| labels | RFC |
| 热度 | 👍 18 · 💬 4 |
| 分类 | Architecture / Engine |

## 一句话结论
提议废弃并移除 `--engine-use-ray` 标志:用 Ray 启动引擎进程的方案已被更低开销的进程分离方式取代。

## 提案动机
在 `async_engine` 代码路径中,`--engine-use-ray` 允许用 Ray 在独立进程启动 LLM 引擎,初衷是把服务端 Python 开销与引擎主调度循环分离。但若干因素使该选项变得少用/不受欢迎:Ray 是可选组件,单节点环境通常不使用;序列化与 RPC 开销通常抵消理论上的性能收益;隔离服务端与引擎通常有其他方式(multiprocessing、threading 等);且社区正以更低开销的方式在服务端做这种分离(关联 #6883)。

## 提出的方案
- 先废弃:保留该标志一个版本周期并给出弃用警告
- 再移除:若无重大反对(pushbacks)则删除该标志
- 反馈期:1 周

## 优化点与收益
正文未给出量化数字。收益为减少维护面,并让服务端/引擎分离转向更低开销的方案(见 #6883)。

## 现状与落地
已关闭(2024-08-14,距创建约两周,与 1 周反馈期节奏吻合)。正文未记录最终执行情况,落地情况需查关联 PR。

## 关联
- #6883:https://github.com/vllm-project/vllm/issues/6883

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
