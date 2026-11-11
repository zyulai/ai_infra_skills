# RFC #49536: [RFC] Kimi K3 decode: FlashInfer reuse and integration scope

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#49536](https://github.com/sgl-project/sglang/issues/49536) |
| 作者 | YaoleWu |
| 创建日期 | 2026-11-10 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |

## 一句话结论
探讨 Kimi K3 decode 复用 FlashInfer MLA value-gate 原语时的最小有用集成范围:先落原生消费点,全量 executor 另行决策。

## 提案动机
Kimi K3 服务混合长 prefill、缓存恢复与反复 decode。对单节点 B300/SM103、TP8、无投机的 serving,目标是在保持 KV 容量、recurrent-state 所有权、HiCache 行为与公共 API 契约的前提下降低 decode 延迟。已有候选实现(#48510)带来 opt-in 的 93 层 prepared executor(DCP1/DCP8 profile);在请维护者长期承载这个模型特定实现之前,RFC 提问:既然 FlashInfer 已提供可复用原语,现在值得维护的最小有用集成是什么。作者建议先在现有 SGLang 消费者中验证原语复用,生命周期/正确性修复单独落地,全量 executor 在匹配的整体模型结果与所有权/CI 方案出现前保持候选状态。

## 提出的方案
- 起点:FlashInfer 已合并的 MLA value 投影 + gate 原语(yyihuang 工作,`flashinfer.cake_mla_value_gate`)—— BF16 latent values 与连续 latent 轴的 BF16 W_UV、strided BF16 gate;BF16 舍入的 value 投影乘以**未舍入 FP32 sigmoid**后输出 BF16;请求长度 ≤0 的行(含无效 gate 的填充行)输出零;W_UV 在图捕获外绑定,库内自动 8 行/16 行 program 选择。
- 三条集成路径对比:原生模型内增量复用 FI(保留现有调度、attention 与缓存/恢复路径,但需完整 K3 调用点、显式数值策略与图/权重生命周期验证)、维持 opt-in prepared executor(需维护模型特定代码、workspace/weight 绑定、分布式等待与 fallback 义务)、vendor 内核放外部插件(内核发布节奏与 SGLang 解耦,但需引擎接口、owner、pin 与匹配 GPU CI 的约定)。
- 首选原生切片:复用现有 `project_mla_values` hook 或新增窄范围 K3 调用点,补齐 FI 所需的 ready gate 与 valid-row/请求元数据;从公开的 BF16 decode 几何与 DCP1 起步,不支持场景保留原路径,LoRA/correction 路径暂留原实现,原生 post-merge DCP8 随后跟进。
- 精度策略需先达成一致:原生 BF16 gate 保留中间 BF16 sigmoid 舍入,而 FI 目前仅暴露 `sigmoid_precision="fp32"`,原生 opt-in 需要自己声明并验证数值契约。
- 落地序列:①消费者级原语复用(带 FI facade/preparation 生命周期、数值策略、fallback 与 eager/changed-input graph 检查)→ ②对现有消费者有影响的生命周期/正确性修复单独落地 → ③全量 executor 继续以 #48510 为候选,直到证据与所有权/CI 方案齐备。

## 优化点与收益
正文未给出用于当前验收的量化数字。历史 approximately 11% 的 integrated-build/profile 测量早于后续修复,仅作背景而非当前验收依据。已有证据:B300/SM103 上 FI `f454a17f` + SGLang 源码包检查,在 B1/q1、B25/q1、B64/q2 的 eager 与 changed-input graph replay 下 CuTe `MlaValue` 与 FI 位级(bitwise)一致。验收要求架构对比覆盖 native baseline、native + FI、prepared executor 三者(FI 开/关隔离),固定 checkpoint/tokenizer、输入、并发、图覆盖、内存预算等变量,报告 TTFT、decode latency/throughput、峰值内存、保留 KV 容量与数值漂移,覆盖单节点 SM103 TP8 无投机 workload。

## 现状与落地
仍为 open,向维护者征求四项决策:①首选原生消费点复用还是先走 prepared/plugin 路径,以及可接受的具体调用点改动;②FI 的 FP32 sigmoid 契约可否作为原生路径 opt-in,还是必须先做原生 BF16 舍入兼容;③全量 executor 应 in-tree 还是走 vendor 插件架构,由谁负责引擎接口、发布 pin 与 GPU CI;④全量 executor 需要哪些最少匹配的模型/恢复/容量证据才值得其维护成本。#48510 保持开放作为候选;11 月 10 日的 consumer follow-up 记录了实际原生/FI segment、BF16/FP32 差异、完整 segment 计时与官方 0.7.3 包内容差距。

## 关联
- #48510(候选 prepared executor PR)
- #48626(broader serving context)
- FlashInfer #8867(MLA value/gate 原语)
- #43949(monokernel RFC,插件架构讨论)
- #43675(原生 K3 reload 修复)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
