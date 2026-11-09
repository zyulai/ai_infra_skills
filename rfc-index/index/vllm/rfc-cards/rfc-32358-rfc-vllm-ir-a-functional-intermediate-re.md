# RFC #32358: [RFC]: vLLM IR: A Functional Intermediate Representation for vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32358](https://github.com/vllm-project/vllm/issues/32358) |
| 作者 | ProExpertProg |
| 创建日期 | 2026-01-14 |
| 状态 | ⚪ closed(关闭于 2026-08-26) |
| labels | RFC, stale, torch.compile, vllm-ir |
| 热度 | 👍 12 · 💬 7 |
| 分类 | Other |

## 一句话结论
提议 vLLM IR:torch 之上的函数式中间表示,把自定义算子的语义与实现/分发分离,改善编译融合与内核调度。

## 提案动机
torch.compile 的融合/变换 pass 难以处理 `RMSNorm`、`Quant` 等自定义算子:它们要么分解成脆弱的 torch 算子序列,要么对应各种自定义内核、需逐一单独处理;这些算子都是 `CustomOp` 实例,而 `CustomOp` 自身的分发逻辑复杂笨拙、可见性低。vLLM IR 是填补低层 torch 算子与 vLLM 层(如 RMSNorm)之间空白的函数式 IR,把语义(***semantics***)与实现/分发(***implementation*** / ***dispatching***)分离;它作为 torch FX 表示中的 dialect 运作,可与常规 torch 算子及自定义算子完全互操作,并支持从现有 `CustomOp` 方式分片迁移。

## 提出的方案
- 每个 op 是纯函数,语义独立于实现定义——语义以一段 torch 算子序列描述,兼作默认 torch 原生实现;调用处直接调 op,由 op 负责分发到合适实现
- 内核分发由 `VllmConfig` 中 per-op 优先级列表控制,平台提供默认值、用户可覆盖(示例:`--kernel-config.ir_op_priority.rms_norm=vllm_c`);编译期 op 以 custom op 形式出现在 FX 图中,可先做与实现无关的融合变换,流水线后段再 lower 到所选实现
- 相对 `CustomOp` 的优势:树内外皆可简单扩展地注册 op 与实现;高层函数式编译 IR 便于编译 pass 与 OOT 编译后端;分发的单一事实来源与更易配置的内核选择;经 torch.compile 对不同实现按需 autotune
- 非侵入式落地:不需要改任何模型定义,并为 OOT `CustomOp` 注册提供软迁移路径
- 正文还逐条评估备选方案并说明其不足:包装 `CustomOp` 为 functional custom op(无法自动 lowering)、内核抽象(不解决编译与 autotune 问题)、全量重写为 Helion(工程量巨大)、维持现状

## 优化点与收益
正文未给出量化数字。

## 现状与落地
已关闭(2026-08-26,label 含 RFC、stale、torch.compile、vllm-ir)。反馈期为 2026-01-14 至 01-25 共 11 天;正文未记录最终落地情况,需查关联 PR。

## 关联
- 详细提案(Google Docs):https://docs.google.com/document/d/1zyPdGVX7TnVtvqD9NsaTO4E02u7PY7uudreLrdWp2GM/edit?tab=t.xerdhb9i406r
- 幻灯片(Google Slides):https://docs.google.com/presentation/u/0/d/1k0Zo33KubK7pmhYXg-7G1PjgcEbOhBkSbwiEwUpUyH0/edit

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
