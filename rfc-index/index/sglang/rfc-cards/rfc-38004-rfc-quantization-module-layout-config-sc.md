# RFC #38004: [RFC] Quantization module layout: Config, Scheme, backend kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38004](https://github.com/sgl-project/sglang/issues/38004) |
| 作者 | mmangkad |
| 创建日期 | 2026-09-04 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |

## 一句话结论
把 SGLang 量化模块统一迁移到贡献指南规定的 Config / Scheme / 后端 kernel 包布局,分阶段小 PR 完成。

## 提案动机
量化方法目前两种形态并存:awq、compressed_tensors、gptq、modelslim、quark 五个已按包布局;其余为单文件巨石,把 config 解析、权重创建、post-load 处理与 kernel 分发挤在一起——fp8.py 2765 行、modelopt_quant.py 3050 行、mxfp4.py 2060 行;17 个顶层文件定义 QuantizationConfig 子类,合计 101 文件 39,262 行。实际痛点:行为难定位;同一 helper 被反复实现且悄悄分叉(「is this layer excluded from quantization」有 11 个变体,一次修复没传到另一份);加方法要改 eager import 块、base dict、至多四个平台 override 块加 CLI 列表且会漂移(#37552 之前 --quantization marlin 能过 argparse 却在启动时失败);WEIGHT_LOADER_V2_SUPPORTED 20 项中 9 项指向不存在的类;导入包即拉起全部 24 个 config 模块的第三方依赖;multimodal_gen 另有一套并行量化栈(28 文件 6,620 行,30 处深 import srt 内部)。

## 提出的方案
- 目标布局:layers/quantization/<method>/<method>.py(Config:解析、校验、选 scheme)+ schemes/<method>_linear.py、<method>_moe.py(Scheme:权重创建、加载、后处理)+ hardware_backend/<plat>/quantization/<method>_kernels.py;经 get_linear_scheme / get_moe_scheme 显式选择。非新架构——贡献指南已如此规定,#21126(AWQ)、#26402(GPTQ)是现成范例。
- 声明式注册表:字符串 spec 表(如 "fp8": "…:Fp8Config"),惰性解析 config 类,平台覆盖也表驱动(#37602)。
- 兼容模型:legacy import 走显式 re-export shim(保 isinstance 身份、可 grep),shim 只在最终门控阶段、发布过带弃用警告的 release 后移除;base_config 等符号按 API 长期保留。
- 交付规则:小 PR,标题 [N/N] Quantization Refactor;机械移动与行为变更永不同 PR;每个 PR 附等价性证据(registry 表前后对比、机械拆分验证流程)。
- 阶段:Phase 0 清理+注册表+shim 测试(#37552 merged;#37602、#38005 open)→ Phase 1 共享基础(统一 scheme ABC 到 base_scheme.py、新增 fp8_postload.py、拆 utils.py、合并 11 个 is_layer_skipped、替换 WEIGHT_LOADER_V2_SUPPORTED、瘦身 base_config)→ Phase 2 四大巨石拆分(fp8、modelopt、mxfp4、fp8_utils)→ Phase 3 两套 KV-cache 量化系统共置(不语义合并)→ Phase 4 multimodal_gen 改薄适配器 → Phase 5 门控移除 shim。
- 明确不做:不改 FusedMoEMethodBase.apply 签名或 MoeRunner 接线;不整体搬 kernel wrapper 进 hardware_backend/;不语义合并两套 KV-cache 量化系统。

## 优化点与收益
正文未给出量化数字(属可维护性重构)。引用的规模/活跃度数字:multimodal_gen 28 文件 6,620 行;该目录近六个月各月 46、48、49、56、70、84 commits,近 90 天 217 commits——因此不能依赖长命分支,必须小步走。

## 现状与落地
仍在推进(open)。状态表:Phase 0 的 #37552 已 merged,#37602、#38005 open,Phase 1-5 标 planned。作者就开放问题征求意见:FP8 post-load 迁移一个 PR 还是拆线性/MoE 两半;mxfp4 的 delegate 类与 Mxfp4MoEMethod flag 巨石谁胜出;与在途 MoeRunner 迁移的边界;并提议一个评审规则——方法多的新 PR 应拆出 schemes/、注册走一行 registry 条目。

## 关联
- #21126(AWQ 范例): https://github.com/sgl-project/sglang/issues/21126
- #26402(GPTQ 范例): https://github.com/sgl-project/sglang/issues/26402
- #15194(相关 roadmap): https://github.com/sgl-project/sglang/issues/15194
- #37552(Phase 0,merged): https://github.com/sgl-project/sglang/issues/37552
- #37602(注册表 PR): https://github.com/sgl-project/sglang/issues/37602

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
