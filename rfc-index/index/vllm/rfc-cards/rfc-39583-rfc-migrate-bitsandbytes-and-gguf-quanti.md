# RFC #39583: [RFC]: Migrate bitsandbytes and GGUF quantization support to OOT plugin

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39583](https://github.com/vllm-project/vllm/issues/39583) |
| 作者 | mgoin |
| 创建日期 | 2026-04-11 |
| 状态 | 🟢 open |
| labels | RFC, quantization |
| 热度 | 👍 15 · 💬 17 |
| 分类 | Quantization |

## 一句话结论
提议弃用并最终移除使用率极低的 bitsandbytes/GGUF 量化后端(或外移为 OOT 插件),让核心权重加载路径恢复可读、可重构。

## 提案动机
bitsandbytes 与 GGUF 是 vLLM 中相对维护成本使用率极低的两个量化/格式后端(作者估计分别约 0.5% 与 0.1%)。两者都早于现行 `weight_loader_v2` 架构、未迁移,并在共享代码(`linear.py`、`fused_moe/layer.py`、`vocab_parallel_embedding.py`)的关键权重加载路径中注入条件分支,使代码库更难维护与重构。此外两者性能不佳——用户常反映 GGUF 模型用 llamacpp 更快(消费级 GPU 上对 bs=1 的优先级不同)。作者提议弃用并最终移除两者;若二选一,优先移除 GGUF(BNB 使用量更大);移到 OOT 量化插件的方案作者持怀疑态度,因为它们当前需要修改 vLLM 内部结构,可行性存疑。

## 提出的方案
- 现状盘点(正文汇总表):bnb 专用 Python 约 1,426 行、CUDA kernel 0 行、共享代码分支约 95 行/6 处、TP 支持受限(pre-quant 不可用)、8-bit 强制 eager;GGUF 专用 Python 约 1,464 行、CUDA kernel 约 6,000 行、分支约 75 行/5 处、TP/CUDA graph 全支持、模型特定 hack 8+ 个模型;两者合计专用 Python 约 3,100 行、共享分支约 170 行,且都不在 `weight_loader_v2` 白名单上,阻塞 v2 迁移;
- `linear.py` 清理:移除 4 个并行线性类中约 170 行条件分支,weight_loader 变为"定维、narrow、拷贝"的直白逻辑(作者称这是最大收益);
- `weight_loader_v2` 迁移:bnb/GGUF 移除后,旧版 `weight_loader` 路径有望整体删除或大幅简化;
- `fused_moe/layer.py`:移除约 45 行 early-return 分支;模型加载器工厂:移除约 6 个 loader 类中的 2 个;
- 配置/参数:移除 auto-detection 分支、CUDA graph workaround 与 bnb/GGUF 特有校验;
- 构建与依赖:从 `csrc/quantization/gguf/` 移除约 6,000 行 CUDA kernel 及对应 CMakeLists 条目(加快构建),移除 `bitsandbytes`、`gguf` pip 依赖。

## 优化点与收益
主要收益非行数本身,而是让 `linear.py` 的权重加载方法恢复可读、可重构,并解除 `weight_loader_v2` 迁移的阻塞;附带收益:减少约 170 行共享路径分支与约 6,000 行 CUDA kernel、加快构建、少两个 pip 依赖。量化数字均照抄自正文:使用率约 0.5%(bnb)/0.1%(GGUF,作者估计),专用代码约 2,890 行(5 个自包含文件)、GGUF kernel 约 6,000 行;性能方面正文未给出量化数字(仅定性引用用户反馈)。

## 现状与落地
state=open:反馈期为两周,已 CC 多位维护者;正文同时列出"直接移除"与"OOT 插件化"两条路径,倾向前者,尚无定论。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
