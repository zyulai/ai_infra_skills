# PR #30265: [AMD] Fix GLM-5.2 MTP Quark excludes

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wangjiaxin99(co-author @Raiden-Makoto) |
| 合并 | 2026-07-08 |
| 分类 | Speculative Decoding(major) |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/30265 |

## 一句话结论

修复 GLM-5.2 MXFP4 在 AMD 上 Quark 量化的 MTP/NextN 权重加载,使 MTP 解码吞吐提升约 3.68 倍。

## 动机

GLM-5.2 使用 `GlmMoeDsaForCausalLM`,但其 draft/MTP 路径复用 DeepSeek NextN 实现。Quark 把 MTP 排除权重记录在 checkpoint 前缀 `model.layers.78.*` 下,而 SGLang 用 `model.*`、`model.decoder.*` 与 fused MoE 前缀 `model.decoder.mlp.experts` 构建 draft 运行时模块。前缀与粒度不匹配导致 Quark `exclude_layers` 无法命中部分 bf16 MTP 权重,一些 MTP 模块被错误建成 MXFP4 参数,加载 bf16 权重时 shape mismatch。该问题与 Wafer 的 GLM-5.2 AMD 博客描述的 MTP prefix mismatch 一致。

## 优化点

这不是内核优化,而是打通 AMD 上 GLM-5.2 MTP(EAGLE)推理路径的量化加载修复:新增 GLM 专属 NextN 类 `GlmMoeDsaForCausalLMNextN`,替代 `DeepseekV3ForCausalLMNextN`;把 DeepSeek NextN 的 quant-config 处理提取为 `_resolve_nextn_quant_config()` 供 GLM 覆写;运行时扩展 Quark `exclude_layers`,把 `model.layers.78.eh_proj/enorm/hnorm/shared_head.norm` 映射到 `model.*`,decoder block 权重映射到 `model.decoder.*`,并对 routed expert 的 leaf 排除项补加 coarse fused-MoE 前缀 `model.decoder.mlp.experts`(fused MoE 模块按 coarse 前缀查询,仅映射 leaf 名不够)。同时把 Quark `layer_quant_config` 的 key 做同样 remap,避免静默回退到错误 scheme。混合量化行为保留:被排除的 MTP 模块仍为 bf16,其余 draft 模块沿用各自 Quark 量化配置。

## 关键改动

```text
python/sglang/srt/models/glm4_moe.py: 新增 GlmMoeDsaForCausalLMNextN 与 _resolve_nextn_quant_config(exclude_layers/layer_quant_config 运行时映射)
python/sglang/srt/models/deepseek_nextn.py: 提取 _resolve_nextn_quant_config() 供子类覆写
python/sglang/srt/configs/model_config.py: draft model 架构路由 GlmMoeDsaForCausalLM -> GlmMoeDsaForCausalLMNextN
```

## 性能声明

GLM-5.2 MXFP4,4 张 AMD GPU,TP4,bench_serving random 数据集(input 10000 / output 1500,8 prompts,max-concurrency 1):无 MTP 输出吞吐 49.15 tok/s、TPOT 19.58 ms、E2E 30515.29 ms;开 MTP(EAGLE,steps 5,topk 1,draft tokens 6)后 180.70 tok/s、TPOT 4.75 ms、E2E 8297.65 ms,accept length 5.94。即输出吞吐约 3.68 倍、TPOT 约 4.12 倍降低、E2E 约 3.68 倍降低;TTFT 基本不变(MTP 主要加速 decode)。

## 验证与回归风险

验证为上述 4-GPU AMD 端到端 serving + bench_serving 命令(PR 给出完整 server/bench 命令),PR 未提供新的单元测试文件。风险面:改动触及 DeepSeek NextN 共享路径(`deepseek_nextn.py` 重构与 `model_config.py` 架构路由),DeepSeek 系列的 Quark 量化 NextN 加载依赖 `_resolve_nextn_quant_config()` 保持等价;GLM 专属 mapper 需与 `DeepseekV2WeightLoaderMixin` 的 NextN 权重命名规则同步维护。

## 关联

#29781(DeepSeek NextN quant-config 处理的先前方向,本 PR 建立其上并扩展);#29373(ROCm draft-depth `cuda_runtime.h` guard,已解决)。同分类其他 PR 本卡素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
