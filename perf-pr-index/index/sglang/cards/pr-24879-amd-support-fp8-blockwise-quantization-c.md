# PR #24879: [AMD] support fp8 blockwise quantization combine for mori ep

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @billishyahao |
| 合并 | 2026-05-13 |
| 分类 | Quantization (tier: major) |
| Label | amd |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/24879 |

## 一句话结论

给 Mori EP dispatcher 的 combine 阶段增加 FP8 blockwise 量化,补齐 FP8/FP4 dispatch 之外的最后一段量化路径。

## 动机

AMD Mori EP 此前只支持 FP8/FP4 的 dispatch 量化,combine 阶段仅有 bf16 或 `fp8_direct_cast` 可选,缺少 blockwise FP8 模式。本 PR 修复 issue #24866,并配合 ROCm/mori#311 的内核侧改动,让 combine 也能走 blockwise 量化以减少通信数据量、改善 MXFP4 模型的精度表现。

## 优化点

引入 `DispatchDtype`(bf16/fp8/fp4)与 `CombineDtype`(bf16/fp8/fp8_direct_cast)两个枚举,替换原有的 `fp8_dispatch`/`fp4_dispatch` 布尔 flag,并贯通 `init_mori_op()` 与 `_MoriEPDispatcherImplBase`。combine 新增 `fp8`(即 fp8_blockwise)模式,与 `fp8_direct_cast` 并列。新增环境变量 `SGLANG_MORI_COMBINE_DTYPE`(auto|bf16|fp8|fp8_direct_cast)做显式控制,统一 dispatch 变量为 `SGLANG_MORI_DISPATCH_DTYPE`(auto|bf16|fp8|fp4),旧变量 `SGLANG_MORI_FP8_DISP`/`SGLANG_MORI_FP4_DISP`/`SGLANG_MORI_FP8_COMB` 保留但带弃用告警。默认值按模型权重类型自动判定:FP4 模型默认 fp8_blockwise combine,FP8 模型默认 bf16 combine。fp8 blockwise combine 的内部 scale_dim 可被 `MORI_FP8_COMBINE_SCALE_DIM` 覆盖(默认 56)。

## 关键改动

```text
python/sglang/srt/layers/moe/token_dispatcher/moriep.py: DispatchDtype/CombineDtype 枚举、combine dtype 解析、set_quant_config 自动判定
docker/rocm.Dockerfile: MORI_COMMIT 从 v1.1.1 更新到 96ffa169710f...(配合 mori#311)
```

```python
class CombineDtype(Enum):
    bf16 = "bfloat16"
    fp8 = "float8_blockwise"
    fp8_direct_cast = "float8_direct_cast"
```

## 性能声明

精度测试(amd/DeepSeek-R1-0528-MXFP4,DEP8,GSM8K,16 组 `SGLANG_MORI_DISPATCH_DTYPE` x `SGLANG_MORI_COMBINE_DTYPE` 组合全部实测):精度落在 91.2%-94.5%,无回归。`fp8_blockwise` combine 比 `fp8_direct_cast` 精度约高 2%(如 fp4 dispatch:94.5% vs 91.2%)。吞吐(tps)随组合变化:fp4+bf16 848 tps 最高,默认 auto/auto(fp4 dispatch + fp8 blockwise combine)为 93.8% acc、771 tps,fp8 dispatch 组合约 518-536 tps。PR 未单独给出与改前基线的端到端加速比。

## 验证与回归风险

验证手段为上述 16 组合的 GSM8K 精度矩阵;Checklist 中 unit tests、文档、benchmark 三项未勾选。风险面:MXFP4 模型默认 combine dtype 由 bf16 变为 fp8_blockwise(改变数值路径);依赖 rocm.Dockerfile 拉取特定 mori commit,仅影响 AMD/Mori EP 路径;弃用环境变量仍可用但会打告警,未来移除。

## 关联

修复 issue #24866;内核依赖 ROCm/mori#311。正文未提及其他 sglang 关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
