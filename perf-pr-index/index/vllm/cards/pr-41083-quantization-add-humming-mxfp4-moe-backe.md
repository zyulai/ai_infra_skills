# PR #41083: [Quantization] add humming mxfp4 moe backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2026-05-03 |
| 分类 | MoE (tier: major) |
| Label | performance, ready, deepseek, nvidia |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/41083 |

## 一句话结论

为 MXFP4 MoE 增加 Humming 后端选项,DeepSeek-V4-Flash 上 prefill TPS 从 10532 提升到最高 19404。

## 动机

vLLM 此前在 #34556 中初步集成了 Humming(https://github.com/inclusionAI/humming/),但 MXFP4 MoE 的 kernel oracle 尚未接入 Humming 后端,DeepSeek-V4 等 MXFP4 MoE 模型只能走 Marlin W4A16 等 weight-only 路径。作者正在把 Humming 后端集成到 dense/moe kernel oracle,以提供 W4A16 与 W4A8(激活 FP8)两种混合精度执行选项。Humming 目前不是 vLLM 强依赖,需 `--moe-backend humming` 显式开启。

## 优化点

机制层面:在 `Mxfp4MoeBackend` 枚举新增 `HUMMING`,并接入 backend→kernel 类映射(`HummingIndexedExperts`/`HummingGroupedExperts`/`BatchedHummingGroupedExperts`);`get_humming_moe_gemm_type` 改为支持 `grouped_contiguous`,去掉了 Hopper TMA 自动选 grouped 的逻辑。权重侧新增 `humming_utils.py`:`prepare_humming_moe_layer` 把 MXFP4/gpt_oss_mxfp4 权重转换成 Humming 标准格式并按 pad_n=256/pad_k=128 做布局变换,`get_humming_moe_quant_config` 从 layer 上的 schema 构建 `FusedMoEQuantConfig`;构造接口从传入 `HummingMoEMethod` 改为显式 `moe_config + quant_config`。`HummingConfig.override_quantization_method` 对 gpt-oss 模型禁止 `--quantization humming`,要求改用 `--moe-backend humming`。W4A8 通过 `VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}'` 开启激活量化。性能收益主要来自 moe kernel 与 moe sum kernel 的替换。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py: HUMMING 后端枚举、kernel 类映射与权重转换分支
vllm/model_executor/layers/fused_moe/fused_humming_moe.py: experts 类重构为显式 moe/quant config 构造
vllm/model_executor/layers/quantization/utils/humming_utils.py: 新文件,MoE 层权重准备与 quant config
vllm/config/kernel.py: moe backend 列表新增 "humming"
vllm/model_executor/layers/quantization/humming.py: schema 状态移到 layer,select_gemm_impl 传显式参数
```

## 性能声明

DeepSeek-V4-Flash + H20 x 4(DP4、EP、kv-cache fp8、block-size 256、FULL_AND_PIECEWISE),`vllm bench serve` TPS:Marlin W4A16 Prefill 10532.12 / Decoding 1650.82;Humming W4A16 Prefill 15052.24 / Decoding 1727.73;Humming W4A8 Prefill 19404.33 / Decoding 1730.07。GSM8K(2-shot, flexible-extract exact_match):Marlin 0.9272,Humming W4A16 0.9227,Humming W4A8 0.9219。strict-match 分别为 0.1486/0.1471/0.1433。

## 验证与回归风险

PR 正文给出 lm_eval GSM8K 对比命令与结果,未见新增自动化测试(素材 diff 中无 test 文件变更)。风险面:改动触及通用 `fused_moe/layer.py` weight_loader(移除 humming 特判)、`mxfp4.py` 的 Marlin/其他后端共用函数签名(新增 `layer` 参数)与 `VLLM_HUMMING_MOE_GEMM_TYPE` 语义(不再 bool 化、新增 grouped_contiguous 取值);删除 `humming_moe_utils.py`。非 Humming 后端行为应不变,但 MXFP4 oracle 路径被触及。

## 关联

同区域:#34556(Humming 初始集成,含 moe sum kernel 引入);MXFP4 MoE 其他后端:Marlin、FlashInfer CUTLASS、triton(同在 oracle/mxfp4.py)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
