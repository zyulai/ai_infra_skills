# PR #51808: [Online quantization] Support activation quant key override for online MXFP4 linear/MOE quant methods

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-11-05 |
| 分类 | MoE(tier: major) |
| Label | documentation, ready, nvidia, quantization |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 9 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/51808 |

## 一句话结论

在线 MXFP4 quantization 支持激活 quant key 覆写(如 `activation: null` 派发 AITER_MXFP4_BF16),此前这类覆写直接抛 NotImplementedError。

## 动机

覆写激活 quant key 是选中特定 MoE 后端(如 `AITER_MXFP4_BF16`、`AITER_MXFP4_MXFP4`)的唯一途径,而这些后端此前无法以任何方式选择;在线量化一遇到激活覆写就抛 NotImplementedError(issue #52167)。

## 优化点

仅 `Mxfp4OnlineLinearMethod`/`Mxfp4OnlineMoEMethod` 接受激活覆写(需 `accepts_activation_override = True`,经 `OnlineRequantizationMixin.resolve_activation_quant_key` 解析),构造参数 `activation_quant_key`(默认 `kMxfp4Dynamic`);每个激活 key 要么派发到能计算它的内核、要么报错,绝不静默替换格式(如 ROCm EMULATION 后端只算 MXFP4 激活,其他 key 显式 NotImplementedError);静态激活 key 一律拒绝(在线量化无校准 scale);显式 `activation: null` 语义化——请求 BF16 未量化激活,与"未提供"区分(`QuantSpec` 以 `_UNSET` + `fields_set` 跟踪显式字段);歧义简写名(mxfp4/mxfp8)按字段解析:weight 取 static、activation 取 dynamic;预量化 checkpoint 上 activation-only 覆写只作用于匹配层的 checkpoint quant method,不再顺带在线量化 checkpoint 跳过的层。

## 关键改动

```text
vllm/config/quantization.py: QuantSpec _UNSET/fields_set、_WEIGHT_QUANT_KEY_NAMES、quant_key_name
vllm/model_executor/layers/quantization/ 与 fused_moe/oracle(mxfp4 等,9 文件未含于 diff): Mxfp4Online* 方法 activation_quant_key 与 oracle 端 activation key 校验
tests/quantization/test_online.py: 覆写 honored-or-rejected、null 语义、checkpoint 方法应用矩阵
docs/features/quantization/online.md: activation overrides 与 targets spec 形式文档
```

## 性能声明

PR 未注明吞吐/精度数字。功能验证(MI350X):相关测试 127 passed / 4 skipped(`VLLM_ROCM_USE_AITER=0/1` 均然);完整套件 1 failed / 653 passed / 676 skipped,唯一失败 `test_rocm_mxfp4_moe_oracle[...AITER_MXFP4_BF16]`(mismatch 0.3704 vs 阈值 0.3)仅在整跑出现、隔离运行于分支与 main 均通过。示例:Qwen/Qwen3.5-35B-A3B `--quantization online` + targets `{"*experts*":{"weight":"mxfp4","activation":null}}` 在单卡 MI350X 派发 `AITER_MXFP4_BF16` 并成功服务 40 层 experts。

## 验证与回归风险

测试覆盖 online 方法解析、MXFP4 MoE oracle、fused shared expert/FSE 一致性、requantization 路径(PRC 列出 14 个相关测试文件)。已知未完成项(正文明确列出):其他 online weight key 尚不支持激活覆写;OCP MX emulation 不计算其他激活格式;Quark/Humming 的 per-layer targets 覆写不到达 kernel 选择处(需 oracle 层处理)。风险:`--moe-backend aiter_mxfp4_fp8` 与在线 MXFP4 组合仍在权重处理期失败(无静态 scale,行为不变);fused shard targets 现在比较 resolved spec,显式 null 与省略 activation 被视为不同配置。

## 关联

同分类 MoE:#53911(DeepEP-v2 融合内核);上游 issue #52167。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
