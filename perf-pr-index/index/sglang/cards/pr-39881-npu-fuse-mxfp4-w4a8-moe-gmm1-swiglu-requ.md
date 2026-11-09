# PR #39881: [NPU] Fuse MXFP4 W4A8 MoE gmm1 + swiglu + requant into one kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @iridiumine |
| 合并 | 2026-09-18 |
| 分类 | MoE(tier: major) |
| Label | npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/39881 |

## 一句话结论
NPU 上把 MXFP4 W4A8 MoE 的 gmm1+swiglu+requant 折进单个 aclnn kernel,每层 MoE 算子延迟降约 15%,TTFT 各分位一致改善。

## 动机
未融合的 MXFP4 W4A8 MoE 链路在每层 gate/up 投影要跑三个 kernel:gmm1、独立的 swiglu 激活 kernel、动态 MX requant。NPU A5 已提供 `npu_grouped_matmul_swiglu_quant_v2`,可把三者折叠为单个 aclnn kernel,但 MXFP4 路径尚未接入。

## 优化点
`NPUW4A8MXFP4MoEMethod` 新增 `apply_fused_gmm1_swiglu`,镜像 `NPUMXFP8MoEMethod` 的融合调用形态,但传入 FP4 weight dtype 与 e8m0 weight/x scale dtype;`pertoken_scale` 处理与该类既有 unfused `apply()` 一致(DeepEP bf16 路径上在此处量化,或已有 scale 时 reshape 成 pair 形式),返回 (e4m3 激活, e8m0 block scale) 供 w2 gmm 使用。Ascend MoE runner 把 `NPUMXFP8MoEMethod` 的 isinstance 判断换成 `_uses_fused_gmm1(kernel)` 探测,使 MXFP8 与 MXFP4 W4A8 都跳过独立激活步骤;`self.activation = None` 的 fail-loudly 语义保留。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py: NPUW4A8MXFP4MoEMethod 增加 use_fused_gmm1/GroupedMatmulSwigluQuant/hidden_states_quantizer 与 apply_fused_gmm1_swiglu
python/sglang/srt/layers/moe/moe_runner/ascend.py: _uses_fused_gmm1 探测替代 MXFP8 isinstance 判断,fused 路径跳过独立激活
```

## 性能声明
Serving benchmark:128 requests、每请求 16K input tokens、request rate 0.4、max concurrency 32、300s(NPU A5,tp-size 8、dp-size 2、DeepEP、DFLASH 投机;模型路径 PR 未写明具体名称)。Per-operator MoE 链(gmm1+swiglu+mxquant+gmm2)1.15ms → 0.98ms(-14.8%,约 15%)。TTFT:Mean 2901.01→2676.64ms(-7.7%)、Median 2556.05→2487.71ms(-2.7%)、P90 -6.2%、P95 -13.7%、P99 7413.64→4624.62ms(-37.6%);input token throughput 6810.63→6811.22 tok/s(+0.01%)。GSM8K 200 样本 accuracy 0.975、invalid 0.005。

## 验证与回归风险
PR checklist 显示未添加单元测试(该项未勾选);验证为端到端 serving benchmark + GSM8K 精度(运行环境含 DeepEP、DFLASH 投机、dp-attention 等大量 NPU 环境变量)。风险:融合路径依赖 A5 的 `npu_grouped_matmul_swiglu_quant_v2` 对 FP4 权重/e8m0 scale 的数值一致性(作者称与 unfused 链路除输出 fp8 requant 外一致);runner 端 `activation=None` 的 fail-loudly 约束需对所有 fused 方法成立。

## 关联
无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
