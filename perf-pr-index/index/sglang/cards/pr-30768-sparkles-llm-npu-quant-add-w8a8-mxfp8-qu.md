# PR #30768: :sparkles: [llm][npu][quant] Add W8A8 MXFP8 quantization for Qwen3 MoE on Ascend NPU

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @TallMessiWu |
| 合并 | 2026-07-29 |
| 分类 | MoE(major) |
| Label | documentation, quant, npu, run-ci |
| Diff | 素材未注明 diffstat(diff 片段可见 14 个文件,含 2 个新增,另有 3 个文件因篇幅略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/30768 |

## 一句话结论

在 Ascend NPU(A5)上为 Qwen3/Qwen3.5 MoE 增加 W8A8 MXFP8 量化,Qwen3-30B-A3B 端到端吞吐约 +44%、TPOT 约 -30%,精度与 BF16 持平。

## 动机

延续 #21584 的 NPU 量化路线(此前已落地 dense W4A4 #23795、dense W8A8 MXFP8 #22352),把 MXFP8 覆盖到 LLM 的 MoE 部分:FusedMoE 专家层用 8-bit 权重 + 8-bit 激活(block_size=32)降低带宽与计算开销。所用融合 MoE MX 内核(`npu_grouped_matmul_swiglu_quant_v2`、`npu_dynamic_mx_quant`)仅在 Ascend A5(Ascend 950)可用。

## 优化点

online(`--quantization mxfp8`)与 offline(msmodelslim 预量化权重,经 checkpoint 的 `quant_model_description.json` 无 flag 选择)两条路径共享同一个 per-gmm 内核 `NPUMXFP8MoEMethod`。online 侧 `Fp8Config.get_quant_method()` 把 FusedMoE 分派给新 `NPUMXFP8OnlineMoEMethod`(继承 `UnquantizedFusedMoEMethod`,仅替换 `create_moe_runner`);BF16 专家权重在加载时经 `npu_dynamic_mx_quant` 一次性量化为 e4m3 + e8m0 块 scale。forward 链路:`ascend_tp` dispatcher 用 `npu_moe_init_routing_v2(quant_mode=3)` 把 per-token MXFP8 激活量化融合进 routing → gmm1 `npu_grouped_matmul_swiglu_quant_v2`(gate_up + swiglu + requant 融合,输出已为 gmm2 量化好,免去独立激活步骤)→ gmm2 `npu_grouped_matmul`。DeepEP 无 MXFP8 dispatch dtype,保持发 BF16、由 gmm1 自行量化,两条 dispatcher 汇合到相同内核输入。权重与 scale 保存为非连续转置视图,先 `npu_format_cast` 成 FRACTAL_NZ 再转置(gmm1 的 CheckMXTranspose 要求两侧转置标志一致);PR 实测改成 contiguous 反而慢 6.2%(decode,128 experts)。另修复 offline 路由 gate:量化过的 `mlp.gate` 此前被无 scale 地 cast 成 BF16 导致输出全错,现按 description 驱动正确反量化。

## 关键改动

```text
srt/hardware_backend/npu/quantization/moe_methods.py: 新增 NPUMXFP8MoEMethod(权重在线量化/FRACTAL_NZ 布局、gmm1 融合 swiglu-requant、gmm2)
srt/hardware_backend/npu/quantization/online_moe_methods.py: 新增;NPUMXFP8OnlineMoEMethod(--quantization mxfp8 入口)
srt/layers/quantization/modelslim/schemes/modelslim_mxfp8_moe.py: 新增;offline ModelSlimMXFP8MoEScheme
srt/hardware_backend/npu/moe/matmul.py: 新增 GroupedMatmulSwigluQuant(npu_grouped_matmul_swiglu_quant_v2 封装)
srt/hardware_backend/npu/moe/init_routing.py: MXFP8_QUANT_MODE=3 与 flat→pair-split scale 重排
```

```python
# gmm1:gate/up + swiglu + requant 单内核
group_list = expert_tokens.cumsum(0) if group_list_type == 1 else expert_tokens
return torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
    x=hidden_states,
    weight=[weight] if transposed else [weight.transpose(1, 2)],
    group_list=group_list, **scale_args,
)
```

## 性能声明

Ascend A5,Qwen3-30B-A3B,与 BF16 基线对比(online 与 offline 两模式,两者相差在噪声内):

- 吞吐:request throughput 3.03 → 4.37 req/s(+44.2%);total token throughput 1832.69 → 2643.95 tok/s(+44.3%);input/output token throughput 均 +44.3%。
- 延迟:mean E2E 10101.74 → 7004.54 ms(-30.7%);mean TPOT 47.44 → 32.79 ms(-30.9%);mean TTFT 168.29 → 134.96 ms(-19.8%);P99 TPOT -38.2%(BF16 的 P99 TTFT 1204.71 ms 被单次 3100.86 ms 离群 run 拉高,另两轮为 263.73/249.55 ms)。
- FRACTAL_NZ vs ND(同基准):total token throughput +2.0%(online)/+3.0%(offline),mean TTFT -6.5%/-6.4%;独立内核探针(128 experts)decode +1.4%、prefill +3.8%,噪声底 0.2–0.3%。
- 精度(GSM8K,3 轮):BF16 平均 95.91,online 95.93(+0.03),offline 95.78(-0.13),均小于 BF16 自身 0.45pp 的轮间波动。

## 验证与回归风险

验证以 A5 端到端性能与 GSM8K 为主;PR 未附新增自动化单测(diff 中无测试文件),checklist 中单测项未勾选;正文 Status 标注 WIP/Draft(但 pack 元信息显示已于 2026-07-29 合并)。风险面:改动触及 NPU 与通用层的交界——`fp8.py` 的 FusedMoE 分派仅限 `is_npu() and use_mxfp8`,`unquant.py` 去掉硬编码 `dispatcher_output_dtype` 会影响 NPU 非量化 MoE 路径,`moe_runner/ascend.py` 的 run() 为 MXFP8 增加 fused 分支;`overrides.py` 在 NPU 上跳过 mxfp8→flashinfer_trtllm 的 runner 强制(否则会触发 w1/w3 shard 交换、gate/up 装反的静默退化,PR 记录了该回归);FRACTAL_NZ 对齐规则在 `utils.py` 为 fp8 扩展。仅 ascend_tp 路径做过端到端验证,DeepEP 路径未验证。

## 关联

前序:#21584(Ascend MXFP8/MXFP4 总跟踪)、#22352(dense W8A8 MXFP8,复用其 `NPUMXFP8LinearMethod`/`Fp8Config` NPU 分派/`ModelSlimMXFP8Scheme`)、#23795(dense W4A4)。同系列 int8/W4A8/W4A4 MoE scheme 与本 PR 共享 ModelSlim 模式。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
