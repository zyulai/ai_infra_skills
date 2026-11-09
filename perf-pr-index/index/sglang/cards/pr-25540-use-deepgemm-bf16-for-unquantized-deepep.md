# PR #25540: Use DeepGEMM BF16 for unquantized DeepEP LL MoE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-05-18 |
| 分类 | MoE(tier: major) |
| Label | quant, run-ci |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 ep_moe/layer.py 与 unquant.py |
| 链接 | https://github.com/sgl-project/sglang/pull/25540 |

## 一句话结论

把未量化的 DeepEP 低延迟 MoE 走 DeepGEMM BF16 masked runner,替代本地 PyTorch bmm 兜底,c=512 吞吐 +8.8%。

## 动机

前序 #24906 给所有 `quant_config=None` 的 MoE 层开了 BF16 DeepEP dispatch,范围过宽。目标场景是 Qwen3.5 NVFP4 的 MTP/NextN draft MoE:主 decode MoE 可用 `flashinfer_cutedsl + DeepEP`,而 draft MoE 未量化(`quant_config=None`),此前依赖一个本地 `forward_unquantized_deepep_ll` PyTorch bmm 兜底,效率不佳。本 PR 将行为收窄到"非 NPU/HIP 上未量化的 DeepEP low-latency MoE 且 MoE runner 为 deep_gemm"这一情形,并改用 #22822 引入的 `dispatcher_output_dtype` 机制。

## 优化点

在 `EPMoE.__init__` 中为上述窄场景设置 `deprecate_flag=True`,使执行委托给通用 `FusedMoE` 路径:dispatch、`quant_method.apply`、DeepGEMM runner、combine,并断言 `ENABLE_JIT_DEEPGEMM`。dispatch 侧通过 `layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})` 让 DeepEP LL dispatch 保留 BF16 激活(该配置移到 `UnquantizedFusedMoEMethod.process_weights_after_loading`,替代原先 `__init__` 里的临时做法);`run_moe_core` 的 DeepEP LL 分支删除 `quant_config is None` 时的本地 bmm/silu 兜底 `forward_unquantized_deepep_ll`,统一走 DeepGEMM 的 `GROUPED_GEMM_NT_BF16_MASKED` masked runner。#25525 的 `flashinfer_cutedsl + DeepEP` MoeRunner 路径保持不变。

## 关键改动

```text
python/sglang/srt/layers/moe/ep_moe/layer.py: 新增未量化 DeepEP LL 的 deprecate_flag 分支;删除 forward_unquantized_deepep_ll 本地 PyTorch 兜底
python/sglang/srt/layers/quantization/unquant.py: process_weights_after_loading 中按条件对 dispatcher 设置 dispatcher_output_dtype=bf16
```

```python
elif (
    quant_config is None
    and self.w13_weight.dtype == torch.bfloat16
    and get_moe_runner_backend().is_deep_gemm()
    and get_moe_a2a_backend().is_deepep()
    and get_deepep_mode().enable_low_latency()
    and not _is_npu
    and not _is_hip
):
    assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM, \
        "Unquantized DeepEP low-latency MoE requires DeepGEMM BF16"
    self.deprecate_flag = True
```

## 性能声明

`nvidia/Qwen3.5-397B-A17B-NVFP4`,随机 1k ISL / 1k OSL,并发 4..512,共 12 GPU(decode 8 GPU),DeepGEMM vs 移除本地兜底前的 `HEAD^` baseline,输出吞吐 tok/s:

| 并发 | baseline | DeepGEMM | 变化 |
| ---: | ---: | ---: | ---: |
| 4 | 74.711 | 71.369 | -4.5% |
| 8 | 151.433 | 155.797 | +2.9% |
| 16 | 292.595 | 330.539 | +13.0% |
| 32 | 541.348 | 579.363 | +7.0% |
| 64 | 1144.301 | 1137.058 | -0.6% |
| 128 | 2207.457 | 2267.953 | +2.7% |
| 256 | 4275.715 | 4539.305 | +6.2% |
| 512 | 8239.496 | 8965.792 | +8.8% |

中位 TPOT c=512 为 45.313 ms(baseline 48.958)。日志确认 DeepGEMM `GROUPED_GEMM_NT_BF16_MASKED`(N=2048 K=4096 / N=4096 K=1024,num_groups=64)被 JIT 编译,baseline 无此标记。正文解读:c=4 与 c=64 中性/有噪声,其余并发更好,高并发 c=256/512 分别 +6.2%/+8.8%。

准确性:GSM8K 5-shot, max_tokens=16384, temperature=0.0,`Qwen3.5-397B-A17B-NVFP4` 得分 0.974886(输出 3732.387 tok/s),FP8 参照 `Qwen3.5-397B-A17B-FP8` 为 0.976408,精度相当。

## 验证与回归风险

验证以 GSM8K 精度任务与 SLURM sweep 基准为主,无新增单测(Checklist 勾选了 benchmark 项)。影响面被刻意收窄:仅"未量化 + BF16 权重 + DeepEP LL + deep_gemm runner + 非 NPU/HIP"组合触发,其他量化路径不变;删除的 `forward_unquantized_deepep_ll` 只服务该窄场景。NPU/HIP 上 DeepEP LL 本就允许无 DeepGEMM,逻辑未动。风险主要在依赖 #22822 的 `dispatcher_output_dtype` 机制与 #25525 的 MoeRunner 迁移,二者必须同时在场。

## 关联

前序:#24906(最初过宽的 BF16 DeepEP dispatch)、#22822(dispatcher output dtype 机制)、#25525(flashinfer_cutedsl+DeepEP 迁移到 MoeRunner)、#25525 同分支。同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
