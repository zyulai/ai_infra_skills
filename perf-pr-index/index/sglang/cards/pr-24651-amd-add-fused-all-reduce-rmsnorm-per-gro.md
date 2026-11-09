# PR #24651: [AMD] Add fused all-reduce RMSNorm per-group quant for Qwen3.5 FP8

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw |
| 合并 | 2026-07-22 |
| 分类 | Quantization |
| Label | documentation, quant, amd, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/24651 |

## 一句话结论

在 AMD/aiter 上把 TP all-reduce、RMSNorm 与 per-group FP8 量化熔成单 kernel,Qwen3.5 FP8 低并发 decode 吞吐 +4.6%~5.8%。

## 动机

Qwen3.5 FP8 的 attention/GDN 输入归一化路径中,`prepare_attn` 之前 all-reduce、RMSNorm、per-group 激活量化是三次独立执行,FP8 投影再消费 `(fp8, scale)`。每层多次 kernel launch 与中间激活往返在小批量 decode 下开销明显,而 aiter 已具备单 kernel 完成 AR+RMSNorm+per-group quant 的能力,值得直接接入。

## 优化点

新增 TP API `tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group` 与 `GroupCoordinator.fused_allreduce_rmsnorm_quant_per_group`:单 kernel 分发仅限 ROCm + aiter + gfx95(`is_gfx95_supported()`),并按 K % group_size、总字节数(<= 8*1024*8192)、world_size 等条件 fail-fast,失败返回 None。`LayerCommunicator.prepare_attn` 在 `_sglang_needs_allreduce_fusion` 且 layer 声明 `enable_fused_ar_quant=True` 时优先走融合 helper,回退顺序为:单 kernel AR+RMSNorm+quant → 已有 fused AR+RMSNorm 再加独立 per-1x128 quant(2 kernel)→ 通用路径。GDN 层用 `keep_bf16=True` 让 kernel 同时写出 bf16 旁路输出,`in_proj_qkvz` 直接吃 `(fp8, scale)`、`in_proj_ba` 读 bf16,避免反量化;标准 attention 用 `keep_bf16=False`。`SGLANG_DISABLE_FUSED_AR_QUANT=1` 提供算子级退出。

## 关键改动

```text
python/sglang/srt/distributed/parallel_state.py: GroupCoordinator 融合入口与 gfx95/形状 eligibility
python/sglang/srt/layers/layernorm.py: _forward_with_allreduce_fusion_quant_per_group 三级回退
python/sglang/srt/layers/communicator.py: prepare_attn 接线(enable_fused_ar_quant / keep_bf16)
python/sglang/srt/models/qwen3_5.py: 模型侧 gate 与 (bf16, fp8, scale) tuple 消费
benchmark/kernels/all_reduce/benchmark_fused_ar_rms_quant_amd.py: 三种路径微基准 + 正确性检查
```

```python
if self.enable_fused_ar_quant and _use_aiter and hasattr(
        self.input_layernorm, "forward_with_allreduce_fusion_quant_per_group"):
    quant_result = self.input_layernorm.forward_with_allreduce_fusion_quant_per_group(
        hidden_states, residual, use_attn_tp_group=False,
        keep_bf16=self.fused_ar_quant_keep_bf16)
if quant_result is not None:
    hidden_states, residual = quant_result
```

## 性能声明

Qwen3.5-397B-A17B-FP8,bench_serving random 8192 input/1024 output,ratio 0.8(TP=4 server 命令):cc=2 output tok/s 186.42→197.21(+5.8%),mean TPOT 10.46→9.88 ms(+5.5%);cc=8 598.54→625.89(+4.6%);cc=32 1332.17→1328.84(-0.2%,TPOT -0.4%)。GSM8K(TP=8,1319 题 5-shot):Baseline 0.946 → Fused 0.955,高于 0.94 门槛。kernel 数剖析(TP=8,60 层,每 decode pass):移除 plain AR 60、add_rmsnorm_quant 60、dynamic_per_group_scaled_quant 60、GDN comm 侧独立量化 45,新增 `allreduce_fusion_kernel_1stage_per_group` 60,`prepare_attn` 合计 -165 kernels。

## 验证与回归风险

新增 AMD 微基准脚本内置解量化正确性检查(fp8 相对误差 < 0.15、bf16 旁路与 fp8 反量化差 <= 1.0),覆盖 eager prefill 与 CUDA graph decode 两种模式及 Qwen3.5-397B hidden=4096 形状;accuracy 侧为 GSM8K e2e。风险面:单 kernel 分发严格限 ROCm/aiter/gfx95,`parallel_state` 中对 world_size==6 与 TP=8 K∈(4096,7168] token>=8 的 1-stage/2-stage 边界有特判,其他硬件与非 Qwen3.5 模型走原回退链不受影响;`enable_aiter_allreduce_fusion` 开启时模型侧自动启用融合 quant,可用环境变量关闭。pack 中 CI Extra 一项失败。

## 关联

前序为已有 `--enable-aiter-allreduce-fusion`(AR+RMSNorm 两算子融合)路径,本 PR 是其向量化 quant 的扩展。同区域(AMD/quant)其他 PR 素材未提供,无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
