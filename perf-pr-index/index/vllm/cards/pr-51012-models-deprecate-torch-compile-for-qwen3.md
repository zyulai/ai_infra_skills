# PR #51012: [Models] Deprecate `torch.compile` for Qwen3.5 models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Isotr0py |
| 合并 | 2026-10-29 |
| 分类 | CUDA Graph / Compiler |
| Label | new-model, torch.compile, ci/build, qwen, cpu, nvidia, kimi, k3 |
| Diff | diffstat 缺失(pack 标注 +None/-None);diff 较长且有 10 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/51012 |

## 一句话结论

Qwen3.5 弃用 torch.compile 改用 breakable CUDA graph + 显式 fused kernel,decode 反超 compile 基线 3~11%。

## 动机

Qwen3.5 此前去掉了 `@support_torch_compile`,改用 breakable CUDA graphs(经 `VLLM_USE_BREAKABLE_CUDAGRAPH`,在 `vllm/config/vllm.py` 自动启用)。隔离实验(同一旧代码,compile vs breakable cudagraph)显示这一切换本身使 decode-step 延迟增加 45~64%、prefill 延迟增加 22~39%,因为丢掉了 inductor 的算子融合。本 PR 用手写 fused kernel 补回这些融合,最终在 decode 上比 torch.compile 基线快 5~11%,prefill 持平。

## 优化点

NVIDIA 平台专属的 Qwen3.5 实现改为 eager + breakable CUDA graph,并补齐一组显式 Triton fused kernel:(1) `fused_pre_fc_norms`,把 MTP drafter 的两次 Gemma 式 RMSNorm(scale by `1+w`)与 concat 融合为单 kernel;(2) `fused_sigmoid_gate_mul`,单 kernel 完成 `x * sigmoid(gate)` 输出门控;(3) `fused_gdn_gated_rms_norm`,直接从 packed qkvz 投影的 strided 视图读 z,省掉 z 的 reshape copy;(4) GDN 的 b/a 保持 strided chunk view 直传,不再 `.contiguous()`;(5) RMSNorm 换 flashinfer `gemma_rmsnorm`/`gemma_fused_add_rmsnorm`,以及 fused all-reduce+RMSNorm 路径。非 contiguous/非 CUDA 输入自动回退 eager,保证通用性。

## 关键改动

```text
vllm/models/qwen3_5/__init__.py: 平台隔离入口,CUDA 走 nvidia 实现,ROCm 等保留旧 torch.compile 实现
vllm/models/qwen3_5/nvidia/model.py: 新增 NVIDIA eager 实现(939 行),Qwen3_5DecoderLayer 安装 fused kernel
vllm/kernels/triton/fused_sigmoid_gate_mul.py: 新增 sigmoid 门控乘法 fused kernel(含 eager fallback)
vllm/kernels/triton/fused_pre_fc_norms.py: MTP drafter 双 GemmaRMSNorm+concat fused kernel
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: GDN strided b/a + fused gated norm 钩子,去掉 z reshape copy
vllm/config/vllm.py: Qwen3.5 全系架构在 CUDA 上默认启用 breakable cudagraph
```

## 性能声明

环境:每 run 1x NVIDIA GB200 (SM100, aarch64),bf16,真实权重(Qwen/Qwen3.5-9B dense、Qwen/Qwen3.5-35B-A3B MoE);服务侧两树均带 `--no-enable-prefix-caching --max-num-batched-tokens 8192`。

- E2E serving(`vllm bench serve`,并发至 128):decode ITL -3.0%~-8.8%,prefill TTFT -7.0%~-8.9%(9B decode cc=32 ITL 4.84→4.50 ms,-7.0%;35B-A3B cc=128 ITL 14.09→13.67 ms,-3.0%)。
- 隔离 forward(`vllm bench latency`):decode step -4.9%~-10.7%,MTP decode -17%~-38%(9B bs32 8.98→5.58 ms),prefill 持平。
- Greedy 解码输出与 torch.compile 基线 token-identical(TP1 与 TP2)。
- 附注:更高并发下差距收窄但不反转;新版默认 prefix caching 曾使 TTFT 虚高 ~17-20%,已用相同 serving flags 规避。

## 验证与回归风险

Kernel 单测 245 项通过(GB200,含 `test_fused_sigmoid_gate_mul.py`、`test_gemma_rms_norm_triton.py`、`test_fused_pre_fc_norms.py`、`test_gdn_qwen35_strided_inputs.py`,GDN strided/fused-norm 路径断言 bit-exact);greedy smoke TP1/TP2 token-identical;`test_qwen3_5_quantization.py` 4 项通过。新增平台分发测试:断言 NVIDIA 变体不继承 `TorchCompileWithNoGuardsWrapper`、legacy 实现保留。风险面:仅 NVIDIA 平台行为改变,ROCm 及其他平台走旧实现;非 16B 对齐/非 contiguous 输入依赖 eager fallback 正确性;breakable cudagraph 默认开启面扩大到 Qwen3.5 全系架构。

## 关联

同区域:breakable CUDA graph 机制此前仅对 `DeepseekV41ForCausalLM` 等默认启用(见 `vllm/config/vllm.py`);索引内 torch.compile 路径相关卡 PR #23207(Qwen2.5-VL 启用 torch.compile),GDN/线性注意力相关卡 PR #24526(Qwen3-Next 支持,本 PR 修改其 `qwen3_next.py` 复用 attention/decoder layer);fused allreduce+RMSNorm 属同方向融合(见 PR #20691)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
