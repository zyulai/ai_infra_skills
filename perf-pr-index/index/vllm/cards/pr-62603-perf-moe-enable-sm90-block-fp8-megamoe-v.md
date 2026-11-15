# PR #62603: [Perf][MoE] Enable SM90 block-FP8 MegaMoE via FlashInfer MoE-EP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @qianlihuang |
| 合并 | 2026-11-12 |
| 分类 | MoE(tier: major) |
| Label | ready, deepseek, nvidia, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/62603 |

## 一句话结论

用 FlashInfer 0.7.2 起随附的 SM90 FP8 MegaMoE(`Sm90PullFp8MegaMoeConfig`,单 kernel 融合 dispatch、两个 expert GEMM 与 combine)让 block-FP8 MoE 模型跑上 Hopper:GLM-5.3-Flash decode(8×H200, DP8/EP8)output 吞吐 +11% 至 +18%、TPOT p50 −6% 至 −14%,prefill 饱和吞吐 +23%;backend opt-in,默认后端不变。

## 动机

vLLM 此前没有能跑在 Hopper 的 MegaMoE:DeepGEMM MegaMoE(`deep_gemm_mega_moe`)与 FlashInfer MoE-EP megakernels(`flashinfer_moe_ep_cutedsl`、`flashinfer_moe_ep_mega_deep_gemm`)都要求 SM100。本 PR 走既有 `--moe-backend flashinfer_moe_ep_cutedsl` 路径接入 FlashInfer 的 SM90 FP8 MegaMoE,无模型代码改动,兼容 128x128 block-FP8 MoE 模型即可用。

## 优化点

- **Kernel**:FlashInfer `Sm90PullFp8MegaMoeConfig`,FP32 128x128 权重 scale 与 SwiGLU clamp;push-style SM90 backend 无 SwiGLU clamp,不能服务 `swiglu_limit` 模型,故选 pull-style。
- **权重布局**:kernel 要求 FC1 行按 `[pair, {gate, up}, 8]` interleave 且同一 128x128 scale——每个 scale block 覆盖两个 checkpoint block 的行,两者取 `max(s_gate, s_up)`;2 的幂 scale ratio 下留在 E4M3 normal 范围内的值不变,更小的 rescale 值可能 round,其他 ratio 会在 rescale 的 FC1 行上多一次 E4M3 rounding。`sm90_fp8_block_moe_ep_weights` 负责重排与 scale 合并。
- **Workspaces**:kernel 总是处理整个 token buffer(launch geometry 按该 size 选),单一 `max_num_batched_tokens` buffer 会让 decode 慢于 baseline;adapter 增加 64/256/1024-token workspace,每次调用取能容纳 DP 各 rank 最大 batch 的最小者,拿不到 DP-wide token counts 时用默认 workspace(各 rank 必须选同一 buffer)。
- **Loaders**:fp8、compressed-tensors、online FP8、Quark FP8 四个 MoE quant 方法现在调用 experts 的 `process_weights_after_loading`(对其他 experts 是 no-op);SM90 路径从 quant config 而非 weight dtype 判定,FP8 权重保持 dtype,从预处理权重重建时选同一路径。
- **Idle ranks**:pass-through stages 下 experts 拥有 EP collective,无 token 的 rank 也要进入(与 #62073 的改动相同)。
- **限制**:需要 expert parallelism 与 `nvshmem4py`;拒绝 DBO、runtime weight transfer、`apply_router_weight_on_input`(SM90 kernel 在 fc2 前加权 activation,不等价于加权 expert 输入);shared experts 不融合;FlashInfer SM90 input staging 是 torch 组合的,C128 下每步约 1.2 ms。

## 关键改动

注:pack 的 diff 被截断(注明 7 个文件略去),以下仅覆盖已展示部分。

```text
vllm/model_executor/layers/fused_moe/flashinfer_moe_ep.py: 权重格式增加 fp8_block 与 (9,0) capability 映射;sm90_fp8_block_moe_ep_weights(gate/up interleave + shared scale);_SM90_WORKSPACE_CAPACITIES=(64,256,1024);API 增加 Sm90PullFp8MegaMoeConfig
vllm/model_executor/layers/fused_moe/experts/flashinfer_moe_ep.py: supported schemes 增加 kFp8Static128BlockSym;process_weights_after_loading 走 block_fp8 适配器(拒绝 apply_router_weight_on_input,scale 参数名兼容 weight_scale_inv)
vllm/config/kernel.py: flashinfer_moe_ep_cutedsl 文档更新(Blackwell NVFP4/MXFP4 或 Hopper block-FP8)
tests/kernels/moe/test_flashinfer_moe_ep.py: SM90 block-FP8 精确输出/aliasing/idle-rank/CUDA-graph 测试与 4 个模型 geometry
tests/models/test_deepseek_v4_fi_moe_ep.py: oracle/fp8 backend 映射与 SM90/SM100 支持 reason 测试
(quantization loaders 等 7 个文件被略去)
```

```python
+# The SM90 FP8 megakernel folds SwiGLU over FC1 rows ordered
+# [pair, {gate, up}, _SM90_GATE_UP_INTERLEAVE].
+_SM90_GATE_UP_INTERLEAVE = 8
+_FP8_BLOCK = 128
+# The SM90 FP8 megakernel runs its whole token buffer, with a launch geometry
+# picked for that size, so small batches get small buffers of their own.
+_SM90_WORKSPACE_CAPACITIES = (64, 256, 1024)
+
+
+def sm90_fp8_block_moe_ep_weights(
+    w13: torch.Tensor,
+    w13_scale: torch.Tensor,
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """Reorder block-FP8 `w13` (gate rows, then up rows) for the SM90 kernel.
+
+    After the gate/up interleave a 128-row scale block holds 64 gate and 64 up
+    rows from two checkpoint blocks. Both take the larger of the two scales and
+    the other half is rescaled. Power-of-two scales make that an exponent
+    shift, so only values that drop out of the E4M3 normal range round.
+    """
+    num_experts, two_i, hidden = w13.shape
+    intermediate = two_i // 2
+    num_blocks = intermediate // _FP8_BLOCK
+    assert intermediate % _FP8_BLOCK == 0 and hidden % _FP8_BLOCK == 0, w13.shape
+    gate_scale, up_scale = w13_scale.split(num_blocks, dim=1)
+    shared_scale = torch.maximum(gate_scale, up_scale)
```

## 性能声明

- **Decode**(8× H200,DP8/EP8,DecodeBenchConnector,ISL 128K/OSL 512,MTP3 synthetic acceptance [0.8, 0.6, 0.5],两 pass/臂,serial A/B/A):output tok/s base 911/2895/5204/7977/10074 → 本 PR 1009(+10.8%)/3402(+17.5%)/5893(+13.2%)/9326(+16.9%)/11917(+18.3%),并发 8–192;TPOT p50 7.60→6.86 / 9.20→7.96 / 9.27→8.72 / 10.81→9.85 / 11.58→10.87 ms。第二次 commit 的一次额外 pass 相同 base 为 +8% 至 +24%。
- **Prefill**(4× H200,TP2×DP2+EP4,natural MTP3,ISL 2048/OSL 1):饱和 req/s 21.44 → 26.35(+22.9%,3 次中位);TTFT p50 在 11/15/19/21 req/s 为 370/488/787/1365 → 409/418/515/630 ms;TTFT p99 同序 838/939/1438/2699 → 737/808/820/1059 ms。基线在 23/25 req/s TTFT p50 达 3.8/4.5 s,本 PR 能在 610/831 ms 服务 22.7/24.7 req/s;11 req/s 时 median TTFT 高 10%(base 两臂 351/389 ms)。
- **单 MoE layer,EP8**(GLM-5.3-Flash expert shapes,ms/layer):8 tokens/rank 0.265 → 0.247(−7%),512 0.834 → 0.461,8192 9.204 → 4.798(至 −48%)。
- **GSM8K**(1,319 题,5-shot,natural MTP3):DEP8 0.917 → 0.919;TP2/DP2 0.918/0.920 → 0.917/0.920;MTP acceptance 0.804 → 0.802——两种配置 comparable。
- 正文注明 serving 与 GSM8K 测量为历史数据,未为本次 merge 重跑。

## 验证与回归风险

- FlashInfer 0.7.3:CPU 32 passed;H200 4 passed(合成 GLM-5.3-Flash 与 Qwen3.8-Flash-Next expert geometry,EP2/EP4,精确输出、storage aliasing/swaps、eager idle-rank 参与、非空 CUDA Graph replay);同一选择在 FlashInfer 0.7.2.post1 也通过。
- 验证用当前 Python 源码跑在既有 native vLLM 0.31 / PyTorch 2.13.0+cu130 build 上(非完整 native rebuild);local pre-commit 与独立 code review 通过。
- 风险:backend 需要专家并行与 `nvshmem4py`,CI 不运行 SM90 GPU 测试(至少两块 Hopper GPU);拒绝 DBO / runtime weight transfer / `apply_router_weight_on_input`;shared experts 不融合;非 2 的幂 scale ratio 会有额外 E4M3 rounding;SM90 input staging 每步约 1.2 ms(C128);workspace 选择依赖 DP-wide token counts,否则用默认(单 buffer 会拖慢 decode)。

## 关联

- #58540:同一 kernel 用于 DeepSeek-V4 FP4→FP8 转换 checkpoint(在被 #54049 移除的文件上);#62073:Beam on SM100(相同 idle-rank 改动);#60895:GLM-5.3-Flash 的 DeepGEMM MegaMoE(SM100 only);#53527:DeepSeek-V4 经 DeepGEMM fork。FlashInfer upstream:#4113(SM90 pull-style FP8 MegaMoE)、#6999(SM90 W4A8,超出本 PR scope)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
