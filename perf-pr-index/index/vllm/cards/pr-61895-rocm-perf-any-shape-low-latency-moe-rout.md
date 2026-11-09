# PR #61895: [ROCm][Perf] Any-shape low-latency MoE router GEMM on MI355

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @AndreasKaratzas |
| 合并 | 2026-10-16 |
| 分类 | MoE（tier: major） |
| Label | rocm, ci/build |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 7 个文件（CI yaml、gate_linear.py、rocm_fp32_router_gemm.py 及两个测试文件） |
| 链接 | https://github.com/vllm-project/vllm/pull/61895 |

## 一句话结论

MI355 (gfx950) 上把 MoE router GEMM 低延迟 kernel 从 5 个硬编码形状泛化到任意形状，decode 提速 1–14%。

## 动机

此前 gfx950 上只有 5 个硬编码 `(hidden_size, num_experts)` 形状走低延迟 router GEMM 快速路径，其余模型的 gate 线性层退回 `F.linear`/hipBLASLt，每 MoE 层每 decode step 付出 9–13 µs。目标是对齐 NVIDIA 侧 CuTeDSL `ll_fp32w`/`ll_bf16` router kernel（#44343、#50174）的覆盖面。

## 优化点

将 `rocm_fp32_router_gemm` Triton kernel 泛化为任意形状：(1) 引入 `EVEN_K` 编译期分支与非整除 K 的 masked tail 加载，支持 64/100/513 等任意 hidden_size；(2) `_launch_config` 重写为在 MI355 上按 12 个形状 sweep 拟合的通用 launch 规则——窄 router（<64 experts）保持 `BLOCK_M=1`，宽 router 随 token 数增大 `BLOCK_M` 以复用权重；fitted 于 12 形状、10 个 held-out 形状验证；(3) 支持 bf16 权重（不再要求 fp32 权重拷贝），输出仍为 fp32；(4) 增加体积 guard：`M·N·K ≥ 2^27`（fp32）或 `tokens×experts > 8192`（bf16）时回落原路径；(5) `GateLinear` 分层中 gfx950 的 fp32 router tier 对任意形状启用，tier 6（bf16 权重 fp32 logits）在 gfx950 上也改走同一个 dispatch custom op（`fp32_router_gemm_dispatch`），运行时分支保持在 custom op 内以保证 torch.compile 安全。仅 gfx950；gfx942、CUDA、biased/quantized gate、batch-invariant 模式不变。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/router/rocm_fp32_router_gemm.py: 泛化 kernel（masked K tail、任意形状、bf16 权重、sweep 出的 launch 规则与 size guards）
vllm/model_executor/layers/fused_moe/router/gate_linear.py: gfx950 tier2/tier6 改走 fp32_router_gemm_dispatch，移除硬编码形状检查
tests/kernels/test_rocm_fp32_router_gemm.py: 新增任意 hidden_size、特殊值、确定性、bf16 权重与 GateLinear 分派测试
.buildkite/test-amd.yaml: 新增 "(MI355) Gate Linear Router Kernels" CI job（847 passed）
```

kernel 分派核心（tier 6 gfx950 路由）：

```python
if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
    if self.is_gfx950:
        output = torch.ops.vllm.fp32_router_gemm_dispatch(x, self.weight, False)
```

## 性能声明

端到端 MI355（`vllm bench serve`，random dataset，prefix caching off，1–8 input/256–512 output tokens，main 与 PR 同时在两块 GPU 上交替跑，decode TPOT Δ 取 5 次中位数）：8 个受影响 MoE 模型 decode 1–14% 变快，低并发收益最大——ERNIE-4.5-21B-A3B +14.1/+3.4/+2.1%（C1/C8/C32）、Trinity-Nano +9.6/+4.9/+4.0%、Gemma-4-26B-A4B-it +8.9/+5.5/+3.5%、DeepSeek-V4-Flash(TP2) +2.0/+1.2/+1.0%；控制组 Qwen3-30B-A3B 持平（−0.2/0.0/+0.2%）。8K prefill TTFT 全部持平（±0.7% 内）。机制解释：router GEMM 每 MoE 层每步从 9–13 µs（F.linear）/5–11 µs（hipBLASLt）降至 ≤8 token 时 1.9–4.9 µs，每步省 0.2–0.6 ms。kernel 级（MI355 单卡，30 次 graph replay 中位数，M=1–32）：fp32 新形状 1.2–6.8× vs F.linear（中位 4.1×）；bf16 1.7–5.4× vs hipBLASLt（M≤8）、1.04–2.9×（M=16–32）。精度：GSM8K 5-shot 全模型与 main 持平（McNemar p=1.0/0.34）；kernel 对 float64 最大相对误差 1.3e-7（fp32）/8.9e-8（bf16）。

## 验证与回归风险

新增专用 CI job 跑三个 router 测试文件（847 passed，10/10 loops）；每个新测试都对变异 kernel/dispatcher（无 K-tail mask、旧形状表、绕过 dispatcher、无 bf16 cutoff、无 tier-6 路由）可失败。风险主要在 dispatch 回落逻辑：bf16 路径在 `tokens×experts > 8192` 与 fp32 路径在 `M·N·K ≥ 2^27` 时回落 `torch.mm`/`F.linear`，guard 判断失误可能让本该走快速路径的形状回落或反之；旧形状硬编码测试（(2048,64)）被移除。Scope 限于 gfx950，NVIDIA 与 gfx942 路径不受影响。

## 关联

同区域：#44343、#50174（NVIDIA CuTeDSL `ll_fp32w`/`ll_bf16` router kernel，本卡对齐其能力）；#59023（Kimi-K3 7168×896 gate 单形状 bf16 扩展）；#57626（强制 5 个形状用 fp32 权重）；#61687（AITER tier，正交）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
