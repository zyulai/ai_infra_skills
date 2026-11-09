# PR #59069: [ROCm][Kimi-K3] Fuse AttnRes output with per-token FP8 input quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @whx-sjtu |
| 合并 | 2026-10-08 |
| 分类 | Quantization \| tier: major |
| Label | performance, rocm, ready, quantization, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/59069 |

## 一句话结论

Kimi-K3 AttnRes 融合 per-token FP8 量化，16384 tokens segment 682.589→657.554 µs。

## 动机

Kimi-K3 的 AMD AttnRes 输出会先物化成 BF16 激活，而紧随其后的 linear 又立刻把它量化。PR 把这一 per-token FP8 转换融合进 AttnRes，向支持的 consumer 直接传原生 `QuantizedActivation`（store FP8 + scale → input GEMM），省掉独立的量化 kernel。

## 优化点

仅对 `kFp8DynamicTokenSym` 且所有 consumer 接受同一 key 时启用该路径；混合 MoE/router consumer、不支持的 linear kernel、batch-invariant 模式仍走原路径。保留量化前的 BF16 舍入与 prefix/block write 契约。`amd/ops/attn_res.py` 在 Triton kernel 中增加 `QUANT_MAX` 并返回 `(output, scale)`；`amd/linear.py`、`amd/mla.py`、`amd/kda.py` 适配原生量化输入。

## 关键改动

```text
vllm/models/kimi_k3/amd/ops/attn_res.py: 在 AttnRes 内核内做 per-token FP8 量化并返回 (output, scale)
vllm/models/kimi_k3/amd/linear.py、mla.py、kda.py: 消费原生 QuantizedActivation
tests/models/kimi_k3/test_amd_attn_res.py、test_amd_mla_direct_return.py: 数值与 capability 测试
benchmarks/kernels/benchmark_kimi_k3_amd_fp8_fusions.py: 八卡 segment 基准
```

```python
+        QUANT_MAX=0.0 if quant_dtype is None else torch.finfo(quant_dtype).max,
...
-    return output
+    return output if scale is None else (output, scale)
```

## 性能声明

Kernel 段基准 `benchmark_kimi_k3_amd_fp8_fusions.py`：八块 MI355X GPU、TP8/DCP8，九轮 rotated CUDA-graph samples/rank、median of per-rank medians、四个 residual blocks，比较 AttnRes + quant + 一个 MLA input GEMM（N2112/K7168）。Tokens→Before/After（µs）：16 → 17.110/16.597；256 → 31.701/30.168；4096 → 183.573/173.933；16384 → 682.589/657.554。PR 注明这是 producer/projection segment benchmark，one-token 基本不变，不证明 PD 吞吐增益。paired rank-0 prefill trace：独立 AITER per-token quant kernel 调用数 4009→3078（去掉 931 次），AttnRes 仍 3553 次。全模型集成（与 #59070 合并、非本 PR 单独归因）：TTFT cold 32K 1701.03→1673.86 ms、exact 32K repeat 83.37→77.13 ms、TPOT 9.867→9.840 ms；GSM8K 前 128 题 5-shot 127/128（99.21875%）before/after 一致。

## 验证与回归风险

CI（2026-10-08）：AttnRes 与 MLA/KDA direct-return 套件在 AITER disabled 23 passed、enabled 23 passed；pre-commit 与 Python 3.12 mypy 通过。数值：scale rtol 1e-6、舍入差异限于相邻 FP8 code 且 <1e-4 的值。回归风险：Blast Radius 标为 ROCm；两向 revert 恢复原路径、无 checkpoint/cache 格式变更；作者注明 full-dataset accuracy 与 isolated per-PR 模型测量仍为合入前 review 工作，且 outputs/logprobs 非 bit-exact。

## 关联

正文明确列出：去重检查涉及 #50682、#52968、#53154；配套 DCP prefill fusion 为 #59070；本 PR 独立于 PD connector 工作，且未宣称所有 KDA 输入都被融合（rowwise Torch FP8 KDA kernel 未暴露所需 input capability）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
