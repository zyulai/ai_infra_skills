# PR #66791: [Perf][Qwen4Exp] Fuse HyperConnection up projection and gate mix on SM90

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @positive666 |
| 合并 | 2026-11-19 |
| 分类 | Attention / Kernel(tier: major) |
| Label | qwen |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66791 |

## 一句话结论

Qwen4Exp 的 SM90 HyperConnection 路径此前在 up projection 与四流 mix 之间物化一个 10,240 列的 BF16 gate;本 PR 用 Triton 把投影、sigmoid、mixing 融合为一个 kernel(H=2560、K=320、BF16、4 流、>=512 本地行),H20 TP8 官方 nightly 上 32K/128K prefill TTFT 中位数降约 2.8-3.0%,与 QSA(#65606)组合时 261,888 tokens 输入吞吐 +9.14%。

## 动机

SM90 路径上 `hc_up_gate_mix` 只有 SM100 实现(`torch.ops._C.qwen4_exp_hc_up_gate_mix`,来自 #64898),Hopper 回落到 `hc_gate_mix(xn, F.linear(lowrank, w_up), 4)`——先物化 10240 列 BF16 gate 再做四流混合,多一次大张量读写。SM90 与 SM100 数值契约不同:SM90 在 sigmoid 前把 gate 舍入到 BF16,SM100 保持 FP32,故不能直接复用 SM100 kernel。

## 优化点

- 新 Triton kernel `_hc_up_gate_mix_sm90_kernel`:BLOCK 64x128x32、num_warps=4、num_stages=4;每个输出 tile 取四条 HC 流的对应特征(`gate_cols = cols // (BLOCK_N // HC) * HIDDEN + features`),acc 用 FP32 `tl.dot` 循环 K=320。
- 数值契约:照 SM90 路径 `gate = tl.sigmoid(acc.to(tl.bfloat16).to(tl.float32))` 保持 BF16 gate 舍入与 FP32 mixing 顺序(`tl.fma(g3, x3, tl.fma(g2, x2, tl.fma(g1, x1, g0 * x0))) / HC`),复用既有 dispatch 与权重布局,无新并行/通信。
- `hc_up_gate_mix_supported`/`hc_up_gate_mix` 改为 SM90 走 Triton 分支(带形状/dtype/device 断言),SM100 family 且存在 `qwen4_exp_hc_up_gate_mix` 时保持原 op。
- 专业化范围:BF16、4 流、HIDDEN=2560、RANK=320、>=512 本地行(`MIN_FUSED_UP_M = 512`)。

## 关键改动

```python
# vllm/models/qwen4_exp/nvidia/ops/hc.py(pack diff 摘录)
@triton.jit(do_not_specialize=["num_tokens"])
def _hc_up_gate_mix_sm90_kernel(
    lowrank, weight, xn, output, num_tokens, ..., BLOCK_M, BLOCK_N, BLOCK_K
):
    # Each output tile needs matching features from all four HC streams.
    HIDDEN: tl.constexpr = 2560
    RANK: tl.constexpr = 320
    HC: tl.constexpr = 4
    GROUP_M: tl.constexpr = 8
    ...
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for block in range(RANK // BLOCK_K):
        ...
        acc = tl.dot(a, tl.trans(w), acc)

    # Match F.linear's bf16 output and the original stream accumulation order.
    gate = tl.sigmoid(acc.to(tl.bfloat16).to(tl.float32))
    ...
    mixed = tl.fma(g3, x3, tl.fma(g2, x2, tl.fma(g1, x1, g0 * x0))) / HC
```

```diff
-        hasattr(torch.ops._C, "qwen4_exp_hc_up_gate_mix")
-        and current_platform.is_device_capability_family(100)
+        (
+            current_platform.is_device_capability(90)
+            or (
+                current_platform.is_device_capability_family(100)
+                and hasattr(torch.ops._C, "qwen4_exp_hc_up_gate_mix")
+            )
+        )
```

## 性能声明

- H20 TP8/EP8 官方 nightly(`7c3a9302`,Torch 2.14.0、Triton 3.8.0、FlashInfer 0.7.4;A/B/B/A 四次独立 server 启动,2,496 timed requests):32,768/128/4 TTFT 中位数 2,575.6 → 2,502.6 ms(-2.99% / -2.92% 两 boot pair);131,072/128/1 6,236.4 → 6,066.8 ms(-2.79% / -2.75%);4,096/32/4 538.6 → 365.8 ms(-32.19% / -21.03%,但 E2E 仅 -1.36%,4K 中位数对到达节奏敏感,重放历史到达钟后为 -13.49% / -2.29%)。
- 集成镜像 H20 TP8/EP8(重基于 `81a8cb063e62`):32,768/4 TTFT 2.950 → 2.868 s(-2.76%)、131,072/1 6.263 → 6.090 s(-2.77%)、261,888/1 13.263 → 12.945 s(-2.40%);1,024/1 TTFT +2.95%(两 boot pair 反向,+5.71% / -5.68%)。
- 早期 H20 TP4/EP4(基线 U=8db73f26c):32,768/32/4 TTFT 4,825.28 → 4,715.70 ms(-2.27%);131,072/32/1 -2.18%;261,888/32/1 -2.13%;U-to-U TTFT P50 drift 0.01%-0.26%。与 QSA #65606 组合(同一 U 基线):131,072/32/1 组合 4.71%、261,888/32/1 组合 8.34%,261,888 tokens 组合输入吞吐 +9.14%(两独立节省预测 7.90%,作者强调是近似可加性而非普遍线性)。
- 单 kernel 冷 L2 CUPTI:1,024 rows 68.71 → 61.46 us、8,192 rows 498.85 → 390.47 us(不含模型其余部分)。
- 质量:五-shot greedy GSM8K 64 例切片、C1 且每题重置 prefix cache,四个臂均 60/64;32K/128K 检索与算术 8 例 × 4 启动 token ID 全同(64/64)。注意:长随机 greedy 输出非逐位等价,U 自身重复也改 3/4 输出,上游重复波动原因未解决。

## 验证与回归风险

- 18/18 HC 测试通过(含 strided low-rank 输入、奇数行数、BF16 参考对比);`test_hc_up_gate_mix` 参数化加入 512(服务阈值)并保留 1599/1600/4097;3 个 serving-dispatch 检查过 511/512/513 rows;rank-local 源检查与 kernel probe 确认 fused 路径(97 fused HC calls per step);242 个 embedded cubins 对比设备代码/常量/资源元数据/shared memory。
- 风险(正文明示):性能结论限定于该 H20 runtime——旧 Torch 2.14/CUDA 13.0 的 closed-loop 4K run 曾把 TTFT 中位数从 487 ms 恶化到 658 ms,新 runtime 复测 820.09 → 803.06 ms;H100/H200、DP>1、更大部署窗口未测;短请求 TTFT 是 caveat(TP4 历史股票到达钟下 pooled TTFT +38.63%,均值却 622.7 → 618.7 ms、P95 927.6 → 910.5 ms)。

## 关联

- 为 #64898 的 SM100 实现添加 Hopper 路径;组合测量涉及 QSA PR #65606(非依赖);同一 kernel 族的后续 raster 调优见 #68154(测试矩阵中的 1,599/1,600 边界来自它)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
