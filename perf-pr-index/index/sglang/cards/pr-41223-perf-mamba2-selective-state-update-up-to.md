# PR #41223: [Perf] Mamba2 selective_state_update up to 2x faster on B200 via 8x1 launch config for dstate 128 (+7.5% Nemotron-3-Super serving)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @TarzanZhao |
| 合并 | 2026-09-26 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);实际仅改动 1 个文件的 1 行配置 |
| 链接 | https://github.com/sgl-project/sglang/pull/41223 |

## 一句话结论

把 Mamba2 decode kernel `selective_state_update` 在 dstate 128 时的 launch config 改为 (8,1),B200 上 kernel 最高提速 2.02x,Nemotron-3-Super 端到端吞吐 +7.5%。

## 动机

Mamba2 decode kernel `selective_state_update` 是 memory bound:每一步都要读写所有请求的 fp32 SSM state。当 64 < dstate <= 128 时,现有 launch 配置为 `BLOCK_SIZE_M=4, num_warps=4`,每个线程只处理 4 个元素,in-flight 的 load 数量太少,无法填满 HBM 带宽——在 B200 上 kernel 只能跑到约 3.1 TB/s,远低于硬件能力。

## 优化点

改动只涉及一处 launch 配置:将 dstate <= 128 档位的 `(BLOCK_SIZE_M, num_warps)` 从 `(4, 4)` 改为 `(8, 1)`,使每个线程处理 32 个元素,显著增加每线程的访存量与 in-flight load,从而填满 B200 的 HBM 带宽(批量 512 时从 3.1 提升到 6.3 TB/s)。作者试了 20 组配置,`(8, 1)` 在 batch 128 和 512 最快,batch 8 时比最快的 `(8, 4)` 落后 4%,但综合最优。该 kernel 为 Triton JIT 实现,配置改动不改 kernel 本体代码,输出与基线 bitwise 一致。

## 关键改动

```text
python/sglang/kernels/ops/mamba/triton_ops/mamba_ssm.py: 修改 selective_state_update
launcher 中 dstate<=128 档位的 (BLOCK_SIZE_M, num_warps) 配置,4,4 -> 8,1
```

关键摘录(diff):

```python
-            else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else ((4, 8))))
+            else ((8, 4) if dstate <= 64 else ((8, 1) if dstate <= 128 else ((4, 8))))
```

## 性能声明

**Kernel 单测**(单张 B200,CUDA graph 计时、冷 L2,headdim 64、dstate 128、fp32 state、bf16 输入、每 GPU 16 heads,对齐 Nemotron-3-Super TP8):batch 8/128/512 分别为 1.19x / 1.86x / 2.02x 提速(baseline 4.77/44.7/173.5 us → 4.02/24.1/85.7 us),batch 512 时带宽 3.1 → 6.3 TB/s。

**端到端**:8x B200(TP8),`sglang.bench_serving`,ShareGPT 1000 prompts,request rate inf,Nemotron-3-Super-120B-A12B-BF16:输出 token 吞吐 10,505 → 11,289 tok/s(**+7.5%**,3 次取均值);每 GPU 峰值显存 164.2 → 163.9 GB。

## 验证与回归风险

作者对比了 baseline 与本分支的 greedy 补全(token id 与 logprob)以及 kernel 单独的输出与更新后 state,全部 bitwise 一致(64 条 ShareGPT prompts x 128 tokens x 3 runs、8 条 3.2k–3.5k tokens prompts、5 种 head 数 x 184 种 batch 共 920 个点)。未新增测试,`test/registered/layers/mamba/test_mamba_ssm.py` 覆盖此路径并在 B200 上 340 passed。影响范围:任何 GPU 上、经默认 Triton `selective_state_update` 且 64 < dstate <= 128 的 Mamba2 decode 模型都会受此配置影响;作者仅在 B200 上实测,H100/H200 未测量(作者指出 vLLM 在 H100/H200 的 tuned config 多选 1–2 warps,方向一致但非实测)。

## 关联

无(pack 内未提及同区域其他 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
