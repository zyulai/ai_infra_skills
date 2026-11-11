# PR #63540: [Perf][DSv4] Process 4 heads per program in fused_inv_rope_fp8_quant

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @guanzheh |
| 合并 | 2026-11-07 |
| 分类 | Speculative Decoding(tier: major) |
| Label | ready, deepseek, quantization, DSv4 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/63540 |

## 一句话结论

DeepSeek-V4 的 fused_inv_rope_fp8_quant 每 program 处理 4 heads,16k-token kernel 在 B200 提速 3.53x。

## 动机

`fused_inv_rope_fp8_quant` 对 attention 输出施加 inverse RoPE 并量化为 block FP8 供 `wo_a` 使用。原实现每个 (token, head) 启动一个 single-warp program:一个 512 维 head 只有 1 KiB bf16,每 program 在-flight 字节太少,饱和不了大 prefill 的 HBM——B200 上仅 1.5 TB/s,占 16k-token DeepSeek-V4-Flash prefill step(TP2)的 8.6%。

## 优化点

每个 program 改为处理同一 output group 的 4 heads(`_heads_per_program = gcd(heads_per_group, 4)`;H24_G4 等每组 6 heads 的形状自动取 2),grid 第二维相应缩小。GPT-J inverse RoPE 的 (even, odd) pairs 改在寄存器内 swap,不再为每个元素从 global memory 重读 partner。`num_warps` 依 program 数分派:program 数 <2048(decode)用 4 warps 换 latency,≥2048(大 prefill)用 1 warp 让更多 program 驻留、保持 HBM 忙。输出与 main bitwise identical(FP8 values 与 scales 均一致),且无 runtime JIT compile(warmup keys 覆盖两个 num_warps choice)。

## 关键改动

```text
vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py: kernel 引入 BLOCK_H,寄存器内 RoPE,dispatch/warmup 增加 num_warps
tests/kernels/test_fused_inv_rope_fp8_quant.py: 新增 H24_G4 参数化用例(6 heads/group,每 program 2 heads)
```

```python
# Few programs (decode): more warps per program for latency. Many (prefill):
# one warp each, so more programs stay resident and keep HBM busy.
num_warps = 4 if grid[0] * grid[1] < 2048 else 1
```

## 性能声明

Kernel 微基准(FlashInfer CUPTI,CUDA graph,cold L2,median µs;H=32 heads/rank;Modal:CUDA 13.0、torch 2.14.0+cu130、Triton 3.8.0):16k tokens 时 B200 536.32→152.03 µs(3.53x,5.3 TB/s)、H100 631.04→293.28 µs(2.15x,2.7 TB/s)、H200 631.33→207.84 µs(3.04x,3.9 TB/s);T=1 时 B200 2.72→2.85 µs(0.95x),decode sizes 至多慢 0.3 µs/call。`vllm bench forward`(V4-Flash TP2+EP,2×B200,dummy weights):prefill 16384 tokens 237336→222768 µs(0.940,−14.4 ms);prefill 与 mixed batches 快 6%,decode 不变;该 kernel 的 43 次 launch 共降 17.2 ms。`torch.profiler`:main grid `[16384,32,1]` 534 µs → PR `[16384,8,1]` 156 µs;驻留 CTA 每 SM 32→28,每 CTA 处理 4 heads,约 3.5x 更多在-flight 数据。`vllm bench serve`(real weights,8k in / 1k out):TTFT p99 在 concurrency 1–128 降 5.1–5.6%;TTFT p50 与 throughput 运行间噪声大,仅声明 TTFT p99。

## 验证与回归风险

单测 106 passed on B200(两种 scale layout、quant groups 32/128、`quantize=False`、两个 num_warps 分支、新 H24_G4);H100/H200 93 passed + 8 个 SM100-only skip(5 个 `test_einsum_end_to_end` 失败 main 上同样存在)。bitwise 对比 main:B200/H100/H200、T=1–16384、H=16/32/64、head-padded 与 offset views、extreme values,无 mismatch。gsm8k(B200 CI config `DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`,TP2+EP,FP8 KV,MTP,各 3 次):main 0.9469/0.9492/0.9507,PR 0.9492/0.9500/0.9484,MTP acceptance 2.63–2.64。风险:decode 尺寸约 0.95x 的轻微回退已被测量并接受;改动限于 DSv4 该 kernel 及其测试。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
