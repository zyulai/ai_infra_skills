# PR #61355: [Perf] Tune Triton GDN decode tiles on SM100-family GPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-10-13 |
| 分类 | Speculative Decoding(major) |
| Label | (none) |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 dispatch 源文件与对应测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/61355 |

## 一句话结论

SM100 家族上为 Triton GDN packed decode 选择更小的 value tile,GB300 上 B=1–64 内核提速 1.34–1.74x(FP32 state),E2E 吞吐 +1.0–2.2%。

## 动机

现有 Triton packed GDN decode 启动在 SM100 上小 batch 利用率低:baseline 启动 4×B×HV 个 CTA,batch 小时 CTA 太少,且 FP32 recurrent state 下每 CTA 工作量偏大;更小的 value tile 可以同时改善占用与每 CTA 负载。

## 优化点

12 行 dispatch 改动:K=V=128 且 CUDA SM100 家族(is_device_capability_family(100, 输入张量 device))时,B×HV<=64 选 BV4,否则 FP32 state 选 BV8;其余(含 BF16 state 的 B>=8)保留 BV32。num_warps=1、num_stages=3 不变。BV4 在小 batch 提供约 8 倍 CTA 数;16-bit state 的大 batch 在调优中小 tile 反而回退,故保留 BV32。capability 查询用输入张量的 device;MRv2 warmup/graph capture 已编译可达变体,不引入额外 warmup、同步或在线 autotune;prefill 与投机验证走独立路径,FlashInfer 仅作对照、未集成。

## 关键改动

```text
vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:
  SM100 家族 BV4/BV8/BV32 启发式
tests/kernels/fla/test_fused_recurrent_packed_decode.py:
  参数化扩展到 batch/头布局/activation/state dtype
```

```python
if (K == V == 128 and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100, device_id=dev.index)):
    if B * HV <= 64:
        BV = 4
    elif initial_state.dtype == torch.float32:
        BV = 8
```

## 性能声明

GB300(152 SMs),BF16 QKV,(H,HV)=(4,12),K=V=128,Q/K norm on,CUPTI+CUDA graph+cold L2:FP32 state B=1–64 提速 1.34–1.74x(B=1: 5.36→3.09 µs);BF16 state B=1/2/4 为 1.29x/1.34x/1.21x,B≥8 为 1.00x。TP1–TP8 泛化(FP32 state)1.19–1.87x。E2E:Qwen3.8-Flash-Next-FP8,4×GB300 TP4+EP、MRv2、无投机,NVIDIA SPEED-Bench throughput_8k:输出吞吐 +1.0–2.2%(如 C=4: 714.7→730.4 tok/s),GSM8K 五-shot 与 C=8 检查分数逐题不变。

## 验证与回归风险

新增 GPU reference 套件 tests/kernels/fla/test_fused_recurrent_packed_decode.py,14 passed on GB300(覆盖 BV4/BV8/BV32、独立 activation/state dtype、contiguous/strided QKV、padded/null state 槽位);Triton 输出与 state 与 baseline bit-for-bit 一致;148-SM green context 验证 B200 占用边界(非物理 B200 性能测量);现有 B200 CI kernels-fla-ops-test-b200 运行该套件。风险限于 GDN decode 的 tile 选择,其他硬件/后端路径不变。

## 关联

#54181(SM120 调优)、#60400(ROCm 调优)、#60403(FlashInfer decode opt-in,未集成);硬件/后端范围互不相同。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
