# PR #46024: [Perf] mhc: Split hc_combine_norm_mxfp8 rows by batch size and zero padding scales in the kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @akhilg-nv |
| 合并 | 2026-10-29 |
| 分类 | Speculative Decoding |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失;PR 自述 +41/−25(运行时,1 文件)、+74/−1(测试,1 文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/46024 |

## 一句话结论

DeepSeek-V4.1 mHC combine 内核按行数选 CTA 拆分并在内核内清零 padding scale,单核快 20-46%。

## 动机

hc_combine_norm_mxfp8 是 DeepSeek-V4.1 mHC 在 Blackwell 上构建 attention 输入的内核:按 pre 权重组合 4 路 residual stream、做 RMSNorm,并以 BF16 与 MXFP8 输出(供 wqkv_a)。batch ≤128 行时 decode 每步 39/40 层都会跑,此前有两个固定开销:固定 4 CTA/行且每个 CTA 重算整行(该拆分是行数上限只有 8 时定的,#42245 提到 128 行后未 revisit);MXFP8 scale buffer 用 torch.zeros 分配,每次调用前多一个 fill kernel。目标是在保持输出 bitwise 一致的前提下消除两者。

## 优化点

`_parts_for(m, k)` 改为与同文件 hc_combine_norm 相同的按行数拆分:≤8 行 4 CTA/行、≤48 行 2、更多行 1(≤8 行拆分不变)。scale buffer 改为 torch.empty,网格在最后一行之后追加 padding program,每个负责把 scale buffer 中一个 512 字节块里超出 batch 的行清零(与 rmsnorm_mxfp8 的做法一致);内核新增行数参数 M,per-row 逻辑不变地放进 `if row < M`。每个 CTA 仍按相同顺序计算整行,拆分只决定哪个 CTA 存哪些列,因此输出 bitwise 不变。

## 关键改动

```text
python/sglang/kernels/ops/layernorm/hc_combine_norm.py: 按行数选择 parts;_alloc 改 torch.empty;内核尾部分支清零 padding scale
test/registered/kernels/ops/layernorm/test_mxfp8_epilogue.py: 新增 split-invariant 与 Triton interpreter padding 测试;bitwise 测试补 48/49 行
```

```python
def _parts_for(m: int, k: int) -> int:
    parts = 4 if m <= 8 else (2 if m <= 48 else 1)
    while parts > 1 and (k % (parts * 32)):
        parts //= 2
    return parts

# _alloc: scale buffer 不再用 torch.zeros
s = torch.empty((k // 32) * (triton.cdiv(m, 128) * 128), dtype=torch.uint8, device=device)
```

## 性能声明

Kernel microbenchmark(单 GPU、CUDA graphs,BF16 x [m, 4x5120] + FP32 pre,7 次重复取中位、每次 100 次 graph replay):49-128 行平均 B200 6.24→3.40us(−46%)、B300 5.63→3.20us(−43%)、VR200 4.56→2.68us(−41%);1-32 行各快 20-26%;fill kernel 消失。E2E:DeepSeek-V4.1-Flash,4×B200 单机,InferenceX AgentX DEP4 c192,真实 speculative acceptance,2 组 ABBA:per-request decode time per token −0.32%(95% 区间 −0.56%~+0.02%,在 0.6-1.1% 的 run-to-run 噪声内),output tok/s/GPU −0.01% 与 −0.12%,结论 no measurable change;原因:decode 时 combine 与同 residual 的 mHC 统计(side stream)并发,combine+GEMM 迭代 19.82→21.89us(48-120 行均值)。估算上限(未实测):B200 DEP4 c64 约 0.8%/decode step。GSM8K under AgentX 负载:97.67%→96.33%(exact McNemar p=0.29)。

## 验证与回归风险

test_mxfp8_epilogue.py 在 B200 上 107 passed;对旧内核 byte-for-byte 3,082-3,850 案例 0 mismatch(B200/B300/VR200),覆盖 1-128 行、scale 1e-3/1/1e3、拆分强制 1/2/4、10 次 CUDA-graph replay;新 interpreter 测试(TRITON_INTERPRET=1 子进程)专门覆盖 GPU 测不出的"padding program 误写 live row"盲区。作用域:仅 Blackwell + MXFP8 输出 + hidden 5120 进入该内核,Hopper/ROCm 不受影响;PR 如实记录 B200 上与 DeepGEMM side-stream 统计并发时 72 行起每迭代慢 1.3-4.0us。

## 关联

#42245(把 fusion batch 上限提到 128 行)、draft #41628(同一拆分包含在更大改动中)、同文件 hc_combine_norm 的既有拆分。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
