# PR #27896: [Perf] Skip per-call mat_a/scales_a padding in cutlass FP8 blockwise GEMM

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-06-12 |
| 分类 | Speculative Decoding (tier: major) |
| Label | run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/27896 |

## 一句话结论

cutlass FP8 blockwise GEMM 的激活量化改为预对齐 4 行分配,免去每次 GEMM 的 2x fill + 2x cat,并发 64 吞吐 +2.1%。

## 动机

`cutlass_w8a8_block_fp8_linear_with_fallback` 先量化激活,再调 `fp8_blockwise_scaled_mm`,其 sgl-kernel wrapper 每次调用都把 `mat_a` 与 `scales_a` 补齐到 4 行的倍数——每次补齐各是一次 `torch.zeros` fill 加一次 `torch.cat`,即每个 GEMM 多出 2 个 fill、2 个 cat kernel。cutlass SM90 blockwise kernel 要求 A 操作数行数是 4 的倍数,而 NEXTN speculative decode 时 M 只有 1-5(draft token 数),padding 几乎每次都触发;又因量化输出缓冲每次新分配,wrapper 必须重复补齐。

## 优化点

把对齐成本从"每次 GEMM 前"挪到"量化分配时":新增 `sglang_per_token_group_quant_fp8_row_padded`,按 `m_pad = ceil_align(m, 4)` 直接分配 `x_q (m_pad, k)` 与列主序 `x_s (m_pad, k//group)`(先按 `(k//group, m_pad)` 建再转置,保证 `stride(0)==1` 的 M-major 布局),在 `[:m]` 切片上跑既有 v2 量化 kernel,`[m, m_pad)` 行故意不初始化。这样 wrapper 的 `pad_tensor()` 看到 `pad_rows == 0` 直接短路,四个 memory kernel 不再发射;GEMM 在 padded buffer 上运行,caller 把输出切回真实行数。由于 blockwise FP8 GEMM 沿 M 行独立,garbage 行不会被读取,数值结果不变。v2 kernel 不可用(group size 不在 {16, 32, 64, 128})时回退旧量化路径。`weight_scale.T` 仍按原样传入(kernel 要求 `scales_b.stride(0)==1`,内部自行物化),不在本 PR 范围。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_kernel.py: 新增 row-padded 量化函数(v2 kernel + fallback)
python/sglang/srt/layers/quantization/fp8_utils.py: cutlass fallback 路径改用 row-padded 量化并切回输出行数
test/registered/quant/test_fp8_blockwise_row_padding.py: 单元测试(新文件)
```

## 性能声明

H20-3e,TP=4,Qwen3.5-122B-A10B-FP8,NEXTN speculative decoding(num-steps 4、eagle-topk 1、draft-tokens 5),CUDA graph on,radix cache off,310 decode steps 的 torch profiler trace:

- Decode-loop kernel 计数:fill 约 65k launches/~69 ms → 315 launches/0.4 ms;cat 约 63k launches/~89 ms → 364 launches/0.7 ms(cutlass GEMM 与 group quant 计数/耗时不变)。
- 端到端:bs=1 单请求延迟 2.21 s → 2.21 s(无变化,CPU-bound 于 speculative-decode 调度);并发 64 输出吞吐 2617 → 2671 tok/s(+2.1%,四轮运行稳定且与基线不重叠)。

## 验证与回归风险

新增单元测试 `test_fp8_blockwise_row_padding.py`(CI base-b,1-gpu-large):断言缓冲 4 行对齐、scales M-major、live 行与 legacy 量化逐元素一致、完整 linear 对 M in {1, 2, 3, 4, 5, 7, 13, 16, 31, 64, 256} 与 legacy unpadded GEMM `torch.equal`(H20-3e SM90 3/3 通过;去切片的 negative control 按预期失败)。端到端 greedy 512-token 补全前后 byte-identical。风险面:依赖 v2 量化 kernel 可用性(否则回退旧行为);输出切片逻辑必须保留,否则会读到未初始化行;影响所有走 cutlass FP8 blockwise linear 的模型路径(不限于 speculative decode)。

## 关联

同区域:cutlass W8A8 block FP8 linear 与 per-token-group FP8 量化路径;pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
