# PR #50686: [Perf] [AMD] gemm: K-start rotation and a flat partial sum for the BF16 row-major small-M GEMM

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jiaqiang-dot-liu |
| 合并 | 2026-11-17 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, jit-kernel, run-ci-amd-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/50686 |

## 一句话结论

给 gfx950(MI355X)BF16 row-major 小 M split-K GEMM 加 per-bucket K-start 旋转(并发 workgroup 错开流式读取权重 K tile)与扁平化部分和归约:Muse Spark decode 的 O 投影 + unpooler GEMM 组按服务日志权重合计 1.095x 提速(TP8/并发 28),decode 整体约 1.5% 更快。

## 动机

`kernels/ops/gemm/bf16_tiled_small_m_gemm.py` 的 row-major split-K 路径(`rowmajor_gemm` / `rowmajor_gemm_partials`)服务 Muse Spark decode/verify 步的 BF16 小 M GEMM:每个 TP8 步跑 attention O 投影 `[8192, 1024]` 与 MoE unpooler `[8192, 4096]`,约每步 1.4 ms;同一路径还服务 pooler、BF16 joint router/pooler、DFlash 草稿模型与 ROCm 上的通用非量化 linear。内核是访存受限的:64 行时 unpooler 约 18 us 流 67 MB(约 3.6 TB/s),更大的草稿形状 `[14336, 8192]` 在同一内核达约 5.1 TB/s。改动来自 Hyperloom kernelforge campaign。

## 优化点

- `ROW_MAJOR_ROT`:per-bucket 的 K-start 旋转表。长 K bucket 中每个 N block 的 K 循环从 tile `(pn * ROT) % KT_PER` 开始,并发 workgroup 流不同 K tile 而非同一条;`ROW_MAJOR` 表本身不变,无条目的 bucket 保持顺序循环。九个 bucket 入表(如 `(8192, 4096, 64): 7`、`(14336, 8192, 16): 7`);`KT_PER <= 8` 的启动与 `[8192, 7168]` 各 bucket 实测旋转会变慢,故不加入。
- `_sum_partials` 改扁平 1-D 归约:每线程持有连续元素,SK 个部分和 load 全部独立在飞;`rowmajor_gemm` 按 `_sum_block(mn)` 选 block(≥64K 元素 (1024,4) 否则 (256,1)),tiled 路径保持原启动,签名不变。
- campaign 提出但被排除的两项:单启动 split-K 带共享 arrival counter(双并发流可能竞争,且没有 Muse decode 调用它)、`(8192, 1024, 64)` 的 warps 改动(无增益,中位 1.000x)。

## 关键改动

```python
ROW_MAJOR_ROT = {
    (8192, 4096, 64): 7, (8192, 4096, 128): 5, (8192, 2048, 128): 5,
    (4096, 8192, 128): 5, (4608, 8192, 64): 3, (4608, 8192, 128): 3,
    (2560, 8192, 128): 3, (14336, 8192, 16): 7, (14336, 8192, 32): 7,
}

# _rowmajor_partial: if ROT > 0
    start = (pn * ROT) % KT_PER
    for i in range(KT_PER):
        kt = (start + i) % KT_PER
        ...

# _sum_partials: flat reduction
    acc = tl.load(P + i, mask=msk, other=0.0)
    for s in tl.static_range(1, SK):
        acc += tl.load(P + s * MN + i, mask=msk, other=0.0)
```

## 性能声明

(全部照抄 PR 正文,未独立核实;MI355X, ROCm 7.2;CUDA graph 轮换权重超出 256 MB MALL,权重从 HBM 来;当前与新区内核交错 3 轮,264 个 (shape, M) 点)

- 内核 speedup(min/median/max):MoE unpooler [8192,4096] 1.025/1.052/1.139;TP4 O 投影 [8192,2048] 0.997/1.042/1.060;MoE pooler [4096,8192] 0.999/1.052/1.082;joint router/pooler [4608,8192] 1.032/1.049/1.079;TP8 O 投影 [8192,1024] 0.997/1.000/1.004;DFlash 草稿 [14336,8192] 1.079/1.134/1.162、[2048,8192] 1.008/1.037/1.047、[2560,8192] 1.004/1.028/1.044、[8192,7168] 0.989/1.002/1.049。对照:当前内核对自身同 harness 在 0.989–1.013。
- 按 decode batch mix 加权:Muse decode GEMM 组(O 投影 + unpooler)TP8/并发 28 为 1.095x、TP4/并发 36 为 1.052x。
- 端到端(InferenceX AgentX trace replay,Muse Spark Open MXFP4 TP8/EP8,并发 28,3600 s,模拟接受长 4.05):decode 约 1.5% 更快(TPOT p50 3.88–3.90→3.84 ms,吞吐/GPU 21786/21636/21669→21965);e2e normalized interactivity 变化在该负载约 ±2% 的 run-to-run 波动内,故只声明内核级增益与 decode 提速。

## 验证与回归风险

新增 `test/registered/amd/test_bf16_rowmajor_small_m_gemm_gfx950.py`(MI35x CI,`stage-b-test-1-gpu-small-amd-mi35x`,90 s):每个 ROW_MAJOR bucket 三个行数对 FP32 参考(含 SK contract)、旋转 vs 顺序循环、tiled 路径;破坏旋转或求和会让 54/45 个 case 失败。SNR vs FP32 55.4 dB 与现内核相同;最大相对差 4.9e-3(FP32 求和顺序改变后在 BF16 舍入前)。GSM8K 5-shot 1319 题逐题对比:Muse Spark TP8/EP8 + DFlash2,基线两跑间 98.9% 提取答案一致、本 PR 与两基线各 98.6%,精度 0.9742/0.9727 → 0.9757,差异在基线自身波动内。风险面:仅 gfx95 路径可达;不改 LDS/寄存器占用;性能仅在 MI355X 实测。

## 关联

Hyperloom kernelforge campaign(AMD-AGI/Hyperloom,发现来源);`unquant.py`/`dflash.py` 在 `is_gfx95_supported()` 下导入该路径;基线代码与 main(552e16968)逐字节一致。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
