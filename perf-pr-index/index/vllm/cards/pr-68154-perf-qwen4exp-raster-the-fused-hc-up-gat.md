# PR #68154: [Perf][Qwen4Exp] Raster the fused HC up+gate mix tiles N-fastest

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-11-18 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | qwen |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/68154 |

## 一句话结论

把 #64898 的 fused Qwen4Exp HC up projection + gate mix kernel 的输出 tile 遍历顺序从 M-fastest 改为 N-fastest、去掉 TMA descriptor 的 256B L2 promotion,并把单 CTA / CTA-pair 切换点从 1024 移到 1600 tokens——GB300 上 M=16384 时 1.13x(98.4 → 86.8 us)、32768 时 1.16x,输出与 main 逐位一致,无 M 变慢。

## 动机

一个 tile 每流只读每行 `xn` 的 128 B,而 main 的 `xn` tensor map 用 256B L2 promotion,每次 fetch 都把相邻 128 B(属于下一列 block)一起带入:M-fastest 时下一个列 block 要等所有 row block 跑完才执行,届时那些 line 已被驱逐(`xn` load 是 `EVICT_FIRST`),被 promoted 的一半要从 DRAM 再取;N-fastest 时相邻列 block 并发执行,promoted 的一半被利用,而无 promotion 时每个 tile 恰好取自己要的字节。`w_up`(6.6 MB,当初选 M-fastest 的原因)在两种顺序下都留在 L2。

## 优化点

- Tile 调度:`m0 = (t / Cfg::NUM_N_BLOCKS) * BLOCK_M * kCtas + rank * BLOCK_M`、`c0 = (t % Cfg::NUM_N_BLOCKS) * BC`(原为 `t % num_m_blocks` / `t / num_m_blocks`),scheduler 与主循环三处一致改为 N-fastest。
- `make_tensor_map` 的 L2 promotion 从 `CU_TENSOR_MAP_L2_PROMOTION_L2_256B` 改为 `CU_TENSOR_MAP_L2_PROMOTION_NONE`(四个 map 共用同一 helper;配对 A/B 表明只有 `xn` 的 promotion 有影响,`lowrank`/`w_up`/输出 map 在噪声内持平,四个全 NONE 与只改 `xn` 相同)。
- `kLargeM` 由 1024 改为 1600:crossover 落在 1537..1664(152 SM 与 cap 到 148 SM 的 B200/B300-like 网格上都如此),区间内任一点距更快配置差 ~1.5% 以内,1536 或 2048 会损失最多 5%,1600 居中。
- 尝试未保留(16384 行):grouped raster G=1-16(平)、TMA L2 prefetch 下 1/2/4 个 xn tile(慢 3-13%)、寄存器输出替代 TMA store(慢 10%)、2 xn stages(GEMM 饿死,100-129 us)、全部 108 种 L2 cache-hint 组合(无一超过 main)。

## 关键改动

```diff
 // csrc/qwen4_exp/hc_up_gate_mix.cu
-// Measured crossover between the two configurations on B300.
-constexpr int64_t kLargeM = 1024;
+// Measured crossover between the two configurations on GB300 (148-152 SMs).
+constexpr int64_t kLargeM = 1600;
```

```diff
-      const int32_t m0 =
-          (t % num_m_blocks) * BLOCK_M * kCtas + rank * BLOCK_M;
-      const int32_t c0 = (t / num_m_blocks) * BC;
+      const int32_t m0 =
+          (t / Cfg::NUM_N_BLOCKS) * BLOCK_M * kCtas + rank * BLOCK_M;
+      const int32_t c0 = (t % Cfg::NUM_N_BLOCKS) * BC;
```

```diff
-      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
+      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

## 性能声明

- Kernel 时间(单个 GB300 `sm_103` 152 SM;CUPTI、CUDA graph、冷 L2;4 进程/GPU 不同 seed 的中位数,括号内为范围;main 为 out-of-tree 构建的同一 main kernel 源,与 main 内建 op 逐位相同且差 ≤0.2%):512 10.9 → 10.1 us(1.09x)、1100 14.8 → 13.7(1.08x)、1599 17.6 → 17.3(1.02x)、1600 17.7 → 17.4(1.02x)、2049 20.1 → 19.9(1.01x)、6000 40.8 → 38.5(1.06x)、12000 73.1 → 65.5(1.12x)、16384 98.4(98.3-98.5) → 86.8(86.4-87.2) us(1.13x)、32768 192.1 → 166.3 us(1.16x);512 至 12000 之间 1.01-1.12x,无更慢的 M。
- M=16384 时达到可实现 floor 的 63% → 72%(floor 62.4 us = 436.5 MB @ 7.0 TB/s 实测 on-SM copy;compute floor 54.8 us,按 cuBLAS bf16 rate)。
- 顺序 × promotion 消融(16384,括号 32768):M-fastest + 256B promotion 97.8 us(193.0)、M-fastest + 无 promotion 90.6(172.2)、N-fastest + 256B promotion 88.4(172.0)、N-fastest + 无 promotion 87.0(167.4)——关闭 promotion 几乎全部收回 M-fastest 的损失(32768 处全部收回),N-fastest 在两列都更快。
- E2E:无 A/B(输出逐位一致且只改一个 kernel 的 tile 顺序与 fetch size),正文估计 E2E 效果即 kernel 节省:96 calls x ~11.6 us ≈ 1.1 ms per 16k-token prefill step(约 0.4%),低于 serving 噪声(#64898: main vs main 至 0.8%);`xn`(336 MB)远超 L2,冷 L2 kernel 时间有代表性。

## 验证与回归风险

- `pytest tests/models/qwen4_exp/test_hc_ops.py`:19 passed(GB300),新增 M=1599/1600 覆盖新切换点;全部 20 个 M 输出逐位相等,对 fp32 参考最大 0.51 bf16 ULP。
- Dispatch 门槛(≥512 tokens、SM100 family、`VLLM_BATCH_INVARIANT`、LoRA)不变;阈值扫过 148 与 152 SM 两种网格(1408-2048 表)。
- 风险:切换点常数按 GB300(148-152 SM)实测,B300 上的原 1024 注释被替换;其余 SM 数未测;E2E 未做 A/B,由逐位一致与 kernel 节省推得(正文明确说明这是推算而非测量)。

## 关联

- 作用于 #64898 引入的 fused HC up+gate mix kernel;与 #66791 的测试矩阵共享 1599/1600 边界。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
