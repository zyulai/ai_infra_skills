# PR #39680: [Kernel] Coalesce the KDA CuTe DSL decode state transpose: ~3x faster, bit-identical

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yekerr |
| 合并 | 2026-09-18 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | pack diffstat 缺失;diff 为 cutedsl_kda.py 内 8 处两行索引表达式替换(共 16 行) |
| 链接 | https://github.com/sgl-project/sglang/pull/39680 |

## 一句话结论

修正 KDA CuTe DSL decode kernel 状态转置的访存映射,L1 sector 数降约 8 倍,kernel 提速约 3.1 倍且输出 bit-identical。

## 动机

CuTe DSL KDA decode kernels(`kda_kernel_small_batch` / `_varlen` / `kda_kernel_large_batch` / `_varlen`)经 shared memory 做 global `(V, K) ↔ shared (K, V)` 转置,thread→element 映射以 V 为 fast-varying index(`k_load = flat_idx // TILE_V`)。但状态张量是 `(pool_size, HV, V, K)`、K 连续、V 按 `K * sizeof(fp32) = 512 B` strided:一个 warp 的 32 lanes 命中 32 个不同 32-byte sector、每个只用 4 字节——约 28 sectors/request,而理想约 4。写回侧同样如此。decode step 是 state-bandwidth-bound(每 token 读写完整 fp32 `[HV, V, K]` 状态),该模式主导 kernel 时间。

## 优化点

在 8 处转置点(每个 kernel 变体的 load 对与 write-back 对)交换两条索引表达式,让 K 成为 fast-varying index。`flat_idx` 在每个循环中恰好遍历 `[0, TILE_K * TILE_V)`,新旧表达式都是到同一 `(k, v)` tile 的双射:相同元素落进相同 shared-memory slot,`k_load < TILE_K` guard、`v_global < v.shape[3]` 边界检查、barrier、循环 trip count 与算术顺序全部不变,因此输出张量与 in-place fp32 状态更新 bit-identical。改后 warp 的 32 lanes 读写 K 方向 32 个连续 fp32 = 128 连续字节 = 4 个 sector。已知 tradeoff:`TILE_V_PADDED = 36` 下 compute loop 无 bank conflict,但新转置映射固定 v 走 32 个连续 k,引入 4-way shared-memory bank conflict(Nsight Compute 证实),端到端仍 ~3x;swizzled layout 或 128-bit 向量化留作 follow-up,保持本 PR 是纯 bit-identical 索引交换。

## 关键改动

```text
python/sglang/kernels/ops/attention/cutedsl_kda.py: 4 个 kernel 变体中 8 处 load/write 索引对交换
```

```diff
-                    k_load = flat_idx // TILE_V
-                    v_load = flat_idx % TILE_V
+                    k_load = flat_idx % TILE_K
+                    v_load = flat_idx // TILE_K
```

## 性能声明

kernel-only CUPTI 计时(bf16,`dqk = dv = 128`,`qo_len = 1`,h=96 即 Kimi-K3,unlocked SM clocks,A/B/A 协议,A-vs-A 漂移 ≤0.03%):

NVIDIA B200 PCIe,`dv = 128`,3 次取中位:

| batch | stock (us) | patched (us) | speedup |
|---|---|---|---|
| 32 | 476.5 | 152.0 | 3.14x |
| 128 | 1817.7 | 584.5 | 3.11x |
| 512 | 7171.1 | 2309.5 | 3.11x |

同卡更宽 sweep(`b ∈ {1,8,16,32,64,128,512} × dv ∈ {128,256,512}`,dense 变体)稳定 ~−67.8%。Nsight Compute(b=32,large-batch):global load sectors/request 28.39 → 3.68,L1TEX sectors 101.0 M → 12.9 M(7.8x fewer),DRAM bytes 不变,DRAM 利用率 9.54% → 29.83%。varlen 两变体仅做 bit-exactness 验证、未计时。

## 验证与回归风险

standalone bit-exactness 检查:9 个 case 覆盖全部 4 个 kernel 变体、三类输入与 `pool_idx < 0` pad 路径,对 bf16 输出与更新后 fp32 状态做 sha256 原始字节比对,全部一致,并与独立 fp32 PyTorch 参考对比。PR 注明 in-tree CI 对该 decode kernel 无注册测试(`test_kda_fused_decode.py` 走 Triton `fla.fused_recurrent` 路径;`test_kda_prefill_cutedsl.py` 只覆盖 prefill),作者提出可补 `test_cutedsl_kda.py` 作 follow-up。风险:varlen 变体缺性能数据;shared-memory bank conflict 在极端 shape 下的影响未量化。

## 关联

benchmark 脚本 #21203(`bench_cutedsl_kda_decode.py`,Triton vs CuTe DSL);GDN 侧 `test_cutedsl_gdn.py` 为同族参考测试。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
