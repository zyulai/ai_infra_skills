# PR #63179: [ROCm] Fuse tiny-M MXFP8 quant into the block32 GEMM; guard lm_head skinny-GEMM by weight size

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2026-10-29 |
| 分类 | Quantization (tier: major) |
| Label | rocm, ready, quantization |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 6 个文件,3 个文件因篇幅省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/63179 |

## 一句话结论

MI355X 上 DSv4.x 两条 decode 快路径:tiny-M 的 MXFP8 激活量化并入 block32 GEMM(少一次 launch),lm_head skinny-GEMM 按权重大小守门,decode ITL 降 2.1%。

## 动机

DeepSeek-V4.x block32 MXFP8 线性层在 decode(M ≤ 8)时,block32 GEMM 之前要先单独跑 `_mxfp8_quant_triton_kernel` 产出 e4m3 tile + E8M0 scale,packed GEMM 立即重读——多一次 launch 与一次激活的全局往返。另一处:`wvSplitK_hf_sml` skinny GEMM 对每个 K-shard 重读一遍权重,权重远超 memory-side cache(L3/MALL)时有效带宽坍塌——TP=4 下 DSpark draft 的 vocab-parallel lm_head shard([32320, 5120] bf16,约 315 MiB,每 decode 步一次)实测只有 1.55 TB/s(213.8 µs/call),而通用 hipBLASLt 类内核同 GEMM 约 6.5 TB/s(51-73 µs)。本 PR 属 DSV4.1-Flash MI355X serving 快路径系列(与 #63180/#63181/#63182 互相独立)。

## 优化点

(1) fused quant+GEMM:新增 `_block32_packed_quant_kernel`,在 GEMM 的 K 循环内以寄存器完成逐 32 块量化(amax → biased E8M0 scale → reciprocal rescale → RN cast e4m3),与独立量化 kernel 算术逐位一致,tiling/K-panel 打包/split-K 归约与 packed kernel 逐字节相同,故输出按构造 bit-exact;仅 M ≤ 8、packed tile、连续 bf16/fp16 输入生效,其余(非连续 x、fp32 x、M>8、非 packed)走 stock 两 launch 路径且结果相同。为 quant prologue 重调了 `_TUNED_FUSED_QUANT` 的 block_m/block_n/warps/stages/waves 五轴(K 结构轴不动以保 bit-exact)。入口在 opaque custom op 内,避免编译调用方冻结 token 数判定;带 `VLLM_DEBUG_DISABLE_FUSED_MXFP8_QUANT_GEMM` 开关与一次性 HIT 日志。(2) lm_head skinny-GEMM guard:在 skinny 分派点按权重字节数判定,`1 < n ≤ 5` 且权重超过 `VLLM_SKINNY_GEMM_MAX_WEIGHT_BYTES`(默认从硬件 memory-side cache 读取,MI355X 256 MiB / MI300X 级 128 MiB,读不到回退 256 MiB)时改走通用 hipBLASLt;n=1 无病理保持 stock。该替换非 bit-exact(不同内核/累加顺序),按 bf16 GEMM 容差校验。

## 关键改动

```text
vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py:
  _block32_packed_quant_kernel、_TUNED_FUSED_QUANT、
  rocm_mxfp8_block32_gemm_quant 入口
vllm/model_executor/layers/utils.py: lmhead_skinny_guard_permits、
  _amd_l3_cache_bytes、_skinny_gemm_max_weight_bytes(文件在省略 diff 中)
vllm/envs.py: VLLM_SKINNY_GEMM_MAX_WEIGHT_BYTES
tests/kernels/quantization/test_rocm_mxfp8_linear.py、
tests/model_executor/layers/test_lmhead_skinny_gemm_guard.py: 新增测试
```

关键机制摘录(rocm_block32_gemm.py,K 循环内量化):

```python
xb = tl.reshape(x, (BLOCK_M * K_PACK, BLOCK_K // 32, 32))
amax = tl.maximum(tl.max(tl.abs(xb), axis=2), TINY)
sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
xq = tl.reshape(xb * tl.exp2(127.0 - sb)[:, :, None], ...).to(e4m3)
acc = tl.dot_scaled(xq, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
```

## 性能声明

E2E(DeepSeek-V4.1-Flash-1017,DSpark k=5,`--monokernel`,MI355X TP=4,官方 serving 命令;12 个位置平衡 A/B cycle,每臂每 cycle 新起 server,ISL 16 / OSL 2048,并发 1;统计量为每 run mean-ITL 的中位数):

- 提交基(main@717cf84342 前):ITL t>0 7.065 → 7.243 ms OFF,即 -177 µs(-2.45%),ITL t0 -188 µs;12 个 cycle 中 11 个更快;TTFT 在噪声内;
- nightly ROCm-10 镜像复测:-127 µs(-2.27%),两组区间完全不相交;
- rebase 后最新 tip:-117 µs(-2.09%),每一对 run 都更快。

微基准支持:目标 shape n=5 时 wvSplitK 313.3 µs vs torch.linear 71.2 µs(4.4×);n=6 的 [32320,7168](442 MiB)本就走 hipBLASLt `Cijk MT64x16x512`(56.2 µs,7.9 TB/s),证明 fall-through 类内核在此配置可用; skinny-GEMM 守门背后的 MI355X 测量表详见正文。精度:gsm8k 5-shot strict,ON 0.9325 / OFF 0.9242,均在 0.9196-0.9340 波动带内。

## 验证与回归风险

新增 28 个测试(21 fused-quant + 7 guard),`pytest ... 111 passed` 于 MI355X(gfx950,nightly ROCm-10)。fused-quant 测试:serving shapes 上对 stock 两 launch 路径逐位一致、对独立于被测 GEMM 的高精度 QDQ fp32 参考 rel err < 5e-3、退化块(全零/次正规/超大/负镜像)一致、非连续/fp32/M>8 回退 bit-identical、per-row-scale 分支不受影响、CUDA graph capture/replay(含有无 warm split-K counters 两态)bit-exact。guard 测试:决策表(n=1 carve-out、cap 可调)、L3 解析(level-3 DATA_CACHE)、env/硬件/回退优先级、每 site 一次性 WARNING、torch profiler 内核选择断言(含 ROCTracer 丢记录重试)。风险:fused-quant 测试为 CDNA4 门控,MI300 CI pool 上 skip(仅本地验证);guard 交换非 bit-exact,依赖 bf16 容差;与系列兄弟 PR 在 envs.py 同一锚点文本冲突(解决方式为两侧保留); Skinny 机制与 #63015/#62846 的 VGPR spill 修复正交,落地后建议复测是否可放宽 cap。

## 关联

- 系列:#63180(fused shared experts)、#63181(fused draft sampling)、#63182(draft-prepare fastpaths)
- 相关:#63015 / #62846(skinny-GEMM tile/spill 修复,正交可组合)、#59782(fused MXFP8 quant 草稿,同为融合思路但按形状 cutoff,本 PR 全形状封顶 M=8 且以 bit-exact 为契约)
- 同区域:DSv4.1 MI355X serving 快路径(本索引 PR #51473 等 ROCm DSv4 卡)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
