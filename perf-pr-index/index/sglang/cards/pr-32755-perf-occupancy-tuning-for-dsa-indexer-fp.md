# PR #32755: [Perf] Occupancy tuning for DSA indexer fp8-quant Q kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Rainchar9119 |
| 合并 | 2026-08-13 |
| 分类 | Quantization(classify.py 给出,tier: major) |
| Label | quant, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 2 个文件(kernel + 新增测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/32755 |

## 一句话结论

把 DSA indexer 的 fused fp8-quant Q kernel 从 4 warp/block 调到 8 warp/block,occupancy 约 38%→86%,大 batch 纯 kernel 时间最多缩短约 22%。

## 动机

`fused_q_indexer_rope_hadamard_quant`(DSA C4 indexer 的 fp8-quant Q kernel,由 DeepSeek-V4 rope-hadamard 路径与 V3.2/GLM rope-first 路径共用)是延迟受限、occupancy 低的 kernel:基线 4-warp block(128 线程,`__launch_bounds__(128,16)`)下调度器 achieved warp occupancy 仅约 38%;NCU 显示首要 stall 是 `long_scoreboard`(warp 等待 global load),compute pipeline 利用率不足 35%。优化杠杆在调度(更多 in-flight warp 隐藏 load 延迟),而非数学路径。

## 优化点

纯 launch-config 改动,RoPE / 128 点 Hadamard / fp8-e4m3 动态量化 / weight scaling 的数学路径不动:

1. 8 warps/block(256 线程)+ `__launch_bounds__(256,16)`:kernel 增加 `kNumWarps`/`kMinBlocksPerSM` 模板参数(默认 8/16),每 block 常驻 warp 数翻倍,occupancy 约 38%→约 86%,覆盖 `long_scoreboard` stall;可用 `-DQ_BLOCK_SIZE` / `-DQ_MIN_BLOCKS_PER_SM` 编译期覆盖。
2. lane-0-only `weights_out` 写出:scale/weight 是 warp-uniform,原先每个 warp 有 31 个同地址冗余 store,现仅 lane 0 写。

每个 (token, head) 行是完全独立的 warp 工作项(无跨行状态),哪个 SM/warp/顺序执行都不改变其 128 输出 bit,故输出对两个模板配置均 bitwise-identical。作者还在早期版本尝试过 single-wave grid cap + persistent grid-stride loop,ablation 显示 CTA 改动承担几乎全部收益、persistent 路径在 B≥2048 反而慢约 3%,已删除。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh: kFusedQBlockSize 128 -> 256,注释说明
test/registered/kernels/ops/attention/test_dsv4_indexer_quant.py: 新增,双模板路径正确性测试
```

```c
// 8 warps per block: warp-per-(token, head) work-item dispatch (Q kernel).
// 256 threads lifts scheduler occupancy (~38% -> ~86%) on the fp8-quant path;
// math is unchanged, output is bitwise-identical.
constexpr uint32_t kFusedQBlockSize = 256;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / device::kWarpThreads;
```

## 性能声明

NCU 纯 kernel 时间,B200(sm_100),baseline/candidate 交错测量以抵消时钟漂移:B=128 时 5184→4400 ns(0.85),B=2048 37552→30784 ns(0.82),B=8192 141072→110512 ns(0.78),B=16384 279296→216352 ns(0.78);B≤64 为 launch-bound 保持持平(B=1 略升至 1.04)。CUDA-event wall-clock 交叉验证一致(B=256 0.95、B=2048 0.81、B=16384 0.79),非 profiler 伪影。CTA-size sweep 显示 256 线程是拐点:寄存器保持 21/thread,384/512 线程无进一步收益。未给出端到端 E2E 服务吞吐数字。

## 验证与回归风险

新增 registered 测试 `test_dsv4_indexer_quant.py`(19 passed):V4 与 V3.2 两模板路径对 torch reference 校验(B∈{1,8,64,256,512,2048}×{int32,int64}、strided weight),另对 B∈{1..16384} 全量验证与旧 kernel byte-exact(q_fp8 0 字节差异、weights_out 0 元素差异、全 finite)。CI Base/Extra 均通过。风险面小但存在:默认 block/occupancy 参数按 sm_100/B200 调优,其他架构(如非 Blackwell)若需可经 `-DQ_BLOCK_SIZE`/`-DQ_MIN_BLOCKS_PER_SM` 覆盖;该 kernel 为 V4 与 V3.2/GLM 两条 indexer 路径共用,共享改动的回归面即这两条路径。

## 关联

同区域:DSA indexer / DeepSeek-V4 attention kernel 优化(如 #26471 online c128 compress、#32755 所属 jit-kernel 系列)。PR 正文未列出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
