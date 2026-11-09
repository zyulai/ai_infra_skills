# PR #15631: [jit-kernel] Add CuTe DSL GDN Decode Kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @liz-badada |
| 合并 | 2026-01-18 |
| 分类 | CUDA Graph / Compiler |
| Label | dependencies, sgl-kernel, run-ci |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);新增 cutedsl_gdn.py 单文件约 1500 行,另有 4 个文件从 diff 中裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/15631 |

## 一句话结论

用 CuTe DSL 重写 Qwen3-Next GDN 的 fused decode kernel 替换 Triton 实现,H200 TP=2 端到端输出吞吐 +4.6%。

## 动机

GDN(Gated Delta Net,用于 Qwen3-Next 等混合线性注意力模型)的 decode 此前走 Triton fused sigmoid gating delta rule kernel,decode 是逐 token 热路径,kernel 效率直接决定 ITL。PR 未在正文详述 Triton 版的具体瓶颈,但目标是提供更快的 decode kernel;作者注明本版是 "ssm state transpose free" 版本,后续还有预转置 ssm state 一次、性能更好的版本。

## 优化点

在 `python/sglang/jit_kernel/cutedsl_gdn.py` 用 NVIDIA CuTe DSL(nvidia-cutlass-dsl >= 4.3.0)实现 fused sigmoid gating delta rule decode kernel,共四个变体:small/large batch(`SMALL_BATCH_THRESHOLD = 32` 分界)各配 normal 与 varlen 两种布局。kernel 把 recurrent 状态 h0 按 `(TILE_K=128, TILE_V=32)` 分 tile,通过 cp.async 双缓冲(`NUM_STAGES=2`)预取到 shared memory,消除 transpose;gating 的 softplus、`exp(A_log)*softplus`、sigmoid beta 与 qk L2 norm 全部融合进 kernel,标量门控由 warp 内 lane 0 计算后 `shuffle_sync` 广播;`v_new` 与状态更新 `h_new = h_old + k*v_new`、输出 `sum(h_new*q)` 用 warp 级 butterfly reduction 完成,状态原地写回 `h0_source`。backend 侧在 `hybrid_linear_attn_backend.py` 加 CuTe DSL decode 分支,由环境变量 `SGLANG_USE_CUTEDSL_GDN_DECODE=1` 显式开启,默认不改变现网行为。

## 关键改动

```text
python/sglang/jit_kernel/cutedsl_gdn.py: 新文件,CuTe DSL fused GDN decode kernel(4 个变体 + 编译缓存)
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: 增加 SGLANG_USE_CUTEDSL_GDN_DECODE=1 时的 CuTe DSL decode 分支(该文件在裁剪的 diff 中)
python/sglang/jit_kernel/tests/test_cutedsl_gdn.py: 精度与性能测试(在裁剪的 diff 中)
```

关键机制(smem 双缓冲 tile 预取 + 门控融合):

```python
TILE_K = 128; TILE_V = 32; NUM_STAGES = 2
SMALL_BATCH_THRESHOLD = 32
prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
for v_tile in range(prefetch_count):
    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
    cute.arch.cp_async_commit_group()
...
h_new = h_old + r_k_val * v_new
sData[(k_idx, v_idx, stage)] = h_new
```

## 性能声明

E2E `bench_one_batch_server`,Qwen/Qwen3-Next-80B-A3B-Instruct,TP=2,batch 128,output 1024,disable radix cache,input len 1024/2048/4096/8192:

- H200:输出吞吐 5568.82→5825.61 tok/s(input 1024),E2E output throughput speedup **4.6%–5.2%**;ITL 22.99→21.97 ms。
- B200:speedup **2.6%–3.4%**(7064.7→7301.8 tok/s @ input 1024)。
- H20:speedup **1.7%–2.5%**(4365.37→4437.69 tok/s @ input 1024)。

## 验证与回归风险

提供 `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` 精度/性能测试,以及 `bench_one_batch --correctness-test` 正确性命令。风险点:默认关闭(env 开关),开启后 CuTe DSL 依赖 `nvidia-cutlass-dsl >= 4.3.0`,版本不满足时会失败;kernel 按 tile/阈值分支,极端 batch 或 head 配置下的覆盖度依赖测试;labels 含 dependencies,CI 需要相应依赖更新。

## 关联

同区域(Qwen3-Next GDN 优化):#19775(fused GDN gating)、#21019(fuse split/reshape/cat ops)、#30169(CuTe DSL prefill state);后续作者预告了带 ssm state 预转置的更高性能版本。Co-author: @zhou9402, @HongliMi, @xutizhou。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
