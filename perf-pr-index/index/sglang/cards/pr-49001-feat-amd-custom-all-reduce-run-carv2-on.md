# PR #49001: [Feat] [AMD] custom-all-reduce: Run carv2 on MI450 with FABRIC handles and a gfx1250 table

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenkaidu |
| 合并 | 2026-11-07 |
| 分类 | Parallelism / Communication（tier: major） |
| Label | run-ci, jit-kernel |
| Diff | +177/-68, 5 files（pack diffstat 缺失，按 pack diff 文本计数） |
| 链接 | https://github.com/sgl-project/sglang/pull/49001 |

## 一句话结论

MI450 上用 FABRIC handle 共享 carv2 workspace 并配 gfx1250 专属阈值/网格表，DSV4-Flash TP8 serving 反超 RCCL +11.1%（C64）。

## 动机

#47964 让 CustomAllReduceV2（carv2）跑上 ROCm，但 workspace 用 POSIX fd 共享、launch 表在 MI350X 上调优，且 ROCm dispatch 默认留在 AITER。MI450（gfx1250）上这套表让 carv2 在 serving 输给 RCCL：DeepSeek-V4-Flash TP8 C64 落后 21%（1,628 vs 2,051–2,105 tok/s）。decode 中各 rank 到达 all-reduce 的时刻本就不齐；C64 trace 显示 96-block（每 CU 一 block）的 `1shot_push` 大部分时间在等 peer（512 KB 调用 p50 86 µs vs p10 18 µs），周边 compute kernel 慢 15–20%。

## 优化点

三个层面：(1) `HipSymmetricBuffer` 支持把每 rank 的 VMM 分配导出为 64 字节 `hipMemFabricHandle_t`、用 `all_gather_object` 交换，替代 SCM_RIGHTS 上的 POSIX fd；新 env `SGLANG_HIP_VMM_HANDLE_TYPE`（auto 默认：设备支持 `hipDeviceAttributeHandleTypeFabricSupported` 即用 FABRIC；否则 POSIX fd），handle 类型进入现有分配协议，不一致时所有 rank 一起报错。(2) ROCm dispatch：仅在显式设 `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1` 时选 carv2，默认保持 AITER，CUDA 不变。(3) gfx1250 专属表（按 `gcnArchName` 选择，MI350X 保留原表）：`1shot_push` ≤64 KB、`2shot_push` ≤8 MB、不用 pull 算法，更大的 all-reduce 回退 RCCL；push grid 64 block（非每 CU 96）；`2shot_push` ≤4 MB 用 256 线程。表在 TP8 上按 back-to-back / 晚到 rank / 晚到+GEMM（decode 模式）三种模式扫描选定；128 个 push block 在 1 MB 会挂起 GPU，故网格不超过 CU 数。

## 关键改动

```text
python/sglang/srt/utils/hip_vmm_utils.py: FABRIC handle 导出/导入与 auto 协商
python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py: _rocm_gfx1250_configs（64 block、8 MB 阈值）
python/sglang/srt/distributed/device_communicators/custom_all_reduce.py: ROCm opt-in dispatch
python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh: gfx1250 的 2shot_push 256 线程规则
```

```c
// MI450 (TP8, 64-block grid): from 512 KB the rule picks 512-1024 threads for
// 2shot_push; with ranks arriving skewed 256 is up to 40% faster up to 4 MB.
if (two_shot && is_gfx1250() && nbytes <= int64_t{4} << 20) block_size = 256u;
```

## 性能声明

all-reduce 时延（8×MI450 TP8、bf16、CUDA-graph 模式每 graph 16 次 AR，µs/次，ranks 取 max，3 次中位）：4 KB carv2 5.8 vs RCCL 9.9（−41%）；512 KB 13.8 vs 28.7（−52%）；1 MB 16.0 vs 45.3（−65%）；8 MB 62.0 vs 89.3（−31%）；16 MB 回退 RCCL 102.0（0%）。serving（DeepSeek-V4-Flash TP8，cookbook flags，`bench_serving` random 1024/1024，同容器 RCCL/carv2/carv2/RCCL 顺序）：C64 output 2,361/2,374 vs RCCL 2,120/2,142 tok/s（+11.1%），TPOT 29.2→26.2 ms；C256 4,818/4,849 vs 4,557/4,608（+5.5%），TPOT 47.5→44.7 ms；TTFT 不变（>8 MB 的 prefill AR 走 RCCL）。MI350X 表下 carv2 仅 1,628 tok/s（C64）。

## 验证与回归风险

`test_custom_all_reduce.py --num-gpu 2/4/6/8`（FABRIC，`SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1`）每个 world size 310 通过；capability 测试 14 通过、2 失败（#48554 的 zero-cluster-UUID NVLink clique 用例，本 PR 未触碰该代码）；TP8 back-to-back stress（eager + graph replay，含 RCCL 先初始化）与 4 KB–16 MB 对 RCCL bit-exact（FABRIC 与 POSIX fd、整数 bf16 输入）0 mismatch；DeepSeek-V4-Flash TP8 GSM8K 0.950（4 次 0.946–0.955）。注意：当前 main 的 DeepSeek-V4 在 gfx1250 因 #48399 的 JIT `main_q_norm_rope` 构建失败无法启动，serving 数字使用了本地修复（另 PR 处理）。风险：ROCm 默认路径不变（AITER）；阈值只在 TP8 调过；FABRIC 依赖设备属性探测与 all_gather_object 协议。

## 关联

前序 #47964（carv2 上 ROCm）；相关 #48399（JIT 构建问题）、#48554（capability 用例）；CUDA 侧 NVLS all-reduce 融合同批见 #48563。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
