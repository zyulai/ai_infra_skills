# PR #47964: [Feat] [AMD] custom-all-reduce: Run the JIT custom all-reduce v2 on ROCm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenkaidu |
| 合并 | 2026-11-06 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | amd, run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(diff 含 ptx.cuh、custom_all_reduce.cuh、custom_all_reduce_v2.py、configs、hip_vmm_utils.py 及 2 个测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/47964 |

## 一句话结论

把 JIT custom all-reduce v2(carv2)移植到 ROCm(gfx950)并按 MI350X 调优,4-64KB all-reduce 比 AITER 快 20-37%。

## 动机

`CustomAllReduceV2`(JIT custom all-reduce,"carv2")此前只在 NVIDIA 上运行;AMD 侧的 custom all-reduce 是 AITER(或遗留 AOT 版本),因此 ROCm 用不上 carv2 的 Lamport push 算法,也没有融合 collective 所依赖的 push/pull communicator planes。push 算法对 decode 尺寸的 all-reduce 最关键:8x MI350X 上 carv2 `1shot_push` 在 4-64KB 比 AITER 快 20-37%,且 carv2 在 4MB 以内持平或胜过 AITER。

## 优化点

两个 commit:ROCm 移植,然后 MI350X launch geometry 调优。(1) 新增 `hip_vmm_utils.py` 的 `HipSymmetricBuffer`:每 rank 一次 `hipMemCreate`(uncached、POSIX-fd handle),经 `exchange_posix_fds`(SCM_RIGHTS)交换 fd 并映射进所有 peer,替代 torch symmetric memory,不需要 FABRIC handle 与 multicast;对 amdgpu/KFD 驱动 "SVM 地址已占用" 导致的 OOM 提供分块(chunked)回退;所有 rank 先对齐分配结果再交换 fd,失败即全局报错而非挂死。(2) `ptx.cuh` 在 `#ifdef USE_ROCM` 下实现 `ptx::` 词汇表:system-scope `__builtin_amdgcn_global_{load,store}_b128`、`__hip_atomic_*` flag 流量、fence,multimem 一律 trap(ROCm 无 multicast);Lamport 本地 poll 用 agent scope,并用 signal fence 防止 LLVM 把 spin 循环当纯读删掉;`2shot_pull` 在 block barrier 前每个 wave 做一次 system-scope release,否则 graph replay 会偶发读到过期 reduced shard(TP6 约 1/200)。(3) `CustomAllReduceV2` ROCm 适配:无 multicast、无 graph zero-copy 输入注册(XGMI 不保证 peer 缓存输入 buffer 一致性,graph 内经 workspace 中转),`IPCManager` 仅在可注册时构建;gfx950 报 capability 9.5,故给 ROCm 独立调优表,不落入 SM90 表;HIP runtime 低于 MIN_HIP_VERSION (7,2) 时禁用 v2 并告警。(4) MI350X 调优:`num_pull_blocks=32`(每 pull block 自带跨卡 barrier,256 个 block 大多耗在 flag 流量),`1shot_push` 从 128KB 起 1024 threads/block,push grid 保持每 CU 一个 block。

## 关键改动

```text
python/sglang/srt/utils/hip_vmm_utils.py: 新增 HipSymmetricBuffer(HIP VMM + POSIX fd)
python/sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh: ROCm ptx 词汇表
python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh:
  2shot_pull per-wave release fence;1shot_push ROCm block size
python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py:
  ROCm 分配路径、graph 支持限制、HIP 版本门控
python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py:
  新增 _rocm_configs 调优表(num_pull_blocks=32)
```

调优表(节选):

```python
num_push_blocks=num_sm,
# each pull block runs its own cross-GPU barrier: 32 blocks are 40-65%
# faster than one per CU at TP8 from 1 MB
num_pull_blocks=32,
```

## 性能声明

8x MI350X,TP8,bf16,CUDA-graph mode(每 graph 16 次 all-reduce),us/all-reduce(取各 rank 最大值,3 次运行取 median):

- 4KB:RCCL 14.5 / AITER 5.3 / carv2 1shot_push 4.2(best carv2 vs AITER -20%)
- 16KB:AITER 5.6 / 1shot_push 4.4(-23%);64KB:AITER 8.7 / 1shot_push 5.5(-37%)
- 128KB:1shot_push 7.1 vs AITER 7.3(-3%);256-512KB AITER 领先(+3%/+0%)
- 1MB:2shot_push 10.3 vs AITER 11.3(-9%);2-4MB 持平(-2%);8MB 及以上 carv2 落后约 10%
- carv2 在所有尺寸均快于 RCCL

调优 commit 效果(同 setup):TP8 下 `num_pull_blocks=32` 使 `2shot_pull` 快 39-65%、`1shot_pull`(1MB 起)快 35-51%(如 1MB 2shot_pull 46.1 → 16.4us,-65%;4MB 1shot_pull 170.0 → 83.8us,-51%);`1shot_push` 128KB 起 -7~-11%。即使调优后 pull 算法仍落后 `2shot_push`(1MB:16.4 vs 10.3us),PR 建议 ROCm 阈值后续把 1-16MB 路由到 `2shot_push`。

## 验证与回归风险

内核级位精度:8x MI350X、ROCm 7.2(HIP 7.2.26015)、镜像 `lmsysorg/sglang:v0.5.22-rocm724-mi35x`,与 RCCL 对比 `atol=0, rtol=0`。`test_custom_all_reduce.py --num-gpu 2,4,6,8` + `SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1`(4 算法 x eager/graph x fp16/bf16/fp32 x 16B-8MB)每个 world size 310 passed;capability 测试 13 passed;背靠背压力、无 host sync、随机算法/尺寸交错、graph replay、RCCL 先于 carv2 初始化(走分块 VMM 路径)均 0 mismatch(TP8;TP2/4/6 在调优 commit 前验证)。无模型级精度运行——因 ROCm dispatch 本 PR 不选 v2,模型输出不变。风险面:ROCm serving 行为不变(`dispatch_custom_all_reduce` 仍选 AITER);`test_custom_all_reduce.py` 的 graph capture 改为 `capture_error_mode="thread_local"`(ROCm 下参考 NCCL 组 watchdog 在 capture 期间查询事件会触发全局 capture 报错);CUDA 行为不变,但 `_allocate_symmetric_memory` 返回签名变更触及共享代码路径。

## 关联

明确留出 follow-up:重新调优 size thresholds、在 ROCm dispatch 中启用 v2(届时才有 serving 收益)。同分类(carv2/NVIDIA 侧)其他 PR 编号素材未列出,写"无"。
