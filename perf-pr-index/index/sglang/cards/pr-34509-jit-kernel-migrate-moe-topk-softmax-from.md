# PR #34509: [JIT Kernel] Migrate moe_topk_softmax from AOT to JIT

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @VoidC-minor |
| 合并 | 2026-08-16 |
| 分类 | MoE (tier: major) |
| Label | sgl-kernel, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files;diff 含 5 个文件,另有 1 个测试文件因超长被省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/34509 |

## 一句话结论

把 MoE fused softmax+top-k 路由 kernel 从 AOT 迁到 JIT:为运行架构按需编译,单 .so 体积减半,延迟与 AOT 持平(中位比值 1.0005)。

## 动机

`topk_softmax` 是 MoE 的 fused softmax + top-k 路由 kernel,此前仅 AOT 编译,并按每个发布架构各编译一次打进 sgl-kernel wheel,造成 wheel 膨胀。本 PR 是 #17865(sgl-kernel wheel 瘦身)的一部分:补上 JIT 实现,让 kernel 只为当前运行架构按需编译。

## 优化点

CUDA device 代码与 AOT 实现完全一致(改编自 vLLM v0.7.3 / TensorRT-LLM v0.7.1 的 topk_softmax),仅把 host launcher 适配到 tvm-ffi TensorView API,并把 softmax workspace 的分配移到 Python wrapper:power-of-two 且 ≤512 的 expert 数走 warp-specialized 快路径(无 scratch),其余(如 6、160、1024 experts)走两遍 softmax+top-k、写 workspace。`ops/moe/__init__.py` 为 CUDA 注册 JIT backend,ROCm(HIP)保留 AOT wheel 路径;调用点从 `get_kernel("moe.topk_softmax", KernelBackend.AOT)` 改为默认选择。体积上,JIT 单个运行时变体 .so 为 379.69 KiB,而 AOT 单架构全变体合并 .so 为 777.65 KiB(2.048x)。

## 关键改动

```text
python/sglang/kernels/jit/csrc/moe/moe_topk_softmax.cuh: 新增 header-only JIT 移植(device 代码不变)
python/sglang/kernels/ops/moe/moe_topk_softmax.py: load_jit 封装 + register_custom_op + workspace 分配
python/sglang/kernels/ops/moe/__init__.py: CUDA 注册 JIT backend,HIP 保留 AOT
test/registered/kernels/ops/moe/test_moe_topk_softmax.py: 对 torch 参考与 AOT kernel 的正确性测试
test/registered/kernels/benchmark/moe/bench_moe_topk_softmax.py: JIT vs AOT vs Torch 注册 benchmark
```

```python
is_pow2 = num_experts != 0 and (num_experts & (num_experts - 1)) == 0
needs_workspace = not is_pow2 or num_experts > 512
workspace_size = num_tokens * num_experts if needs_workspace else 1
```

## 性能声明

测试/基准环境:H100 80GB (sm_90)、CUDA 13.0、torch 2.12.0+cu130。144 个配置(6 token 数 x 6 expert 数 x 4 topk)全扫:JIT/AOT 延迟比中位 1.0005、最大 1.008,即没有任何配置比 AOT 慢超过 0.8%(device 代码相同,只有 host dispatch 不同)。相对未融合的 PyTorch 基线(`softmax` + `topk`),kernel 中位约 **4.9x faster**;如 num_tokens=8192、num_experts=512、topk=2 时 9.4780 µs vs 206.6805 µs。wheel 体积:moe_topk_softmax(3 dtypes)JIT 379.69 KiB vs AOT 777.65 KiB。

## 验证与回归风险

本地 845 例矩阵(815 passed,30 skipped 为 topk>num_experts):fp32/fp16/bf16 x experts {8,128,256,512,6,160,1024} x topk {1,2,4,8} x softcapping x correction bias x renormalize,外加单 token、topk==num_experts、零 token 边界;对 torch 参考用 tie-robust 索引校验,对 AOT kernel 要求逐位一致(rtol=0, atol=0,曾借此抓到 512 experts 处 dispatcher 不同步)。CI 经 `get_ci_test_range` 裁剪到 17 例;冷 runner 上 JIT 编译约 29s 占 CI 总时长 ~31s 的绝大部分(`est_time=40`)。风险:CUDA 路径换为首次调用时 JIT 编译(冷启动延迟);ROCm 不受影响仍走 AOT;workspace 阈值须与 .cuh dispatcher 及 AOT host entry 保持同步。

## 关联

属于 #17865(sgl-kernel wheel slimming)的一部分;kernel 源自 vLLM v0.7.3/TensorRT-LLM v0.7.1;正文未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
