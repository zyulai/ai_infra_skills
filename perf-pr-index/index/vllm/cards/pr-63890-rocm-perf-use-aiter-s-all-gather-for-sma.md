# PR #63890: [ROCm][Perf] Use AITER's all-gather for small TP all-gathers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @okorzh-amd |
| 合并 | 2026-11-06 |
| 分类 | Speculative Decoding |
| Label | performance, rocm, ready, nvidia |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 `cuda_communicator.py`、`aiter_custom_all_reduce.py`、测试与 comm benchmark 脚本 |
| 链接 | https://github.com/vllm-project/vllm/pull/63890 |

## 一句话结论

gfx950/gfx942 上把小尺寸 TP all-gather 从 RCCL 切到 AITER IPC 实现,DeepSeek-V4.1-Flash decode step 快 0.8-1.9%,微基准最高 3.3x。

## 动机

TP all-gather 一直走 RCCL;但 TP 组若已在跑 AITER custom all-reduce,说明 IPC 基础设施已就绪,小消息的 all-gather 用 AITER 的 unregistered 路径可省去 RCCL 的额外开销(dim>0 还要一次 movedim 拷贝)。此前 #48247 只在 DP 组启用了 AITER all-gather,本 PR 把它扩展到 TP 组;#48446 的 RCCL 路径保留为 fallback。

## 优化点

`CudaCommunicator.all_gather` 在 ROCm 分支最先尝试新加的 `_maybe_aiter_all_gather`:门控条件为 gfx950/gfx942(`get_cdna_version() in (3, 4)`)、TP ∈ {2,4,8}、dtype 为 fp32/fp16/bf16、非空、contiguous、且通过 AITER 自身 `should_custom_ag` 上限(vLLM 128 MiB buffer / (2×TP) = 32/16/8 MiB per rank)。dim>0 时要求 `row × elemsize % 16 == 0`,view 成 2D 后调用 `aiter_comm.all_gather_unreg`(copy-in 到可 IPC 导出的 staged buffer),输出 reshape 回原形状;不满足任一条件则原路返回 RCCL。gfx1250 暂不上(gated off,未测);`VLLM_ROCM_USE_AITER_CUSTOM_AR=0`、非 TP 组、TP6、gate reject 均维持原路径,无新 flag。

## 关键改动

```text
vllm/distributed/device_communicators/cuda_communicator.py: TP all-gather 门控 + AITER 快路径
vllm/distributed/device_communicators/aiter_custom_all_reduce.py: 暴露 all_gather_unreg
tests/distributed/test_comm_ops.py: 新增 test_rocm_tp_all_gather_uses_aiter(9 参数化用例)
benchmarks/kernels/benchmark_device_communicators.py: 增加 AITER vs RCCL 对比与图捕获校验
```

```python
+    def _maybe_aiter_all_gather(self, input_, dim) -> torch.Tensor | None:
+        aiter_comm = self.aiter_ar_comm
+        if (
+            not self.use_aiter_tp_all_gather
+            or aiter_comm is None
+            or input_.dtype not in (torch.bfloat16, torch.float16, torch.float32)
+            or input_.numel() == 0
+            or not input_.is_contiguous()
+            or not aiter_comm.should_custom_ag(input_)
+        ):
+            return None
```

## 性能声明

MI355X(gfx950),DeepSeek-V4.1-Flash,DSpark k=6、合成 AL 3.87、random 32k in / 1k out,position-swapped 对照:

- TP4 `--monokernel` c1/c2/c4:ITL p50 step 分别 -106/-120/-102 us(-1.94%/-1.79%/-1.19%);c1 复跑 -113 us(-2.07%)。
- TP4 c8/c16/c32(logits gather 3.1-14.5 MB/rank 进入 AITER):对 base -185/-350/-133 us(-1.41%/-2.11%/-0.65%),其中 c16/c32 作者判定未确立(轮间漂移 ~200 us)。
- TP8(无 monokernel)c1/c2/c4:-0.81%/-1.11%/-1.24%。Mean TPOT 随 step 变化:对 base c1 为 -2.15%。
- gfx942/MI325X(评审者 @frida-andersson 复测):TP4 monokernel c1 ITL p50 7.15 → 7.01 ms(-140 us)。
- AITER vs RCCL 微基准(768 时点 × 2 轮,1 KiB-16 MiB,bf16/fp32,eager+graph):两轮 AITER 均不输;graph 模式 dim0 中位数 TP2/4/8 为 1.18/1.41/1.43x,eager 为 2.65-2.95x;eager dim>0 最高 3.29x。
- 精度:bit-exact(torch.equal);GSM8K-1319 6 组 base-vs-PR 均值两侧均 0.916;AL 无一致方向。

## 验证与回归风险

新增 CPU mock 测试(含 gate reject、空张量、int32、非 16B 行、转置视图);GPU 侧 gate matrix 79 用例 × 5 轮(TP2/4/8 ± expandable segments)0 失败且 contiguous 用例 bit-exact;新 GPU 测试在仅替换 communicator 文件的 scratch 镜像上按预期失败(revcomm);0 sync 错误、2048 请求 0 失败。风险面:copy-in 设计是刻意选择——AITER 注册捕获路径在 expandable segments 下会报 "cannot be IPC-exported"(`mut_reg` 实验),但 copy-in 每步多花约 25 us(c1);gfx1250、gfx942 的 sweep/c2+、TP2 20-32 MiB 区间未测。

## 关联

扩展 #48247(DP-only AITER all-gather)到 TP;#48446 的 RCCL 路径保留为 fallback。同批 DSv4.1 ROCm 相关:#64101(mono decode TP8);文中列为 follow-up 组合的 #63885/#63888/#63891/#63892/#63893。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
