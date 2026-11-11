# PR #47581: [Perf] [NPU] dspark: Integrate argmax and verify scheduler kernels

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AndyLi429 |
| 合并 | 2026-11-09 |
| 分类 | Speculative Decoding(tier: major) |
| Label | npu, run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(diff 摘录完整:3 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/47581 |

## 一句话结论

在 Ascend NPU 上把 DSpark verify-length 调度与 row-wise argmax 路由到 sgl-kernel-npu 专用算子,scheduler microbench 提速 1.67-1.92x。

## 动机

DSpark 的 verify-length scheduling 与 greedy verification 在 Ascend NPU 上有相当时间花在通用 op 上。本 PR 把受支持的路径改走 NPU 专用 scheduler 与 row-wise argmax 实现。依赖 sgl-kernel-npu#913 提供 launcher-cached 的 `schedule_verify_lens_topk_triton`;NPU kernel 包还需提供 `row_argmax`。

## 优化点

三处分发改动。(1) `ScheduleVerifyLensTopk`:删掉 arch35 上强制回落 torch 的分支,`triton` 路径在 NPU 上改调 sgl-kernel-npu 的 `schedule_verify_lens_topk_triton`(launcher-cached 实现),CUDA 保持 fused kernel——NPU 侧此前无法 lower 原 triton-ascend 方案(无全局 sort,O(n²)-rank 方案依赖后端不支持的 pattern),现在由 kernel 包承接。(2) `sampling argmax`:NPU 上满足 FP32、2-D、`stride(1)==1`、行数 ≤64(或 ≤384 且 vocab ≥65536)、vocab ≥4096 的输入走 `row_argmax`,其余回落 `torch.argmax`;语义上 NPU kernel 与 torch fallback 保持 torch.argmax 的 NaN 选择(CUDA JIT kernel 则从不选 NaN)。(3) NPU dispatch 测试扩展:correctness、特殊值(NaN/inf)、fallback 输入与 scheduler 边界(budget 0、空输入等)。

## 关键改动

```text
python/sglang/kernels/ops/sampling/__init__.py: argmax 在 NPU 上按形状/dtype 门控走 sgl_kernel_npu row_argmax
python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py: ScheduleVerifyLensTopk 在 NPU 上调 sgl-kernel-npu 的 schedule_verify_lens_topk_triton
test/registered/kernels/ops/speculative/test_dspark_npu_dispatch.py: NPU 分发正确性/fallback/边界测试
```

```python
if (
    logits.dim() == 2
    and logits.dtype == torch.float32
    and logits.stride(1) == 1
    and (
        logits.shape[0] <= 64
        or (logits.shape[0] <= 384 and logits.shape[1] >= 65536)
    )
    and logits.shape[1] >= 4096
):
    from sgl_kernel_npu.operators.sample import row_argmax
    return row_argmax(logits)
```

## 性能声明

Operator microbench(Ascend910_9362,FP32,非端到端):scheduler(gamma=6,budget 3×B,10 warmup、10 组×20 calls 取 median,NPU events),原 NPU Triton vs kernel PR-913 的 cached 版:B=8 226.0→135.1 µs(1.67×)、B=32 254.0→139.5(1.82×)、B=128 265.3→138.0(1.92×);另一早期 run 的 Torch NPU scheduler 为 645.7/628.3/781.4 µs(B=8/32/128,整数 sort 跑在 AiCPU),作者注明仅示意、非受控对比。row argmax(vocab 129280,NPU Graph microbench):7 行 Torch 18.3 vs NPU Triton 12.9 µs,112 行 54.8 vs 36.9 µs。E2E(8,192 in/1,024 out,conc 16/128,单次运行):compact 模式相对 origin/main,bs=4:吞吐 +4.19%、TTFT −13.72%、E2E −4.22%、TPOT −0.21%;bs=32:吞吐 +2.21%、TTFT −3.68%、E2E −2.16%、median ITL −0.96%。PR 自述:吞吐增益不能确立 decode kernel 加速(bs=4 的 E2E−TTFT 反由 6,481.91 升至 6,823.36 ms),E2E 降低多来自 TTFT;PR 内 static vs compact 的 TPOT −6.06%/ITL +6.94% 等比较不隔离本 PR 贡献。

## 验证与回归风险

Ascend910_9362(PyTorch 2.10.0、torch-npu 2.10.0.post6、Triton-Ascend 3.2.0,SGLang 58d10d27dc + sgl-kernel-npu e0893fb):focused operator tests 与 SGLang NPU dispatch tests 15/15 通过;覆盖 FP16/BF16/FP32 confidence、ties、budget/threshold 变化、noncontiguous/offset 输入、NPU Graph replay;row argmax 含 NaN/inf 用例;argmax mock 断言 kernel 被恰调用一次、fallback 输入不被调用。风险:模型级输出精度验证 pending(follow-up);E2E 为单次运行、无重复方差,且日志缺 per-rank batch/DP 配置、运行 commit 与采样/SPS 配置;row_argmax 覆盖范围有限(其余输入回落 torch.argmax);硬依赖 sgl-kernel-npu 版本。

## 关联

sgl-kernel-npu#913(依赖:launcher-cached schedule_verify_lens_topk_triton);同分类 Speculative Decoding:#42855、#48613。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
