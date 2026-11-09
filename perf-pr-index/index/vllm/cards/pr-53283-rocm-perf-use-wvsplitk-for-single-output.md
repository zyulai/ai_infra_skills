# PR #53283: [ROCm][Perf] Use wvSplitK for single-output GEMMs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tangzzycc |
| 合并 | 2026-09-23 |
| 分类 | MoE (tier: major) |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动为 1 个 dispatch 文件 + 1 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/53283 |

## 一句话结论

放宽 ROCm wvSplitK 分发条件到 `m == 1`,让 batch-1 decode 的单输出 GEMM(如 Qwen shared-expert gate)不再落入 F.linear,算子快 37.8x。

## 动机

ROCm 上非量化 GEMM 的 `wvSplitK` 分支此前只接受 `m > 8`,导致 `weight.shape[0] == 1` 的形状(单输出行)全部落入通用 `F.linear` fallback。这个形状正是 Qwen 系列 shared-expert gate 在 batch-1 decode 时产生的。而现有 `wvSplitK` kernel 本身已支持 `M_in=1`,只是 Python 侧分发条件将其挡住(issue #52631)。

## 优化点

改动极小但命中热点:`rocm_unquantized_gemm_impl` 的分发条件从 `m > 8 and 0 < n <= 5` 放宽为 `(m == 1 or m > 8) and 0 < n <= 5`,使 m==1 且 n∈[1,5] 的 skinny GEMM 走 `ops.wvSplitK`(以 `num_compute_units()` 为 cu_count),kernel 内部以 `YTILE=1` 特化单输出情形;其余条件(如 wvSplitK 仅支持 n<=5,见先行 PR #40687)不变,超界仍回退 `F.linear`。

## 关键改动

```text
vllm/model_executor/layers/utils.py: rocm_unquantized_gemm_impl 分发条件加入 m == 1
tests/model_executor/layers/test_rocm_unquantized_gemm.py: wvSplitK 路径与 fallback 测试参数化 m∈{1,128},新增单输出真实 kernel 测试(dtype∈{fp16,bf16} x n∈[1,5])
```

关键分发代码:

```python
-        if m > 8 and 0 < n <= 5:
+        if (m == 1 or m > 8) and 0 < n <= 5:
             cu_count = num_compute_units()
             out = ops.wvSplitK(weight, x_view, cu_count, bias)
```

## 性能声明

测试平台:AMD Radeon 8060S(`gfx1151`),e2e A/B 用 Qwen/Qwen3.5-35B-A3B-GPTQ-Int4:

- BF16 `[1, 2048]` 单输出 GEMM 算子延迟:119.484 us → 3.161 us(37.8x faster),profiler 确认命中 `YTILE=1` 的 wvSplitK 特化
- 纯 decode 吞吐:23.74 tok/s → 26.77 tok/s(+12.79%)
- 每 token decode 延迟:42.125 ms → 37.350 ms(-11.34%)

Greedy 输出 token 在前后路径间完全一致。PR 未注明基准的具体 batch/负载构成细节(仅说明为 pure decode 对比)。

## 验证与回归风险

focused 测试套件 21 passed:`tests/model_executor/layers/test_rocm_unquantized_gemm.py`,其中新增的真实 kernel 测试覆盖 fp16/bf16 两种 dtype 与 n=1..5 的单输出形状,并断言与 `F.linear` 参考在 atol/rtol=1e-2 内一致;原 wvSplitK 路径与 n>5 fallback 测试均参数化加入 m=1。回归风险面有限:只影响 ROCm 上 `m==1` 且 `weight.shape[0]∈[1,5]` 的非量化 GEMM 分发;`m > 8` 与 n>5 fallback 行为不变。 注意该 PR 标签为 MoE 分类(因 shared-expert gate 形状),实际为 layer 分发层优化,适用于所有命中该形状的 ROCm 模型。

## 关联

同区域:wvSplitK skinny GEMM 支持 n∈[1,5] 的先行 PR #40687;ROCm wvSplitK 引入脉络见 pr-34709(Enable wvSplitK skinny GEMM kernel);修复的原始 issue 为 #52631。同分类 MoE 其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
