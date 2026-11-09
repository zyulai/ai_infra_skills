# PR #34878: [ROCm][Test] Fix beam search determinism failures from batch-size-dependent FP divergence and removed wrong marker

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34878](https://github.com/vllm-project/vllm/pull/34878) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-02-19 |
| labels | rocm, ready, ci/build |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 beam search 测试因 batch-size 相关的浮点发散反复失败;通过平台条件化的 engine 参数与禁用 skinny GEMM 消除发散,并顺手删掉 buildkite 里错误的 `-m samplers` 标记。

## 问题现象
`tests/samplers/test_beam_search.py` 在 ROCm 上失败。测试对比不同并发级别下多次运行的输出,而并发不同会改变有效 batch size,触发不同 kernel 代码路径;logprob 上小到 1e-5 的差异即可在某个位置翻转 beam 排序,导致整条输出序列发散。

## 根因
正文披露:在 ROCm 上,Triton attention kernel 与 hipBLAS/rocBLAS GEMM kernel(含 skinny GEMM 变体)使用非结合的浮点归约,累加顺序随 M 维(即 batch size)变化——logits 矩阵中同一行, batch 里其他行数不同就可能得到略微不同的值。这对 beam search 是致命的。正文还列出发散来源:skinny GEMM(`VLLM_ROCM_USE_SKINNY_GEMM`)按 M 维选择不同 tiling/归约策略;CUDA graph padding 改变 GEMM 的有效 M 维;async scheduling 使各 scheduler step 的 batch 组成不确定。正文指出这与 #33123(prefix caching 在 ROCm 上的确定性)同类。

## 修复方案
`tests/samplers/test_beam_search.py`:
1. 新增 `EXTRA_ENGINE_KWARGS`,仅在 `current_platform.is_rocm()` 时填入 `async_scheduling=False`、`enforce_eager=True`、`enable_prefix_caching=False`、`max_num_seqs=1`,并传给各测试的 `vllm_runner`;
2. 各测试在 ROCm 上 `monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "0")` 禁用对 batch 几何敏感的 GEMM kernel。
另删掉 `.buildkite/test-amd.yaml`、`.buildkite/test_areas/samplers.yaml` 中错误标记:`pytest -v -s -m samplers` 改为 `pytest -v -s samplers`。

```diff
+EXTRA_ENGINE_KWARGS: dict = (
+    dict(
+        async_scheduling=False,
+        enforce_eager=True,
+        enable_prefix_caching=False,
+        max_num_seqs=1,
+    )
+    if current_platform.is_rocm()
+    else {}
+)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:纯测试与 CI 配置改动,不影响运行时代码。ROCm 上这些 beam search 测试以更严格的确定性设置运行(可能更慢);CUDA 上 `EXTRA_ENGINE_KWARGS` 为空 dict、不设环境变量,行为完全不变。CI 命令去掉 `-m samplers` 后按路径收集测试,收集范围可能变化。

## 关联
- [#33123](https://github.com/vllm-project/vllm/pull/33123) — Prefix caching 在 ROCm 上产生不同输出(同类 batch-size 相关 FP 发散)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
