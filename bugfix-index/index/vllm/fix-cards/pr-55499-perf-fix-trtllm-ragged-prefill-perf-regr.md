# PR #55499: [Perf] Fix TRTLLM ragged prefill perf regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55499](https://github.com/vllm-project/vllm/pull/55499) |
| 作者 | wzhao18 |
| 合并日期 | 2026-09-09 |
| labels | ready, ci/build, nvidia |
| 热度 | 👍 1 · 💬 13 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
TRTLLM ragged prefill 每次 kernel 调用内做检查触发 CUDA 同步造成回归;把检查上移到 metadata 准备,每 forward 一次。

## 问题现象
flashinfer TRTLLM ragged prefill 存在性能回归([flashinfer#4928](https://github.com/flashinfer-ai/flashinfer/issues/4928))。实测(B200):ISL=8k/OSL=128 吞吐,conc=32 时 main 31,872 → 本 PR 32,724(+2.7%);agentic trace replay 吞吐/GPU 相对 no-check 基线的 geomean:main(GPU check)−3.24%、#55408 −0.55%、本 PR −0.04%;P90 ITL conc=32 从 65.64 ms 降到 54.19 ms。

## 根因
正文披露:检查逻辑位于 kernel 调用内部,每次调用都会触发 CUDA synchronization,造成回归。修复思路即把检查跳过、上移到 MLA Prefill metadata 准备阶段,每个 forward 只做一次,而不是每个 kernel 调用都做。

## 修复方案
- `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`:`prepare_metadata` 校验 CPU query lens 并计算 `_has_active_rows`(存在 0 长与活跃行混合时直接抛错);无活跃行时跳过 kernel、`out.zero_()` 并返回 lse=−inf;两处 `trtllm_ragged_attention_deepseek` 调用传 `skip_all_rows_active_check=True` 跳过 kernel 内检查;
- `mla_attention.py`:`ContextChunk` 新增 `all_rows_active` 字段(按 query_lens/seq_lens 全部 >0 计算),`build` 传入 `query_lens_cpu`;context chunk 遇非活跃行抛错;
- 依赖升级 flashinfer 0.6.18 → 0.6.18.post1(docker/Dockerfile、versions.json、requirements/cuda.txt);新增 zero-context 请求用例 `[4, 0, 2]`。

```python
ret = trtllm_ragged_attention_deepseek(
    ...,
    is_causal=True,
    return_lse=return_softmax_lse,
    out=out,
    skip_all_rows_active_check=True,
)
```

## 影响与回归风险
- 影响窗口:正文未写明版本范围(回归即当前 GPU-check 实现)。
- 回归面:TRTLLM ragged prefill 后端与 MLA context chunk 元数据路径;检查从 kernel 内上移后,原先依赖 kernel 内检查兜底的调用方现在靠 metadata 阶段校验与 `all_rows_active` 前置抛错,混合 0 长行会直接报 `ValueError` 而非内核内拦截;flashinfer 版本 pin 变更影响所有 CUDA 构建环境。Test Plan 仅"B200 LM Eval PCP in CI",Test Result 一栏为空;正文未附单测结果。

## 关联
- [flashinfer-ai/flashinfer#4928](https://github.com/flashinfer-ai/flashinfer/issues/4928)(回归描述)
- [#55408](https://github.com/vllm-project/vllm/pull/55408)(对照方案 CPU lens)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
