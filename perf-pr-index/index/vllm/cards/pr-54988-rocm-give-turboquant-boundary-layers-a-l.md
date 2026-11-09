# PR #54988: [ROCm] Give turboquant boundary layers a layout-compatible backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stefankoncarevic |
| 合并 | 2026-09-24 |
| 分类 | Other（tier: major） |
| Label | rocm, ready, quantization |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/54988 |

## 一句话结论

修复 turboquant 边界层在 ROCm 上的 KV cache layout 冲突，使 MI300X 四个评测用例由启动失败恢复为通过。

## 动机

正文 Purpose 说明：由 #50519 加入的 `(MI300) LM Eval TurboQuant KV Cache` 组在 AMD CI build 12548 上四个测试全部失败，服务启动即抛 `ValueError: No KV cache layout satisfies every supported set: [['LHBNC', 'LBHNC'], ['LBNHC']].`。原因是 `turboquant_*` KV cache 从不覆盖整个模型——`TurboQuantConfig.get_boundary_skip_layers` 把 dense 模型的前后各两个 attention 层保留在 native dtype，这些层各自选 attention backend，其余层选 `TURBOQUANT`。ROCm 上边界层落到优先级列表首位的 `ROCM_ATTN`（读 `LHBNC`/`LBHNC`，K、V 分平面），而 `TURBOQUANT` 只读 `LBNHC`（K、V 打包进每 token 每 head 一个 slot）。整个 worker 只能共用一种 layout，交集为空，`get_supported_kv_cache_layouts` 遂抛错。这是启动期正确性问题，非吞吐退化。

## 优化点

把「空交集」当作该 backend 对这一 layer 无效的理由：边界层于是改用 `TRITON_ATTN`（stride-aware、不声明 layout 偏好），run 解析为 `LBNHC`。同一 filter 也作用于显式 `--attention-backend`——该路径本就容忍 `turboquant_*` run 的 mismatch（单个 backend 无法服务所有层），现在把可容忍的无效原因扩展到 layout 冲突。diff 中 `vllm/platforms/rocm.py` 新增 `_uses_turboquant()`（判断 `cache_dtype` 是否 `turboquant_` 前缀）、`_shares_layout_with_turboquant()`（比较 `supported_kv_cache_layouts()` 与 TURBOQUANT 的交集）与 `_get_invalid_reasons()`（在 validate_configuration 无其它无效理由且为 turboquant run 且不共享 layout 时追加 `_TURBOQUANT_LAYOUT_REASON`），并让 `get_valid_backends`/`get_attn_backend_cls` 走该路径；`vllm/v1/attention/selector.py` 把 `_run_kv_cache_dtype` 加入 `_cached_get_attn_backend` 的缓存键，避免按 run-wide dtype 选择时被缓存误命中。正文明确 KV cache dtype 非 `turboquant_*` 的 run 不受影响。

## 关键改动

```text
vllm/platforms/rocm.py: 新增 _uses_turboquant/_shares_layout_with_turboquant/_get_invalid_reasons，使边界层避开与 TURBOQUANT 无公共 layout 的 ROCM_ATTN
vllm/v1/attention/selector.py: 把 _run_kv_cache_dtype 纳入 backend 选择缓存键
tests/v1/attention/test_rocm_attention_backends_selection.py: 补充 ROCm backend 选择单测

# 机制摘录 (rocm.py: _shares_layout_with_turboquant)
def _shares_layout_with_turboquant(backend_class: type["AttentionBackend"]) -> bool:
    layouts = backend_class.supported_kv_cache_layouts()
    turboquant_layouts = (
        AttentionBackendEnum.TURBOQUANT.get_class().supported_kv_cache_layouts()
    )
    if layouts is None or turboquant_layouts is None:
        return True
    return not set(layouts).isdisjoint(turboquant_layouts)
```

## 性能声明

PR 正文未给出定量性能数据（无吞吐/延迟 benchmark）。Test Result 仅给出功能结果：「The group is green, 4 passed in 10m35s over all 1319 questions per config」，即原本失败的评测组变绿、4 个用例通过、耗时 10m35s、每个配置覆盖 1319 道题；这是正确性与评测耗时，非性能指标。

## 验证与回归风险

Test Plan 在单张 MI300X（gfx942）上跑失败组本身 `pytest -s -v evals/gsm8k/test_gsm8k_correctness.py --config-list-file=configs/models-turboquant.txt`，以及 `pytest -v tests/v1/attention/test_rocm_attention_backends_selection.py`。Test Result：该组 4 passed in 10m35s over all 1319 questions per config。改动集中在 ROCm platform 的 backend 选择逻辑、selector 缓存键与 ROCm selector 单测；正文明确「KV cache dtype 非 `turboquant_*` 的 run 不受影响」，显式 `--attention-backend` 的 fallback 条件被扩展为同时接受 `_KV_CACHE_DTYPE_REASON` 与 `_TURBOQUANT_LAYOUT_REASON`。正文未给出其它平台、其它 dtype 或精度路径的回归评估结论。

## 关联

PR 正文明确提及评测组由 #50519 引入、失败记录见 AMD CI build 12548；同分类其它 vLLM PR 编号未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
