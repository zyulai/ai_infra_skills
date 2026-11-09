# PR #37119: [AMD] CI: fix Lean decode crash on the EAGLE path

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37119](https://github.com/sgl-project/sglang/pull/37119) |
| 作者 | mqhc2020 |
| 合并日期 | 2026-09-04 |
| labels | amd, run-ci, jit-kernel, bypass-fastfail |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
AMD EAGLE 路径因未传 Lean 网格缓冲致 `locks=None` 崩溃;补传 `cuda_graph_lean_*` 缓冲并在内核侧判空回退。

## 问题现象
ROCm 上运行 spec-decode(EAGLE)时 scheduler 被崩溃杀死,报错:

```
File "python/sglang/kernels/ops/attention/decode_attention.py", line 2004, in _decode_lean_attention_fwd
    locks.zero_()
AttributeError: 'NoneType' object has no attribute 'zero_'
```

触发条件:spec-decode 捕获路径(`triton_backend.py:667`)是 decode-mode,会走到 Lean kernel,而此时 Lean 缓冲缺失。影响面:AMD CI 上 `test_adaptive_speculative.py` 与 `test_eagle_constrained_decoding.py` 两个 EAGLE 测试文件均报 0 tests run。该问题为 ROCm-only,因为 Lean dispatch 由 `_is_hip` 门控。作者在 MI355X 上用 `decode_attention_fwd` + `enable_lean=True` 且缓冲缺失复现:修复前抛 AttributeError,修复后返回有限输出。

## 根因
正文披露:#33576 向 `ForwardMetadata` 增加了四个 Lean persistent-grid buffers,但 5 个构造点中只在 2 个填充了这些缓冲;spec-decode capture path(`triton_backend.py:667`)不在其中,导致 Lean kernel 拿到的 `locks=None`,调用 `locks.zero_()` 即崩溃。

## 修复方案
两处改动(正文 Modifications):
- `triton_backend.py`:spec 捕获路径改为传递与非 spec 捕获路径相同的 `cuda_graph_lean_*` 缓冲(pack 中的 diff 未包含该文件改动,此处按正文描述)。
- `decode_attention.py`:Lean 分支选择条件增加 `lean_locks is not None`,缓冲缺失时回退标准 kernel 而不是崩溃。

```diff
         and _should_use_lean_decode(
             enable_lean, logit_cap, sinks, xai_temperature_len, score_mod
         )
+        and lean_locks is not None
```

正文说明:本次修复仅恢复原本会中断的代码路径,Lean 数值行为不变,精度测试不适用。

## 影响与回归风险
- 影响窗口:自 #33576 引入 Lean persistent-grid buffers 起的提交;仅影响 ROCm(Lean dispatch 为 `_is_hip` 门控)。
- 回归面:`decode_attention.py` 的 Lean 选择条件收紧,任何未传 `lean_locks` 的调用点会静默回退到标准 decode kernel(性能路径变化,数值不变);`triton_backend.py` 的 spec 图捕获路径开始传入 `cuda_graph_lean_*` 缓冲。

## 关联
- #33576

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
