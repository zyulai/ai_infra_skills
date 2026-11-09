# PR #30309: [DCP][Bugfix][CI] Fix accuracy issue of DCP when using FLASH_ATTN_MLA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30309](https://github.com/vllm-project/vllm/pull/30309) |
| 作者 | FENP |
| 合并日期 | 2025-12-09 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
FLASH_ATTN_MLA 后端无条件声明支持 DCP varlen,但 PR #25049 的 MTP 支持仅在 cp_kv_cache_interleave_size=1 时成立;改为按 interleave size 动态设置开关,修复 decode-context-parallel 精度崩坏。

## 问题现象
在 DeepSeek-V2-Lite-Chat 上以 `--tensor-parallel-size 4 --decode-context-parallel-size 4 --cp-kv-cache-interleave-size 64` 部署并跑 GSM8K(1319 题、5-shot):main 分支 Accuracy 仅 0.028、Invalid responses 0.007;应用本 PR 后 Accuracy 恢复到 0.639、Invalid responses 0.002。即 DCP(解码头上下文并行)+ FLASH_ATTN_MLA + interleave size 64 组合出现严重精度问题。

## 根因
正文披露:PR #25049 为 DCP + FA3 增加了 MTP 支持,但该支持只在 `cp_kv_cache_interleave_size=1` 时成立;而 FA3(FLASH_ATTN_MLA)后端此前无条件设置 `supports_dcp_with_varlen=True`,未检查 interleave size,导致非 1 的 interleave size 下走错了 DCP varlen 路径。正文结论:FA3 backend 应检查 `cp_kv_cache_interleave_size` 并相应设置 `supports_dcp_with_varlen`。

## 修复方案
改 `vllm/v1/attention/backends/mla/flashattn_mla.py` 的构造函数:读取 `vllm_config.parallel_config.cp_kv_cache_interleave_size`,把 `supports_dcp_with_varlen` 从固定 `True` 改为 `(interleave_size == 1)`。

```diff
+        interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
         super().__init__(
             ...
-            supports_dcp_with_varlen=True,
+            supports_dcp_with_varlen=(interleave_size == 1),
         )
```
同时更新 `tests/distributed/test_context_parallel.py`:DeepSeek-V2-Lite-Chat 新增 `dcp_multipliers=[1]` 的默认 interleave 用例,并把 `cp_kv_cache_interleave_size=64` 用例固定 `attn_backend="FLASHMLA"`。

## 影响与回归风险
- 影响窗口:正文未说明(与引入该设置的 PR #25049 合入之后)。
- 回归面:所有使用 FLASH_ATTN_MLA 后端且开启 DCP 的场景:interleave size=1 时行为不变;interleave size≠1 时不再走 `supports_dcp_with_varlen` 路径,退回非 varlen 的 DCP 处理,可能影响该组合下的调度/性能,但精度恢复。非 DCP 场景不受影响。

## 关联
- [PR #25049](https://github.com/vllm-project/vllm/pull/25049)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
