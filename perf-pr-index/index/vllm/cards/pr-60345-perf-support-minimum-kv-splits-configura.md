# PR #60345: [Perf] Support minimum KV splits configuration for TokenSpeed MLA decode 

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-10-09 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/60345 |

## 一句话结论

把 TokenSpeed MLA decode 的 `min_split_kv` 暴露为 attention-config，Kimi K3 nvfp4 E2E 提速 1.64x。

## 动机

正文 Overview 指出，TokenSpeed MLA decode kernel 支持可选参数 `min_split_kv` 来强制一个 KV splits 下限，在某些 workload 下可加速最多 4x（见 lightseekorg/tokenspeed#1693），但 vLLM 侧此前没有配置入口。此 PR 通过既有 `AttentionConfig` 把该 split floor 暴露出来：`--attention-config '{"tokenspeed_mla_min_split_kv": 8}'`，默认值 `1` 保留自动调度，取值校验在 `[1, 256]`，且只作用于 TokenSpeed MLA decode。

## 优化点

(1) 在 `AttentionConfig` 新增 `tokenspeed_mla_min_split_kv: int = Field(default=1, ge=1, le=256)`。(2) 后端在初始化时读取一次，用于 kernel dispatch 与 workspace sizing，并把 `min_split_kv=self._min_split_kv` 直接传给 decode kernel；正文说明拟议的 env var 被移除，改动沿用现有 attention configuration API。(3) `_get_workspace` 重写：当 `min_split_kv > 1` 时按最大 batch 预留 reduction scratch（`max_decode_tokens * max(num_heads, 128) * min_split_kv * (kv_lora_rank + 1) * 4`），改用 `current_workspace_manager().get_simultaneous()` 取代模块级 `_g_workspace` 字典，并记录 GiB 日志。(4) 正文说明 pin 的 `tokenspeed-mla==0.2.15` 已支持该关键字（含默认值 `1`）；提高 floor 可能增加 workspace 内存。

## 关键改动

```text
vllm/config/attention.py: 新增 tokenspeed_mla_min_split_kv 字段与 [1,256] 边界
vllm/v1/attention/backends/mla/tokenspeed_mla.py: _get_workspace 按 floor 预留 workspace，min_split_kv 直传 decode kernel
tests/engine/test_arg_utils.py: CLI / attention-config 解析覆盖
tests/v1/attention/test_mla_backends.py: decode workspace 接口与 min_split_kv 转发用例
```

关键片段（新增配置项）：

```diff
+    tokenspeed_mla_min_split_kv: int = Field(default=1, ge=1, le=256)
+    """Minimum KV splits for TokenSpeed MLA decode. The default of 1 preserves
+    automatic scheduling. Larger values request a split floor and may increase
+    workspace memory. Only applies to the TokenSpeed MLA backend."""
```

## 性能声明

正文称 `min_split_kv`「can accelerate performance by up to 4x speedup under certain workloads」（引用 lightseekorg/tokenspeed/pull/1693）。E2E agentic benchmark：Kimi K3 nvfp4、2xDCP8 + 1xDEP16、c320，**1.64x e2e speedup**；baseline `Throughput per GPU: 11942 tok/s`，`decode with min_split=8: 19678 tok/s`。Accuracy Eval：Kimi K3 nvfp4 DCP8 + DEP16、GSM8k、1319 questions、5-shot、max tokens 256、temperature 0.0、seed 42，Accuracy 0.935、Invalid 0.001、Total latency 45.735 s、Questions per second 28.840、Output tokens per second 3064.495。**重要条件**：正文明确「CUDA/DCP kernel tests and model evaluations were not rerun on this macOS host」，上述 benchmark 与 accuracy 结果是「prior validation of the same kernel setting, before the configuration-interface change」，即改配置接口之前对同一 kernel 设置的既有验证。

## 验证与回归风险

`pre-commit run --files`（含 `vllm/config/attention.py`、`vllm/envs.py`、`tokenspeed_mla.py`、两个测试文件）通过；`mypy-3.12 --hook-stage manual` 通过；隔离源码验证 `/private/tmp/pr60345_validate.py` 覆盖了默认值/边界校验、非法值拒绝、hash 区分、JSON/dotted 参数解析，并用真实 forward 方法跑通 12 个既有 CPU decode contract 用例（含 `min_split_kv` 在 1 与 8 的直接转发），macOS 上隔离了 GPU backend import。新增 CLI suite 覆盖并更新 decode 测试；完整 pytest collection 因 `tests/conftest.py` 缺 `tblib` 依赖而中止。风险：提高 floor 会增加 workspace 内存；只作用于 TokenSpeed MLA decode；依赖 pin 的 `tokenspeed-mla==0.2.15`；本次在 macOS 上未重跑 CUDA/DCP kernel 测试与模型评估。

## 关联

正文引用上游 lightseekorg/tokenspeed#1693（`min_split_kv` 内核能力的来源，为外部仓库 PR）；无同分类其他 vLLM PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
