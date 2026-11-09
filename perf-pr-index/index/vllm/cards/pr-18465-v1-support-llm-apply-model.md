# PR #18465: [V1] Support `LLM.apply_model`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @DarkLight1337 |
| 合并 | 2025-09-20 |
| 分类 | Other \| tier: major |
| Label | performance, frontend, ready, v1, multi-modality, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/18465 |

## 一句话结论

为 V1 的 `LLM` 补上 `apply_model`，使一批依赖它的测试能在 V1 下运行。

## 动机

正文仅一句 "This enables a bunch of tests to be run in V1"。从 diff 可见此前 `tests/conftest.py` 的 `VllmRunner.apply_model` 依赖 V0 内部（`llm_engine.model_executor.apply_model`），或在 V1 下需要 `VLLM_ENABLE_V1_MULTIPROCESSING=0` / `VLLM_ALLOW_INSECURE_SERIALIZATION=1` 再走 `collective_rpc`；`tests/kernels/moe/test_mxfp4_moe.py` 里 `check_model` 曾被整体注释，注明 "TODO: llm.apply_model(check_model) currently relies on V0 internals"。

## 优化点

在 V1 的 `LLM` 上实现 `apply_model`，把函数经 `collective_rpc` pickle 到 worker 执行——新增的 `enable_pickle` fixture 注释 "`LLM.apply_model` requires pickling a function" 并设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。`tests/conftest.py` 的 helper 简化为 `return self.llm.apply_model(func)`，并恢复 `test_mxfp4_moe.py` 里被注释的 `check_model` 验证，多个 quantization / model 测试随之在 V1 下启用。从 diff 看 `test_mxfp4_moe.py` 还把 `QUARK_MXFP4_AVAILABLE` 的判断改用 `find_spec("quark")` 并导入 `QuarkLinearMethod`、`QuarkW4A4MXFP4`、`QuarkW4A4MXFp4MoEMethod`，新增的 `enable_pickle` 是 `scope="function", autouse=True` 的 fixture；恢复的 `check_model` 会取 `model.model.layers[0]` 检查量化后的层。此前 V0 路径通过 `model_executor.apply_model`，V1 路径依赖 `collective_rpc` 与函数序列化，两条分支在 helper 中被这一行统一替换掉。

## 关键改动

```text
vllm/entrypoints/llm.py: V1 的 LLM.apply_model 实现
vllm/v1/engine/llm_engine.py: V1 引擎侧 apply_model / collective_rpc 通路
vllm/engine/llm_engine.py、vllm/executor/executor_base.py、vllm/worker/worker_base.py: 跨 V0/V1 引擎与 executor 的统一入口
tests/conftest.py: VllmRunner.apply_model 简化为 self.llm.apply_model(func)
tests/kernels/moe/test_mxfp4_moe.py: 增加 enable_pickle fixture 并恢复 check_model
```

helper 简化片段（摘自 tests/conftest.py 的 diff）：

```diff
     def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
-        if hasattr(self.llm.llm_engine, "model_executor"):
-            # This works either in V0 or in V1 with
-            # VLLM_ENABLE_V1_MULTIPROCESSING=0
-            executor = self.llm.llm_engine.model_executor
-            return executor.apply_model(func)
-        # This works in V1 with VLLM_ALLOW_INSECURE_SERIALIZATION=1
-        def _apply_model(self):
-            return func(self.get_model())
-        return self.llm.llm_engine.collective_rpc(_apply_model)
+        return self.llm.apply_model(func)
```

## 性能声明

PR 正文未给出定量性能数据：label 含 performance，但正文只有一句目的说明，没有 benchmark 数字，也没有模型 / 硬件 / 精度 / workload 等测试条件。

## 验证与回归风险

正文未给出 Test Plan / Test Result。从 diff 看，通过恢复此前被注释或依赖 V0 的测试来验证：`test_mxfp4_moe.py`、`test_qwen2_vl.py`、`test_awq.py`、`test_compressed_tensors.py`、`test_fp8.py`、`test_gptq_dynamic.py`、`test_lm_head.py`、`test_modelopt.py`、`test_ptpc_fp8.py`、`test_quark.py`、`test_register_quantization_config.py`。回归风险：`apply_model` 需要 pickle 函数（依赖 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`），涉及 V0 / V1 引擎路径差异；PR 未注明其他风险。正文没有给出任何测试结果数字，验证方式完全依赖上述测试文件的恢复：这些用例原先在 V1 下因缺少 `apply_model` 被跳过或注释，改由 `self.llm.apply_model(func)` 统一后即可在 V1 引擎下直接检查模型权重与量化方法；`enable_pickle` 以 autouse 方式对整个模块生效，说明序列化开关是该通路的前提条件。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
