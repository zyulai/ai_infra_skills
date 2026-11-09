# PR #14504: Revert "[V1][Core] Fix memory issue with logits & sampling"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#14504](https://github.com/vllm-project/vllm/pull/14504) |
| 作者 | robertgshaw2-redhat |
| 合并日期 | 2025-03-09 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Sampler / Detokenization |
| bug 类型 | regression |

## 一句话结论
上一版 logits 内存修复(#13776)破坏 LoRA 功能,本 PR 将其整体回滚以恢复原行为。

## 问题现象
正文指出 #13776 的改动破坏了 LoRA 功能,并给出复现命令:`pytest -s -x tests/lora/test_add_lora.py`。正文未描述更详细的具体现象(如报错信息或输出错误)。

## 根因
正文仅披露:#13776「broke LoRA」,未说明具体破坏机制。由 diff 可见本次是完整回滚 #13776 的三处改动:删除 `_dummy_sampler_run` 独立方法、删除 `compile_or_warm_up_model` 中的 sampler 预热与 logits 预分配、恢复 test_cumem.py 的严格内存断言;据此可推断问题出在被回滚的采样器预热/预分配路径与 LoRA 的交互上(由 diff 推断)。

## 修复方案
直接 revert #13776,共三个文件:
- `vllm/v1/worker/gpu_model_runner.py`:删除 `_dummy_sampler_run` 方法;`profile_run` 恢复内联构造 dummy `SamplingMetadata` 并调用 `self.model.sample`,清理时同时 `del hidden_states, logits, sampler_output, dummy_metadata`。
- `vllm/v1/worker/gpu_worker.py`:删除 `compile_or_warm_up_model` 中「预热 sampler 并预分配 logits 内存缓冲」的逻辑及其 OOM 错误包装。
- `tests/basic_correctness/test_cumem.py`:恢复 `assert used_bytes < 2 * GiB_bytes`,去掉此前给 V1 留的 7GiB 例外(该例外注释称 V1 的 logits 内存缓冲因 PyTorch 已知 bug 无法释放)。

关键 diff 摘录(test_cumem.py):
```diff
-    if use_v1:
-        assert used_bytes < 7 * GiB_bytes
-    else:
-        assert used_bytes < 2 * GiB_bytes
+    assert used_bytes < 2 * GiB_bytes
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:V1 的 `profile_run` 与 warm-up 路径恢复到 #13776 之前的形状,LoRA 测试路径恢复;`llm.sleep` 后的内存断言恢复 2GiB 上限,若 logits 缓冲无法释放的底层问题仍存在,V1 sleep mode 测试可能再次失败(被回滚 diff 中的注释曾提及该已知 PyTorch 问题)。

## 关联
- Reverts [vllm-project/vllm#13776](https://github.com/vllm-project/vllm/pull/13776)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
