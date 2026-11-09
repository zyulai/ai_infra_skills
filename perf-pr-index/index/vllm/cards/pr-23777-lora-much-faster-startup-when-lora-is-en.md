# PR #23777: [LoRA] Much faster startup when LoRA is enabled

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @andylolu2 |
| 合并 | 2025-08-30 |
| 分类 | Model-specific Fast Path |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/23777 |

## 一句话结论

LoRA 开启时 dummy_run 不再每次销毁重建 dummy LoRA,CUDA graph 捕获快 >20x。

## 动机

开启 LoRA 启动 server 非常慢,`--max-loras` 较大时尤甚:每次 `_dummy_run` 都会销毁并重建 dummy lora weights,而 warmup 与 CUDA graph capture 阶段要连续跑大量 dummy_run,同样的销毁/重建被反复执行,纯属浪费。

## 优化点

给 `_dummy_run` 增加 `remove_lora` 参数(默认 True,保持原行为):dummy LoRA 的销毁改为可选,使 warmup / CUDA graph capture 期间连续的 dummy_run 共享同一批 dummy loras。调用侧 `_capture_cudagraphs` 与 `compile_or_warm_up_model` 的循环均传 `remove_lora=False`,循环结束后统一调用新增的 `maybe_remove_all_loras` 清理;`maybe_setup_dummy_loras` / `maybe_dummy_run_with_lora` 沿参数链透传,context manager 仅在 `remove_lora=True` 时执行 `remove_all_adapters()`。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: _dummy_run 增加 remove_lora;_capture_cudagraphs 传 False 并统一清理
vllm/v1/worker/lora_model_runner_mixin.py: setup/dummy_run_with_lora 透传 remove_lora;新增 maybe_remove_all_loras
vllm/v1/worker/gpu_worker.py: compile_or_warm_up_model 的 warmup 循环复用 dummy LoRA
```

```python
-            self.lora_manager.remove_all_adapters()
+            if remove_lora:
+                self.lora_manager.remove_all_adapters()
```

## 性能声明

测试命令 `vllm serve meta-llama/Meta-Llama-3-70B --enable-lora --tensor-parallel-size 4 --max-loras 32`:CUDA graph capture(mixed prefill-decode, PIECEWISE,67/67)从 6:09(5.52 s/it)降至 15 s(4.38 it/s),PR 声明 **">20x faster"**。PR 未注明 GPU 型号等其他测试条件。

## 验证与回归风险

PR 未附单元测试,验证即上述 serve 命令的启动观测。风险:改动面小且默认行为不变(`remove_lora` 默认 True,真实请求路径不受影响);共享 dummy LoRA 依赖"warmup/capture 的 dummy_run 之间不留下脏 adapter 状态"这一假设,若未来有路径在 dummy_run 中修改 adapter 状态需重新审视;清理时机从"每次 run 后"移到"阶段末尾",需保证异常路径也能最终清理(素材中未讨论)。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
