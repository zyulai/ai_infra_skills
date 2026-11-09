# PR #29499: [EPLB] Optimize EPLB with numpy

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2026-01-07 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 3 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/29499 |

## 一句话结论

EPLB 原语改用 numpy 实现，sync/async EPLB 的 rearrange 由 1s370ms 降至 1s77ms，约 27% 加速。

## 动机

PR 正文说明目的是提升 sync 与 async EPLB 的性能，并修复 EPLB config 后处理中的 bug。正文列出三类适用于 sync/async 模式的优化，未进一步描述改动前的具体瓶颈数据。

## 优化点

正文列出：移除 `move_from_buffer` 中多次 gpu-cpu 传输；用 numpy 重写 `get_ep_ranks_with_expert`、`move_to_buffer`、`move_from_buffer`，平均每个原语带来 20-25% 提升；新增 `preserve_intragpu_slots` 原语，对 rebalance 算法结果做后处理，保证被调度到同一 GPU 内移动的专家留在原槽位，避免不必要的 GPU memcopy。此外新增 `log_balancedness_interval` 配置参数，用于调节 balancedness 日志频率、减少日志所需的 collectives。diff 中 `EplbModelState` 的 `is_unchanged` 由 `list[bool]` 改为 `np.ndarray`，`eplb_state.py` 引入 numpy，async 路径改用 `recv_metadata` / `RecvMetadata`。

## 关键改动

```text
vllm/distributed/eplb/rebalance_execute.py: numpy 重写搬移原语，新增 preserve_intragpu_slots 与 RecvMetadata
vllm/distributed/eplb/eplb_state.py: 引入 numpy，状态字段改为 np.ndarray，rearrange 复用预计算结果
vllm/config/parallel.py: EPLBConfig 新增 log_balancedness_interval 及 model_validator 校验
vllm/distributed/eplb/async_worker.py: transfer 改用 model_state.recv_metadata
vllm/distributed/eplb/policy/{abstract,default}.py: 策略层配套调整
```

```text
+    log_balancedness_interval: int = 1
+    @model_validator(mode="after")
+    def _validate_eplb_config(self) -> Self:
+        if self.use_async and self.policy != "default":
+            raise ValueError("Async EPLB is only supported with the default policy.")
...
+    is_unchanged: np.ndarray
```

## 性能声明

正文 Benchmark results：Server `vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --disable-log-requests --no-enable-prefix-caching -dp 8 -tp 1 --max-num-seqs 256 --enable-expert-parallel --gpu-memory-utilization 0.9 --enable-eplb --eplb-config.window_size 32 --eplb-config.step_interval 128 --eplb-config.num_redundant_experts 128 --eplb-config.use_async false --max-model-len 4096`，比较 sync EPLB 并 profile 第三次 EPLB 调用，"Main: eplb_state.py: rearrange takes 1s370ms / PR: eplb_state.py: rearrange takes 1s77ms. ~27% speedup"。正文另称 numpy 重写 "gives 20-25% on average for each primitive"。Validation 用 lm_eval gsm8k 在 Qwen3-30B-A3B-Instruct-2507-FP8 DP=8、DeepSeek-V2-Lite DP=4（AsyncEPLB True/False）下给出精度结果。PR 未注明硬件型号。

## 验证与回归风险

Test Plan 写明新增 grouped layer weight exchange 测试与 `preserve_intragpu_slots` 测试；Validation 给出多组 lm_eval gsm8k 精度对比以确认精度无明显回退。正文声明优化同时适用于 sync 与 async 模式，并修复 config 后处理 bug；新增校验要求 async 只能搭配 default policy、`log_balancedness_interval` 必须大于 0。正文未说明对其他模型或硬件的回归风险。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
