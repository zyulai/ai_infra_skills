# PR #57229: [ROCm][Bugfix] Reduce CUDA graph divergences

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57229](https://github.com/vllm-project/vllm/pull/57229) |
| 作者 | mawong-amd |
| 合并日期 | 2026-09-17 |
| labels | bug, rocm, ready, nvidia |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
清理 ROCm 特有的、已失去存在意义的 CUDA graph 分歧:回退 #48526 的「capture 用当前流」与 #55095 的 DS V4/MRV2 强制 eager,并为 torch.compile warmup 增加「存在编译子模块」门控,为 ROCm 上 ModelRunner V2 默认启用铺路。

## 问题现象
ROCm 上存在多处与其他平台不同的 CUDA graph 特化行为,如今已无必要:DS V4 在 ROCm + MRV2 启用时被强制回退 `CudaGraphMode.NONE`,承受不必要的性能损失;graph capture profiling 偏离默认的 side-stream 路径;`compile_or_warm_up_model` 在没有任何 torch.compile 子模块时也会跑 specialization warmup。此 PR 同时作为 ROCm 上 ModelRunner V2 默认启用(#56360)的更干净基础。正文未描述最终用户可见的具体故障现象。

## 根因
PR 正文披露:

1. #48526 当初以观察到性能回归为由,把 graph capture profiling 改到当前流而非新建 side stream,其理由(side stream "shifts the KV/activation buffers' physical placement and slows every bandwidth-bound decode kernel ~20%")并不正确;真正原因是后来在 #53712 与 #55099 中修复的一个涉及长跨流等待(long cross-stream waits)的潜在运行时缺陷。回退后实测性能中性,不再有 20% 减速;
2. #55095 在 ROCm 上把 DS V4 回退到 `CUDAGraphMode.NONE`(MRV2 启用时)的依据是 #52644,但该问题在 `main` 上已不复存在;
3. specialization warmup 的门控只看编译模式,不看模型中是否存在被编译的子模块。

## 修复方案
五处改动:

1. `vllm/v1/worker/gpu_model_runner.py` `profile_cudagraph_memory`:删除 ROCm 下构造 `GraphCaptureContext(current_stream())` 的 cap_ctx 分支,恢复统一 `graph_capture(device=self.device)` side-stream 路径(回退 #48526);
2. `vllm/distributed/parallel_state.py`:`graph_capture` 移除 `graph_capture_context` 参数,总是新建 `GraphCaptureContext(torch.cuda.Stream(device=device))`;
3. `vllm/platforms/rocm.py` `apply_config_platform_defaults`:删除 gfx950 + MRV2 下 DeepseekV4 架构强制 `CUDAGraphMode.NONE` 的回退(回退 #55095 的相关部分),并移除 `tests/test_config.py` 中对应的 `test_rocm_gfx950_deepseek_v4_cudagraph_default` 测试;
4. `vllm/v1/worker/gpu_worker.py` `compile_or_warm_up_model`:torch.compile specialization warmup 增加 `and has_compiled_submodule(self.model_runner.get_model())` 门控。

关键 diff 摘录:

```diff
-        cap_ctx = (
-            GraphCaptureContext(current_stream())
-            if current_platform.is_rocm()
-            else None
-        )
         try:
             set_cudagraph_capturing_enabled(True)
-            with (
-                self._freeze_gc(),
-                graph_capture(device=self.device, graph_capture_context=cap_ctx),
-            ):
+            with self._freeze_gc(), graph_capture(device=self.device):
```

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围(相关分歧由 #48526 与 #55095 引入)。
- 回归面:主要影响 ROCm 路径——graph capture profiling 回到 side stream(所有平台的 `graph_capture` 签名统一,传入显式 context 的调用点不再支持);gfx950 上 DeepseekV4 + MRV2 默认不再强制 eager,CUDA graph 恢复默认启用,行为回到 #55095 之前。非 ROCm 平台受影响点为 `compile_or_warm_up_model` 的 warmup 门控:无 torch.compile 子模块时跳过 specialization warmup(可缩短启动时间,语义无损)。验证:DS V4 全量 gsm8k 在 V1/V2 runner 上精度 0.954-0.972;TP8、并发 1/8/32/128/256、8K/1K 下 DS V4 与 Qwen-3.5-35B 性能在噪声范围内持平;CI 无新增回归。

## 关联
- #56360(ROCm MRV2 默认启用,本 PR 为其铺垫): https://github.com/vllm-project/vllm/pull/56360
- #48526(被回退): https://github.com/vllm-project/vllm/pull/48526
- #53712(跨流等待缺陷修复之一): https://github.com/vllm-project/vllm/pull/53712
- #55099(跨流等待缺陷修复之二): https://github.com/vllm-project/vllm/pull/55099
- #55095(被回退;其依据为 issue #52644): https://github.com/vllm-project/vllm/pull/55095

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
