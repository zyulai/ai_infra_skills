# PR #20057: [BugFix] Fix full-cuda-graph illegal memory access in FA3

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20057](https://github.com/vllm-project/vllm/pull/20057) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-06-25 |
| labels | ready, v1 |
| 热度 | 👍 4 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
FA3 + full CUDA graph 偶发 illegal memory access;full cuda graph 下改用非 AOT 调度并移除 scheduler_metadata 缓冲拷贝逻辑。

## 问题现象
使用 FlashAttention 后端且开启 `compilation_config:{"full_cuda_graph":true}` 时,偶发(sporadic)illegal memory access,由 @WoosukKwon 发现。属难以稳定复现的崩溃类问题:作者复现尝试 1.5 小时未能复现,验证方式为 lm-eval gsm8k(Qwen/Qwen2.5-VL-72B-Instruct,TP8)full_cuda_graph true/false 两组对照跑通。

## 根因
正文未披露根因,仅说明是偶发 illegal memory access。(由 diff 推断)diff 中新增注释解释了机制:full cuda graph 模式尚不支持 AOT 调度,因为 scheduler 与 kernel 必须始终使用相同的 num_splits(在 dynamic split scheduler 中作为上界),而 num_splits 目前由 kernel 启动代码启发式决定;旧实现强制 full cuda graph 走 AOT 调度,并把 scheduler_metadata 拷入固定缓冲、清零剩余部分(Woosuk 的 NOTE 指出不清零会导致部分 thread block 使用无效 scheduler metadata 覆写输出缓冲),该路径存在导致非法内存访问的隐患。

## 修复方案
改 `vllm/v1/attention/backends/flash_attn.py`:
- `__init__`:full cuda graph 下不再要求 AOT 调度(删除原来「full cuda graph 需要 FlashAttention 3 的 AOT scheduling,否则 raise ValueError」的逻辑),改为 `self.aot_schedule = False`,即 full cuda graph 一律用动态(非 AOT)调度。
- 删除 `self.scheduler_metadata` 持久缓冲相关逻辑:`__init__` 中不再创建该 buffer,`build()`/`schedule()` 中不再把每次的 scheduler_metadata 拷入缓冲并清零剩余部分,而是直接使用当次生成的 scheduler_metadata。

```diff
-        if self.use_full_cuda_graph and not self.aot_schedule:
-            raise ValueError("Full CUDA graph mode requires AOT scheduling, "
-                             "which requires FlashAttention 3.")
+        if self.use_full_cuda_graph:
+            # NOTE(lucas): AOT scheduling not supported in full cuda graph mode
+            #  yet. ...
+            self.aot_schedule = False
```

测试:`lm_eval --model vllm --model_args '{"pretrained":"Qwen/Qwen2.5-VL-72B-Instruct","tensor_parallel_size":8,...}' --tasks gsm8k`,full_cuda_graph true/false 均得出正常分数(flexible-extract exact_match 约 0.528/0.529)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:v1 FlashAttention 后端的 full cuda graph 路径(调度方式由 AOT 改为动态、scheduler_metadata 缓冲移除);full cuda graph 不再强制要求 FA3(此前 FA2 会被 ValueError 拒绝,现在走同一动态调度路径);非 full-cuda-graph 路径不涉及。

## 关联
正文未列出(仅提到发现者 @WoosukKwon)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
