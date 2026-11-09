# PR #7844: Refactor dynamic LoRA update to fix incorrect handling of variant weight shapes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7844](https://github.com/sgl-project/sglang/pull/7844) |
| 作者 | lifuhuang |
| 合并日期 | 2025-07-14 |
| labels | lora |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
动态加载 LoRA 时,若后续 adapter 的 rank/target modules 超过初始 adapter,可能失败甚至(CUDA graph 下)静默产出错误结果;本 PR 重构 LoRAManager/LoRAMemoryPool,新增 `--max-lora-rank`、`--lora-target-modules` 参数与校验,不兼容时明确报错。

## 问题现象
动态 LoRA 更新实现存在两个主要限制:Issue 1(已知限制,#7463)是用户必须至少提供一个初始 `--lora-paths`;Issue 2(bug,#7808)是初始 adapter 的形状(target modules、lora_rank)必须与所有后续动态加载的 adapter 相同或「更大」,否则进程可能失败,更糟的是在开启 CUDA graph 时可能产生不正确的结果(静默错误)。触发命令示例:`python3 -m sglang.launch_server --max-loras-per-batch 1 --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache --lora-paths lora=algoprog/fact-generation-llama-3.1-8b-instruct-lora` 后用 bench_serving 动态换 adapter。

## 根因
正文披露:初始 adapter 形状决定了 LoRA 内存池等内部状态的容量,后续动态加载更大/不同形状的 adapter 时无法正确处理权重形状更新,导致失败或错误结果。修复目标是以向后兼容方式解决:允许用户通过新 server args 显式声明容量(类似 vLLM),不兼容时返回显式、可理解的错误,而不是崩溃或隐式出错。

## 修复方案
1. 重构 `LoRAManager` 与 `LoRAMemoryPool` 以正确处理权重形状更新:`LoRAMemoryPool` 在构造时即按 `max_lora_rank` 与 weight names 完成缓冲区初始化,并新增 `can_support()` 判断内存池能否容纳某 adapter(rank 不超过 `max_lora_rank` 且 target modules 是已支持集合的子集);内存池改为加载首个 adapter 时惰性初始化,`apply_lora_configs` 统一推导/应用配置。
2. `load_lora_adapter` 中新增 `validate_new_adapter`,不兼容时抛出 ValueError 并提示用 `--max_lora_rank` / `--enable_lora_modules` 指定期望配置,不再崩溃或产出错误结果。
3. `server_args.py` 新增 `--max-lora-rank`、`--lora-target-modules`(可选 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj),未指定时自动从 `--lora-paths` 的 adapter 推断,保证旧命令结果与性能等价;`model_runner.py` 传入这两个参数;`utils.py` 的 `get_normalized_lora_weight_names` 改为接受模块名集合;并补充正反用例测试与文档。

关键 diff 摘录(validate_new_adapter):

```diff
+    def validate_new_adapter(self, lora_name: str, lora_config: LoRAConfig):
+        incompatible = self.memory_pool and not self.memory_pool.can_support(lora_config)
+        if incompatible:
+            raise ValueError(
+                f"LoRA adapter {lora_name} with rank {lora_config.r} is incompatible with the current LoRA memory pool configuration."
+                ...
```

## 影响与回归风险
- 回归面:LoRA 全链路——`lora_manager.py`、`mem_pool.py`、`utils.py` 的公共接口签名有变(`LoRAMemoryPool.__init__` 新增必填参数、`init_buffers` 签名简化);动态加载不兼容 adapter 时行为从「可能失败/错果」变为「明确报 ValueError」;`runners.py` 同步新增参数。文档注释说明动态更新 adapter 形状仍依赖 FlashInfer backend 弃用与 CUDA graph 重捕获,计划在 25 上半年完成,当前 `apply_lora_configs` 实际只在初始化阶段生效。
- 注:pack 中 diff 在 40000 字符处截断(file granularity),测试文件等后续改动未获取到。

## 关联
- https://github.com/sgl-project/sglang/issues/7463 (Issue 1 已知限制)
- https://github.com/sgl-project/sglang/issues/7808 (Issue 2 bug)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
