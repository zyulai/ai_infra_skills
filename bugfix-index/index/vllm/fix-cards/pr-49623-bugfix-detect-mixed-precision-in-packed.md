# PR #49623: [Bugfix] Detect mixed precision in packed KV cache specs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49623](https://github.com/vllm-project/vllm/pull/49623) |
| 作者 | mgoin |
| 合并日期 | 2026-07-24 |
| labels | bug, ready, v1 |
| 热度 | 👍 2 · 💬 1 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
打包 KV cache 规格(`UniformTypeKVCacheSpecs`)被混合精度检查跳过,worker 与 scheduler 判断不一致导致首个请求失败;改为检查内部 spec 修复。

## 问题现象
使用打包(packed)KV cache 规格且存在混合精度 attention 规格时,首个请求会失败。具体场景(测试计划中给出):以 DSpark 方法 + FP8 KV cache + TP=2 服务 `deepseek-ai/DeepSeek-V4-Flash-DSpark`,提交第一个 completion 请求即触发问题——修复前 `_zero_kv_blocks_kernel` 执行失败;修复后首个 completion 返回 HTTP 200 并生成 "Paris",DSpark 接受 15 个草稿中的 6 个。回归测试层面,修复前 `worker_config.needs_kv_cache_zeroing` 为 false。

## 根因
PR 正文披露:worker 侧 `KVCacheConfig.has_mixed_precision_kv_cache` 检查只查看顶层的 `AttentionSpec` 对象;被打包的 spec(`UniformTypeKVCacheSpecs` 内部的 spec)被跳过,而 `generate_scheduler_kv_cache_config` 却会解包它们。于是 scheduler 判定存在混合精度、将 block 标记为需要清零(zeroing),但 worker 侧因判定"非混合精度"而未创建 zeroer,两侧配置不一致,导致第一个请求失败。

## 修复方案
修改 `vllm/v1/kv_cache_interface.py` 中 `KVCacheConfig.has_mixed_precision_kv_cache` 属性:遍历 KV cache group 时,若 group spec 是 `UniformTypeKVCacheSpecs` 则解包其内部 `kv_cache_specs` 逐一收集精度,否则按原逻辑处理单个 spec,再据此计算精度集合:

```diff
-        kv_cache_precisions = {
-            (group.kv_cache_spec.dtype, group.kv_cache_spec.kv_quant_mode)
-            for group in self.kv_cache_groups
-            if isinstance(group.kv_cache_spec, AttentionSpec)
-        }
+        for group in self.kv_cache_groups:
+            group_spec = group.kv_cache_spec
+            group_specs = (
+                list(group_spec.kv_cache_specs.values())
+                if isinstance(group_spec, UniformTypeKVCacheSpecs)
+                else [group_spec]
+            )
+            kv_cache_precisions.update(
+                (spec.dtype, spec.kv_quant_mode)
+                for spec in group_specs
+                if isinstance(spec, AttentionSpec)
+            )
```

新增回归测试 `test_mixed_precision_kv_cache_with_uniform_type_specs`(tests/v1/core/test_kv_cache_utils.py):构造 FP8 与 BF16 两个 `UniformTypeKVCacheSpecs` 组,断言 worker 与 scheduler 配置的 `needs_kv_cache_zeroing` 一致且均为 true。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是使用打包 KV cache 规格(`UniformTypeKVCacheSpecs`)且层间 KV 精度混合(FP8 + BF16 等)的配置,典型如 DSpark + FP8 KV cache。
- 回归面:仅 `has_mixed_precision_kv_cache` 的判定逻辑变宽(原先漏算的内部 spec 现在被纳入);任何依赖该属性决定 KV cache zeroing 初始化的 worker/scheduler 路径都会受影响,行为变化方向是"更准确地判定混合精度、需要 zeroing 时创建 zeroer"。测试面:焦点回归测试 1 passed,整个 `test_kv_cache_utils.py` 71 passed。

## 关联
- https://github.com/vllm-project/vllm/pull/47574(正文查重时搜索过该 PR,未发现重复修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
