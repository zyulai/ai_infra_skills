# PR #60885: [Bugfix] Reject mismatched float16/bfloat16 KV cache dtype

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60885](https://github.com/vllm-project/vllm/pull/60885) |
| 作者 | cipheraxat |
| 合并日期 | 2026-10-15 |
| labels | bug, rocm, ready, nvidia |
| 热度 | 👍 0 · 💬 8 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
显式 fp16/bf16 KV cache dtype 与模型 dtype 不一致时,后端各自出错;改为在 VllmConfig 校验期直接拒绝并提示改法。

## 问题现象
`dtype=bfloat16` 搭配 `kv_cache_dtype=float16` 时,引擎照常接受配置,随后在 CUDA-graph capture 阶段崩溃(正文根因段);issue #60859 的 A100 数据显示不同后端症状不一:FlashInfer 在 init 时 assert,FlexAttention 能跑但输出静默错误(mean NLL 13.98,`auto` 时为 3.96)。触发条件是显式指定的 float16/bfloat16 KV cache dtype 与模型 dtype 不一致;影响面为所有走显式 dtype 指定的部署,配置期无法发现。

## 根因
正文披露:FlashAttention 的 `mha_varlen_fwd` 要求 query 与 key 共享同一 dtype;引擎此前接受 `dtype=bfloat16` + `kv_cache_dtype=float16` 的组合,直到 capture 时才崩。校验缺失发生在配置层,而各 attention 后端对该组合的表现不一致(断言、崩溃或静默错输出),因此选择在配置构造期一次性拒绝,而非在后端选择层规避。review 决定放在 `VllmConfig.__post_init__`:检查在 `try_verify_and_update_config()` 之后运行,模型 hook(如 DeepSeek V3.2 会把显式 bfloat16 cache 重写为 auto)仍有机会先改写 cache dtype。

## 修复方案
`vllm/config/vllm.py` 新增 `_reject_mismatched_float_kv_cache` 并在 `__post_init__` 中 `try_verify_and_update_config()` 之后调用:

```python
+        raise ValueError(
+            f'kv_cache_dtype="{cache_dtype}" is incompatible with '
+            f'dtype="{model_dtype}". Please set `--kv-cache-dtype` to {allowed}.'
+        )
```

显式 float16/bfloat16 与模型 dtype 不一致即抛错;float32 模型配显式 float16 只提示用 `auto`(因 float32 不是合法 `--kv-cache-dtype`);`"auto"`、匹配的 fp16/bf16 组合、FP8 均不受影响。同时在 `vllm/config/cache.py` 的 `cache_dtype` 文档补一句约束说明,并在 `tests/test_config.py` 增加参数化测试(6 passed)。作者注明端到端 `LLM.generate` 未在 GPU 上运行(本地无 CUDA)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:配置校验路径,把原本延迟到后端 init/capture 的失败前移到引擎配置构造期。此前"能跑但静默错输出"(FlexAttention)的组合现在直接报错——依赖该错误组合启动的既有脚本会从"静默错误"变为"启动失败",属预期收紧;Triton 本可向上转型处理的未量化 KV tile 也被拒绝,正文明确选择拒绝而非走该回退路径。

## 关联
- [#60859](https://github.com/vllm-project/vllm/issues/60859)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
