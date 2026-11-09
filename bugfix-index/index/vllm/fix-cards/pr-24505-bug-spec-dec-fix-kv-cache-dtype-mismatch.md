# PR #24505: [Bug] [Spec Dec]: Fix kv_cache dtype mismatch for Eagle3 drafter on FP8 target

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24505](https://github.com/vllm-project/vllm/pull/24505) |
| 作者 | vllmellm |
| 合并日期 | 2025-09-16 |
| labels | llama, ready, speculative-decoding |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
Eagle3 drafter 未继承目标模型 kv_cache_dtype,FP8 下 KV cache 退为 uint8 致 ROCm Triton kernel 崩溃;修复为继承引擎全局 CacheConfig。

## 问题现象
ROCm 平台上使用投机解码(eagle3 方法)+ FP8 量化时发生 RuntimeError,attention backend 中数据类型不匹配导致 Triton kernel 崩溃。正文以 AMD MI300X 上的基准/正确性测试矩阵(FP8 模式:baseline vs speculative,tp=2)验证:修复后 gsm8k 精度无回归(Exact Match 0.7635 → 0.765),AMD+FP8 下 spec-dec 吞吐 +44%、TPOT -30.7%。正文另指出 CUDA 侧存在相关但不同的问题:V1 引擎 + 强制 Triton backend + eagle3 组合下 `is_kv_cache_dtype_supported` 返回 false,经 `_is_v1_supported_oracle` 抛出断言错误(作者复现并修复了 ROCm 侧,同时提及该 CUDA 场景)。

## 根因
正文明确披露:drafter 模型未从目标模型继承 kv_cache_dtype,其 KV cache 默认为 uint8,而目标模型用 fp8,attention backend 数据类型不匹配导致 Triton kernel 崩溃。正文进一步解释架构层面原因:投机解码由后来加入的独立"scratch model"承载,vLLM 为其单独创建 ModelConfig;主模型与 draft 模型两个独立配置域之间缺乏可靠的配置继承与同步机制——作者曾尝试在 arg_utils/`config/__init__.py` 的 `__post_init__` 中给 draft_model_config 传 kv_cache_dtype 但未生效(auto 模式最终落到 uint8)。最终方案是在 `llama_eagle3.py` 直接获取引擎全局配置,建立主模型与 draft 模型的直接可靠连接。

## 修复方案
`vllm/model_executor/models/llama_eagle3.py`:导入 `CacheConfig` 与 `get_current_vllm_config`;`LlamaDecoderLayer.__init__` 新增 `cache_config` 参数并传给父类;顶层模型 `__init__` 中 `current_vllm_config = get_current_vllm_config()`,构造 decoder layer 时传入 `cache_config=current_vllm_config.cache_config`,使 draft 层拿到主模型的 kv_cache_dtype。其余改动主要为支持该参数在各层构造函数间传递。

```diff
+        current_vllm_config = get_current_vllm_config()
...
         self.layers = nn.ModuleList([
             LlamaDecoderLayer(
                 config=self.config,
+                cache_config=current_vllm_config.cache_config,
```

## 影响与回归风险
- 回归面:eagle3 draft 模型的构建路径(`llama_eagle3.py` 的 LlamaModel/LlamaDecoderLayer 构造签名变化,任何直接实例化这些类的代码需适配);所有使用 eagle3 的配置组合。正文称若主模型 kv_cache_dtype 为 auto 或 float16,draft 拿到的配置同样不变,无副作用;避免了硬编码 "if 'fp8' in ..." 的特判。

## 关联
- https://github.com/vllm-project/vllm/issues/21308#issuecomment-3192203896 (Resolves)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
