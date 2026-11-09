# PR #24505: [Bug] [Spec Dec]: Fix kv_cache dtype mismatch for Eagle3 drafter on FP8 target

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vllmellm |
| 合并 | 2025-09-16 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, ready, llama |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 仅含 `vllm/model_executor/models/llama_eagle3.py` 一个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/24505 |

## 一句话结论

修复 Eagle3 drafter 未继承 FP8 kv_cache_dtype 导致的崩溃,使 FP8+投机解码在 ROCm 上可用并带来 +44% 吞吐。

## 动机

ROCm 平台上,eagle3 投机解码与 FP8 量化组合会触发 RuntimeError:drafter 模型没有继承目标模型的 kv_cache_dtype,其 KV cache 默认落在 uint8,而目标模型用 fp8,dtype 不一致导致 Triton attention kernel 崩溃。根因是投机解码的 "scratch model" 拥有独立 ModelConfig,缺少与主模型之间的配置继承/同步机制;auto 模式下最终解析为 uint8 而非查询主模型配置(issue #21308)。

## 优化点

修复很小但关键:`LlamaModel.__init__`(eagle3 drafter)调用 `get_current_vllm_config()` 取全局引擎配置,把 `cache_config` 直接传给 drafter 的 `LlamaDecoderLayer`;`LlamaDecoderLayer.__init__` 增加 `cache_config` 参数并透传给父类。这样目标模型 kv_cache_dtype 为 fp8 时 drafter 自然拿到 fp8,为 auto/float16 时行为完全不变。作者有意避免 "if 'fp8' in ..." 这类硬编码,使修复对任意 dtype 通用。其余改动均为参数在构造函数间的传递支撑。

## 关键改动

```text
vllm/model_executor/models/llama_eagle3.py: 唯一改动文件——
  LlamaModel.__init__ 获取全局 cache_config 并传入 LlamaDecoderLayer,
  LlamaDecoderLayer 增加 cache_config 形参
```

关键片段:

```python
current_vllm_config = get_current_vllm_config()
self.layers = nn.ModuleList([
    LlamaDecoderLayer(
        config=self.config,
        cache_config=current_vllm_config.cache_config,
        prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"),
    )
])
```

## 性能声明

测试在 AMD MI300X(部分场景含 NVIDIA GPU)、Llama-3.1-8B-Instruct、ShareGPT 1000 prompts。AMD+FP8(tp=1, kv-cache-dtype fp8):spec-dec 吞吐 +44%、TPOT -30.7%(输出吞吐 5274.92 → 6202.75 tok/s,request 26.78 → 34.49 req/s);draft acceptance rate 最终约 47.6%,mean acceptance length 1.95。AMD+FP16:相对基线几乎无加速(约 1.01x),TTFT +4.8%、ITL +50.5%、TPOT 约 -1.7%。NVIDIA+FP16+FlashAttention:吞吐 +41%、TPOT -8.3%,但 TTFT +7.2%、ITL +6.2%。正确性(lm_eval):gsm8k Exact Match 0.7635 → 0.765,hrm8k_math_en 0.3692 → 0.3859,无回归。

## 验证与回归风险

验证用 lm_eval(gsm8k、hrm8k_math_en)确认精度无回归,以及 vllm bench serve / 自研压测脚本对比吞吐与延迟。风险:auto 与 float16 路径行为不变,但 drafter 现在隐式继承全局 CacheConfig,其他依赖 drafter 独立 cache 配置的场景(如显存占用估计)可能受影响。作者另指出 CUDA 平台 `is_kv_cache_dtype_supported` 在 V1 + Triton + eagle3 组合下仍返回 false 并抛 NotImplementedError,该问题未在本 PR 解决。

## 关联

解决 issue #21308。同分类相关:#22872(EAGLE3 多模态)、#25916(EAGLE3 preamble 修正)、#5765、#13365。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
