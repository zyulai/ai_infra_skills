# PR #52861: [Model][NVIDIA] Route DSA models to the CUDA non-compiled path

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2026-08-19 |
| 分类 | Speculative Decoding (tier: major) |
| Label | new-model, speculative-decoding, ready, deepseek, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,16 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/52861 |

## 一句话结论

把 DeepSeek V3.2/GLM-5.2 DSA 模型在 NVIDIA GPU 上路由到专用 CUDA 实现并默认启用 breakable CUDA graph(关编译),MTP=3 批 1 解码吞吐 2.502x。

## 动机

优化的 NVIDIA 类(`vllm.models.deepseek_v32` 下的 CUDA 实现)此前从 model registry 不可达:`DeepseekV32ForCausalLM` 与 `GlmMoeDsaForCausalLM` 都映射到 generic `deepseek_v2` 实现,DeepSeek V3.2 与 GLM-5.2 只能跑通用 runner/graph 路径,其 MTP draft 也无法用匹配实现,且旧代码里有"v32 MTP 不支持 cudagraph,强制 enforce_eager"的 FIXME 覆盖。本 PR 接替因分支同步而丢失的 #49790,是 #48597 跟踪的 NVIDIA DSA routing 项。

## 优化点

(1) 注册表路由:`DeepseekV32ForCausalLM`/`GlmMoeDsaForCausalLM` 改指 `vllm.models.deepseek_v32` 平台入口,CUDA 上导入 `nvidia.model` 实现(SM100/SM103 优化按能力门控、低配 GPU 有 fallback),ROCm/XPU/CPU 保持 generic 实现且 ROCm 仍走编译路径;(2) 新增 `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` 集合,DSA 主模型与 MTP draft 默认开启 breakable full + piecewise CUDA graph,并自动设 `CompilationMode.NONE`(非编译路径),保留 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 显式退出;同时把原内联的 auto-enable 逻辑重构为 `_maybe_enable_breakable_cudagraph`;(3) 两个 DSA 架构加入 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`(ROCm 上排除 V3.2/V4);(4) MTP:`hf_config_override` 对 `deepseek_v32`/`glm_moe_dsa` 生成 `DeepseekV32MTPModel`,移除 enforce_eager 覆盖,llm_base_proposer 认可其 tuple 返回;(5) 放宽 KV cache 限制:去掉 `require_fp8_kv_cache` 硬断言,支持非量化 BF16 与 standard/packed FP8 两种 KV cache 形式,`fused_norm_rope` 的 fp8 判定改用 `is_quantized_kv_cache`。

## 关键改动

```text
vllm/model_executor/models/registry.py: DSA 架构路由到 vllm.models.deepseek_v32,注册 DeepseekV32MTPModel
vllm/models/deepseek_v32/__init__.py: 平台入口,CUDA 走 nvidia 实现,其他平台回落 generic
vllm/config/vllm.py: DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES + _maybe_enable_breakable_cudagraph(设 CompilationMode.NONE)
vllm/config/speculative.py: v32/glm_moe_dsa 的 MTP 映射到 DeepseekV32MTPModel,删除 enforce_eager 覆盖
vllm/models/deepseek_v32/attention.py, common/kernels.py: 放宽 fp8 KV cache 约束,BF16 支持
```

关键代码(平台入口路由):

```python
if current_platform.is_cuda():
    from .nvidia.model import DeepseekV32ForCausalLM
    from .nvidia.model import DeepseekV32ForCausalLM as GlmMoeDsaForCausalLM
    from .nvidia.mtp import DeepseekV32MTP
else:
    # ROCm, XPU, and CPU keep the generic implementation.
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP as DeepseekV32MTP
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV3ForCausalLM as DeepseekV32ForCausalLM)
```

## 性能声明

4x NVIDIA GB200,TP=4,nvidia/GLM-5.2-NVFP4(fp8_e4m3 KV cache,max_model_len 16384,max-num-seqs 256,FlashInfer 0.6.17):完整 GSM8K(1,319 题、5-shot、温度 0、并发 100、MTP=3)准确率 94.768%(1,250/1,319),MTP 平均接受长度 3.373、draft 接受率 79.104%,2,937.604 output tok/s。批 1 SPEED-Bench throughput_16k/low_entropy(8,192 输入/1,024 输出,3 次测量):无 MTP vs MTP=3,output 吞吐 130.17→304.15 tok/s(2.337x,+133.7%),decode 吞吐 135.09→338.05 tok/s(2.502x,+150.2%),TPOT 7.40→2.96ms,TTFT 293.90→340.44ms。FlashMLA_SPARSE + fp8_ds_mla 冒烟:114.01 output tok/s、8.75ms TPOT。作者注明 DeepSeek V3.2 与 pre-SM100 GPU 未在本 PR 基准测试。

## 验证与回归风险

tests/test_config.py 199 passed;registry 导入 3 passed;SM100 fused DSA norm/RoPE cache 测试(auto/bfloat16/fp8)30 passed;pre-commit 全部通过。DeepSeek V3.2 从 H100 compile-startup 与 NVFP4 fusion-E2E 套件移除(NVIDIA 路径默认不编译),ROCm/AITER 编译测试保留。风险面:默认 CUDA graph 模式与 runner 选择的平台相关行为变化影响 DSA 用户;compile 测试覆盖面缩小(不再有 V3.2 编译路径的 E2E 验证);KV cache 类型放宽后 bf16 路径为新覆盖面。

## 关联

同区域:#49790(被本 PR 取代)、#51915(ROCm/MXFP4 opt-in 路径,不改默认路由)、#48597(tracking issue)、#12755/MTP 系(#12755)、Kimi K3(#50000,同在 breakable cudagraph 默认集合)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
