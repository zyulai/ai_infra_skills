# PR #32233: [ROCm][CI] Fix HuggingFace flash_attention_2 accuracy issue in Isaac vision encoder

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32233](https://github.com/vllm-project/vllm/pull/32233) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-01-13 |
| labels | rocm, ready, multi-modality |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
HF `flash_attention_2` 在 ROCm 上绕过 torch.backends.cuda 设置、结果错误,Isaac 视觉编码器输出与 CUDA 完全不同;测试侧强制 vision encoder 用 SDPA(math_sdp)对齐。

## 问题现象
Isaac 模型测试在 ROCm 上失败,输出与 CUDA 完全不同(图片描述文本截然不同):CUDA 生成 "This image captures a vibrant street scene in what appears to be a Chinatown district",ROCm 生成 "The image depicts a scene with a central figure, a man, who is dressed"。

## 根因
正文披露:通过 ROCm/CUDA 逐张量对比定位到视觉编码器的 attention 层——image input、patch embedding、Q/K/V projection 两侧一致(如 Q projection -250937.7 vs -250937.1),而 **self_attn output** 开始分叉(ROCm -10642.1 vs CUDA -9478.6),最终 embeddings 差异更大(-2272.1 vs 5632.5)。bug 在 HuggingFace 的 `flash_attention_2` 实现:它绕过 `torch.backends.cuda` 设置、直接调用 flash_attn 库,在 ROCm 上产生错误结果。

## 修复方案
测试侧修复,不改模型代码:
- `tests/models/multimodal/conftest.py` 新增 `patch_hf_vision_attn_for_rocm()`:仅 ROCm 生效,把 vision encoder 各层 `self_attn.vision_config._attn_implementation` 强制为 `"sdpa"`,SDPA 随即按 conftest 已有设置走 `math_sdp` 后端;

```python
if not current_platform.is_rocm():
    return
inner = getattr(model, "model", model)
if hasattr(inner, "vision_embedding"):
    vit = inner.vision_embedding[0]
    for layer in vit.encoder.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.vision_config._attn_implementation = "sdpa"
```

- `tests/models/multimodal/generation/vlm_utils/model_utils.py` 的 `isaac_patch_hf_runner()` 中调用该 patch,使 Isaac 的 HF 运行走 SDPA。

验证:换 SDPA 后 ROCm inputs_embeds sum=5632.5361、CUDA 5632.5068(mean 均 0.002371),两平台 token 序列一致;而 flash_attention_2 下为 -2272.1099 vs 5632.5000。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅测试基础设施(conftest 与 Isaac 测试工具);只影响 ROCm 上 Isaac HF 测试的 attention 实现选择,CUDA 平台直接 return 不受影响。正文注明为 ROCm SDP 精度问题解决前的过渡方案(TODO:HF 侧修复后移除)。

## 关联
- Related to https://github.com/vllm-project/vllm/issues/30167

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
