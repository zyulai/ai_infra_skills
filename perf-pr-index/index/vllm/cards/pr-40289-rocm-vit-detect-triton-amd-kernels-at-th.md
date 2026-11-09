# PR #40289: [ROCm][ViT] Detect Triton-AMD kernels at their new aiter location

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Lafunamor |
| 合并 | 2026-07-31 |
| 分类 | Attention / Kernel \| tier: major |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1, multi-modality, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, mistral |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/40289 |

## 一句话结论

修复 ROCm ViT 对新 aiter 路径漏检，恢复 Triton-AMD/FLASH_ATTN 快速路径。

## 动机

`vllm/platforms/rocm.py` 的 `flash_attn_triton_available()` 决定 ROCm 上 ViT encoder 是否提供 `AttentionBackendEnum.FLASH_ATTN`。ROCm/flash-attention commit `3f94643`（2026-03-12，"[AMD] Migrate to Triton Backend to Aiter"）移除了 `flash_attn.flash_attn_triton_amd` 子包，把 kernel 迁到 `aiter.ops.triton._triton_kernels.flash_attn_triton_amd`。此后 `find_spec("flash_attn.flash_attn_triton_amd")` 在当前 ROCm wheel 上恒为 `None`，ViT 静默回落到 `TORCH_SDPA`；在 gfx1151（Strix Halo / RDNA3.5）上 SDPA 会产生 NaN/Inf 图像 embedding、使 LM 陷入无限 `!` 输出，其他 ROCm 目标则只是丢失快速路径。

## 优化点

把探测条件放宽为同时接受旧子包与新 aiter 位置：仅当两个导入路径都缺失时才返回 `False`，否则返回 `True`。新增 `_has_spec` 辅助函数并捕获 `ImportError`/`ValueError`。当旧的 `flash_attn.flash_attn_triton_amd` 仍存在（旧构建）时行为不变；仅新路径存在（当前构建）时现在能正确返回 `True`。

## 关键改动

```text
vllm/platforms/rocm.py: flash_attn_triton_available() 用 _has_spec 同时检查 legacy 与 aiter 两个 kernel 路径，仅两者皆无才返回 False
```

```python
+        if not (
+            _has_spec("flash_attn.flash_attn_triton_amd")
+            or _has_spec("aiter.ops.triton._triton_kernels.flash_attn_triton_amd")
+        ):
             return False
```

## 性能声明

PR 正文未给出定量性能数据（该 PR 为后端探测修复，正文未附 benchmark 数字）。PR 未注明测试条件。

## 验证与回归风险

静态验证：新检查在「仅新路径存在」的 aiter 构建上返回 `True`、在两者皆无时返回 `False`。在 gfx1151 上用显式 `--mm-encoder-attn-backend TRITON_ATTN`（另一条代码路径、vLLM 自有的 `vit_triton_attn_wrapper`）确认 Triton ViT kernel 在 Qwen3-VL-A3B AWQ 上产出正确 image embedding。作者明确说明未能端到端验证本 PR 启用的自动选择 `AttentionBackendEnum.FLASH_ATTN` ViT 路径——测试环境中 aiter 的 JIT-built `.so` 因另一独立打包问题不被 importlib 发现；建议有可用 aiter 环境的 reviewer 取消该 flag 复验。回归风险：自动选择路径待验证，旧构建行为不变。

## 关联

无（正文仅引用 ROCm/flash-attention commit `3f94643`，非 vLLM PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
