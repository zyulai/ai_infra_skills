# PR #22335: [AMD] Fix multimodal diffusion test crash on ROCm by falling back to SDPA

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22335](https://github.com/sgl-project/sglang/pull/22335) |
| 作者 | bingxche |
| 合并日期 | 2026-04-09 |
| labels | amd, sgl-kernel, run-ci, diffusion, jit-kernel |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
#20796 后 multimodal-gen FlashAttention 后端经 CUDA-only 的 FA3 分发,ROCm 上 warmup 即 TypeError/NotImplementedError 崩溃;ROCm 选择器在 FA3 不支持时回退 Torch SDPA。

## 问题现象
multimodal-gen(多模态扩散)任务在 ROCm 上,text encoder warmup 请求阶段即崩溃——#20796(`1a8eb890f6`,Kernels community fa3)之后,`FlashAttentionBackend` 的分发经 `sglang.jit_kernel.flash_attention_v3`(FA3,CUDA-only),导致:(1) `_is_fa3_supported()` 比较 `torch.version.cuda >= "12.3"`,ROCm 上 `torch.version.cuda` 为 `None` 抛 TypeError;(2) 即使挡住 None 比较,FA3 路径在 ROCm 上抛 NotImplementedError。此前 ROCm 可通过支持 ROCm 的 `flash_attn` 包走 FA2,新分发层总是路由到 FA3,FlashAttention 后端在 ROCm 完全不可用。ROCm 720 CI 因镜像未装 `flash_attn` 自然回退 Torch SDPA 而幸免;ROCm 700 装有 `flash_attn`,触发崩溃。

## 根因
正文披露:#20796 把 multimodal-gen 的 FlashAttention 分发改成必经 CUDA-only 的 FA3;`_is_fa3_supported()` 未考虑非 CUDA 平台(`torch.version.cuda` 为 None)导致 TypeError,且 FA3 本身在 ROCm 不可用,装了 `flash_attn` 的 ROCm 环境因此选中并进入不可用的 FA 后端。

## 修复方案
- `python/sglang/jit_kernel/flash_attention_v3.py::_is_fa3_supported()`:开头加 `if torch.version.cuda is None: return False`,非 CUDA 平台(含 XPU 及未来平台)提前判不支持。
- `python/sglang/multimodal_gen/runtime/platforms/rocm.py::get_attn_backend_cls_str`:ROCm 注意力后端选择器显式检查 `_is_fa3_supported()`(ROCm 上恒 False),不支持则记日志并回退 `AttentionBackendEnum.TORCH_SDPA`;原有 head-size 校验移到 `target_backend == FA` 守卫下保留。

关键 diff:

```python
                if not _is_fa3_supported():
                    logger.info(
                        "FlashAttention backend now dispatches through FA3 "
                        "(CUDA-only). Using Torch SDPA backend on ROCm."
                    )
                    target_backend = AttentionBackendEnum.TORCH_SDPA
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围(崩溃由 #20796 引入,其合并后至本 PR 前)。
- 回归面:`_is_fa3_supported()` 在所有平台新增 None 守卫(CUDA 上 `torch.version.cuda` 非 None,守卫为 no-op,正文验证无影响);ROCm multimodal-gen 的注意力后端选择:装了 flash_attn 也走 Torch SDPA 而非 FlashAttention;head-size 校验仅在 FA 后端时执行。Test Plan:multimodal-gen-test-1-gpu-amd(4 shards)、2-gpu-amd(2 shards)通过,ROCm 容器内确认选择 SDPABackend。

## 关联
正文未列出(提及 #20796 与测试 run 链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
