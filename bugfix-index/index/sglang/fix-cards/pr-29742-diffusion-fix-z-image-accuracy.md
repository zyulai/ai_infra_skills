# PR #29742: [diffusion] Fix Z-Image accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29742](https://github.com/sgl-project/sglang/pull/29742) |
| 作者 | qimcis |
| 合并日期 | 2026-07-08 |
| labels | run-ci, diffusion, jit-kernel |
| 热度 | 👍 2 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Z-Image-Turbo 批处理/单张生成都严重劣化,修复 text encoder position_ids、采样/归一化对齐原版实现、批内 padding 的 RoPE 偏移与注意力掩码。

## 问题现象
使用 `--batching-mode dynamic` 时,`Tongyi-MAI/Z-Image-Turbo` 在请求被合并批处理时会生成严重劣化的图像。作者进一步发现即使单张生成,当前实现相比 [Tongyi-MAI 原生 pytorch 实现](https://github.com/Tongyi-MAI/Z-Image)也已严重劣化。修复在 1xh100 上以 batching-max-size 5、8 steps、guidance_scale 0.0 对 5 个测试 prompt 逐一生成对比验证(效果图见原帖)。本 PR 修复 issue #28502。

## 根因
PR 正文披露了三层根因:
1. qwen3 text encoder 默认 `position_ids` 形状为 `[1, seq]`,即使 `hidden_states` 是 `[batch, seq, dim]`,导致批处理 prompt 时 RoPE 以错误的扁平化 token 布局施加;
2. z-image 的采样与归一化与原版实现不一致:使用了 scheduler 默认 sigma 路径、外层 autocast、共享的 fp32 累加 RMSNorm,改变了该模型的去噪轨迹与 bf16 激活;
3. 批处理时图像/描述被填充到共享 batch 长度,但没有为每个请求保留 RoPE 偏移,也没有在注意力中掩掉仅因批处理产生的额外 padding,导致混合 prompt 的动态 batch 关注无效 token、使用错误的图像位置。

## 修复方案
可见部分 diff 包含(pack 中 diff 在 40000 字符处截断,以下仅为可见部分):
- 新增 `python/sglang/jit_kernel/diffusion/triton/zimage_native_norm.py`:两个 Triton 内核 `_rmsnorm_scale_kernel`、`_rmsnorm_tanh_residual_kernel`,按 bf16 精度语义执行 RMSNorm+scale / RMSNorm+tanh(gate)+residual,对外提供 `zimage_rmsnorm_scale`、`zimage_rmsnorm_tanh_residual`(不满足条件时返回 None 回退);
- `configs/pipeline_configs/zimage.py`:`ZImagePipelineConfig` 新增 `enable_autocast: bool = False`(去掉外层 autocast),新增 `prepare_sigmas` 覆盖以替换 scheduler 默认 sigma 路径;
- `runtime/layers/attention/layer.py`:新增 `build_varlen_mask_meta_from_lengths` / `build_varlen_mask_meta_from_ranges`,从 host 侧有效 token 区间构建 varlen FA 元数据(cu_seqlens/indices/inv_indices),避免 GPU `nonzero` 动态 shape 路径,用于按请求保留有效区间并掩掉批 padding。

注:text encoder position_ids 形状与 RoPE 偏移的具体修改位于被截断的 diff 部分,内容未可见,不作转述。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:multimodal_gen 的 Z-Image 流水线(配置、去噪归一化内核、varlen 注意力掩码构建)及新增 jit-kernel Triton 模块;`build_varlen_mask_meta_from_*` 为新增 API 并导出。Triton 内核带 `_can_use` 条件判断,不满足 bf16/CUDA/连续性/维度条件时回退 PyTorch 实现。正文性能数据:单张 1024x1024 9 步 loop time 863.27ms → 851.57ms,batch=5 总耗时 4907.5ms → 4781.3ms,单张与批处理均约快 2%。

## 关联
- [issue #28502](https://github.com/sgl-project/sglang/issues/28502)
- [Tongyi-MAI/Z-Image 原生实现](https://github.com/Tongyi-MAI/Z-Image)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
> 注:pack 中 diff 在 40000 字符处被截断(file granularity),截断部分未在本卡转述。
