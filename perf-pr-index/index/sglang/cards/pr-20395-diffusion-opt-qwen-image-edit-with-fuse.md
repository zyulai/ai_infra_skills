# PR #20395: [Diffusion] Opt qwen-image-edit with fuse_residual_layernorm_scale_shift_gate_select01_kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-03-13 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/20395 |

## 一句话结论

为 Qwen-Image-Edit 把 residual+LayerNorm+scale/shift/gate/select01 融合成单个 Triton kernel,denoising 每步 0.6383s -> 0.6256s,e2e 约 2% 提升;该调用点 kernel 时间 167us -> 72us。

## 动机

qwen-image DiT 的 modulation 路径此前在 CUDA 上是一串拆开的 kernel:先 `gate_x * x + residual_x`(scale residual 时)、再单独跑 `apply_layernorm_only`(代码里明确标注"TODO: fuse norm with new select01 kernel, workaround now")、最后调 `fuse_scale_shift_gate_select01_kernel`。三段各读写一遍大张量,launch 与访存开销在小 batch 的扩散模型推理里占比高,torch profiler 显示该调用点耗时约 167us。

## 优化点

新增两个 Triton kernel,把整条 modulation 链在一次行内遍历完成:(1) `_fused_residual_layernorm_scale_shift_gate_select01_kernel`——一行内先算 `residual_out = residual + residual_gate * x` 并写出,再对该 residual_out 做 LayerNorm(mean/var/rstd),可选乘 weight 加 bias,然后按 per-token 的 `index` 从 (scale0/shift0/gate0) 与 (scale1/shift1/gate1) 两组调制参数中 select01 选出一份,算 `y = x_hat * (1.0 + scale) + shift`,同时把选出的 gate 写到 gate_out;(2) `_fused_layernorm_scale_shift_gate_select01_kernel` 为无 residual 的版本。计算全程 fp32(x 载入后 `.to(tl.float32)`),支持 HAS_WEIGHT/HAS_BIAS 编译期分支,`BLOCK_N = min(65536/elem_size, next_pow2(C))`,C 超过 64KB 报错。模型侧 `_modulate` 重写:删除 `_is_cuda` 分支判断与 `apply_layernorm_only` workaround,统一改走两个融合 kernel;`modulate_index` 参数从 `List[int]` 改为 `torch.Tensor`。

## 关键改动

```text
python/sglang/jit_kernel/diffusion/triton/scale_shift.py: 新增两个融合 Triton kernel 及 wrapper(residual 版返回 output/residual_out/gate_out)
python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py: _modulate 改调融合 kernel,删除 layernorm workaround 与非 CUDA 的 torch.where 分支
python/sglang/multimodal_gen/runtime/layers/layernorm.py: 移除 apply_layernorm_only workaround
python/sglang/jit_kernel/tests/test_qwen_image_modulation.py: 与拆分基线(norm_infer + scale_shift kernel)对比的正确性测试
python/sglang/jit_kernel/benchmark/bench_qwen_image_modulation.py: split vs fused 微基准
```

关键片段(residual 版核心计算序列):

```python
residual_out = residual + residual_gate * x
tl.store(res_out_row_ptr + cols, residual_out, mask=mask)
mean = tl.sum(residual_out, axis=0) / inner_dim
...
rstd = tl.rsqrt(var + eps)
idx = tl.load(index_ptr + ...).to(tl.int1)
scale = tl.where(idx, scale1, scale0)
y = x_hat * (1.0 + scale) + shift
```

## 性能声明

测试命令统一为 Qwen/Qwen-Image-Edit-2511、1024x1024、50 步推理、guidance-scale 4.0、开 torch-compile、无 CPU offload:

- e2e:DenoisingStage 每步 0.6383 s -> 0.6256 s,50 步 31.9185 s -> 31.2824 s,约 2% e2e 提升(warmed-up request 32.37 s -> 31.73 s)
- torch profiler:该调用点 167us -> 72us
- 微基准(bench_qwen_image_modulation.py,B x S x D 扫描,单位 us):residual 版 fused 相对 split kernel 全面更快,如 (1,128,1024) 49.82 -> 17.25,(2,2048,3072) 90.27 -> 49.47;layernorm 版 (1,128,1024) 38.30 -> 7.90,(1,2048,3072) 37.44 -> 22.82
- 输出图像正常(PR 附图);PR 未注明具体 GPU 型号

## 验证与回归风险

新增 UT `test_qwen_image_modulation.py`:以旧拆分路径(norm_infer + fuse_scale_shift_gate_select01)为参考,对两个新融合 kernel 做 dtype x batch x seq x hidden 参数化对比(fp32 容差 1e-5,fp16/bf16 容差 5e-2);bench 脚本含 CI 缩减范围。回归面:qwen_image DiT 的 `_modulate` 路径被重写,非 CUDA 平台的旧 torch.where 实现被删除、统一走 Triton kernel(需要 CUDA);NPU 侧文件末尾保留 `fuse_scale_shift_native` fallback import;`modulate_index` 类型签名变更影响调用方;Checklist 项均未勾选,PR 未提供准确性量化数字(仅输出图像截图)。

## 关联

前置工作:同文件既有的 `fuse_scale_shift_gate_select01_kernel`(本 PR 把 LayerNorm 并入其输入端);代码注释中"TODO: fuse norm with new select01 kernel"即本 PR 完成的事项。pack 未列出同分类其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
