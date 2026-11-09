# PR #18336: [diffusion][hot fix] fix torch.compile graph break caused by torch._dynamo.disable

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yingluosanqian |
| 合并 | 2026-02-06 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | sgl-kernel, run-ci, diffusion |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 jit kernel、其测试与 multimodal_gen layernorm |
| 链接 | https://github.com/sgl-project/sglang/pull/18336 |

## 一句话结论

把 diffusion jit kernel 的 `@torch._dynamo.disable` 改为 `torch.library.custom_op`,消除 torch.compile graph break,QwenImage 每步 287ms 回到 258ms。

## 动机

#14717 给 jit kernel(`fused_norm_scale_shift` 等 CuTe DSL 融合核)加了 `@torch._dynamo.disable` 来跳过 Dynamo 追踪,但这在 torch.compile 图中引入 graph break,抬高 CPU 开销,并在部分模型(如 QwenImage,每步 263ms -> 287ms)造成 GPU 气泡。#14717 在最后几次 commit 后没有重跑 e2e 基准,回归未及时发现。本 PR 按 PyTorch 文档推荐,改用 `torch.library.custom_op` 注册自定义算子,使 Dynamo 无需打断全图。

## 优化点

两个 jit 入口改为 `@torch.library.custom_op("sglang::fused_norm_scale_shift"/"sglang::fused_scale_residual_norm_scale_shift", mutates_args=())` 并补 `register_fake`(返回 `x.new_empty` 形状),让 torch.compile 全图追踪;内部 stream 改为运行时取 `torch.cuda.current_stream()`,不再作为默认参数暴露给 op schema。因为 custom_op 的 schema 不接受 `Union[int, Tensor]`,gate 从 `Union[Optional[torch.Tensor], int]` 收紧为 `Optional[torch.Tensor]`:None 在 kernel 内以标量 1 占位(TVM-FFI 不支持 None),layernorm 调用层对 `int gate != 1` 显式 raise;kernel 侧同时删掉了 Int32 标量 gate 的编译分支与 `to_cute_arg` 包装,直接传 torch tensor。被禁用的 int-gate 语义在调用点被拒绝而非静默出错。

## 关键改动

```text
python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py: 两个入口改 torch.library.custom_op + register_fake;stream 运行时获取
python/sglang/multimodal_gen/runtime/layers/layernorm.py: forward_cuda 对 int gate != 1 显式报错
python/sglang/jit_kernel/tests/test_fused_norm_scale_shift.py: 移除 "int1" gate 参数化
```

```python
-@torch._dynamo.disable  # Disable Dynamo tracing
+@torch.library.custom_op("sglang::fused_norm_scale_shift", mutates_args=())
 def fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps=1e-5):
     ...
+@fused_norm_scale_shift.register_fake
+def _fused_norm_scale_shift_fake(x, weight, bias, scale, shift, norm_type, eps):
+    y = x.new_empty(x.shape)
+    return y
```

## 性能声明

H200,`--enable-torch-compile`,DenoisingStage 日志(QwenImage:Qwen/Qwen-Image-2512,1024x1024,50 steps,guidance 4.0,seed 42;Wan2.2/2.1:720p 40 steps 81 帧,8 卡 cfg-parallel + ulysses 4):

- QwenImage:回归后(4739f2e)0.2876 s/step -> 本 PR 0.2582 s/step,denoise 阶段约 1.9% speedup(修复前基线 669a9bd 为 0.2637 s/step;正文表述回归约 9% 已被本 PR 修复)
- Wan2.2-T2V denoise:约 2.0% speedup(4.9828 -> 4.8879 s/step)
- Wan2.1-T2V denoise:约 2.8% speedup(1.0801 -> 1.0533 s/step)
- hunyuan denoise:<1% speedup(1.7645 -> 1.7611 s/step)

正文解释:约 2% 是合理预期——融合 kernel 在约 2200 us 的单层中省 50-60 us,正好约 2%;此前观察到的约 10% 收益来自更快 kernel 与消除 GPU 气泡两部分,气泡如今已被其他机制(torch.compile 等 CPU 侧优化)消除,只剩 kernel 级加速。Wan2.2/Wan2.1/hunyuan 在回归前后均无回归。

## 验证与回归风险

精度:与本 PR 与 669a9bd(#14717 之前)对比 QwenImage、hunyuan video、Wan2.2 的生成图像(截图对比,无数值指标)。单测更新:gate 的 "int1" 模式移除,与 int gate 支持收紧一致。风险:`gate` 类型收紧为 Optional[Tensor],调用方传非 1 int 会 raise(原来允许标量);op schema 变化要求 sgl-kernel 相关 jit 模块同步;register_fake 只给形状不做数值,compile 路径下语义正确性依赖 custom_op 的 mutates_args 声明准确。Checklist 各项均未勾选(素材所示)。

## 关联

前序:#14717(引入 @torch._dynamo.disable 的回归源头)。同区域其他 diffusion compiler 类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
