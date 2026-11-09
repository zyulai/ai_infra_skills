# PR #29281: [KDA-Pilot] Add diffusion causal Conv3D cat-pad CUDA fast path for Cosmos3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-26 |
| 分类 | Other (major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 6 个文件:JIT CUDA kernel、Python 封装、调度接入、compile 配置与测试/基准 |
| 链接 | https://github.com/sgl-project/sglang/pull/29281 |

## 一句话结论

为 Cosmos3 VAE causal Conv3D 的 cat+pad 拷贝模式新增 CUDA JIT fast path,kernel 组按 trace 加权提速约 2 倍,E2E 提升约 2%。

## 动机

Cosmos3 VAE 因果卷积的热点模式等价于对连续 5D NCTHW 张量执行 `F.pad(torch.cat([cache_x, x], dim=2), (Wl, Wr, Ht, Hb, Dl - cache_t, Dr))`,此前只有 Triton 实现。Cosmos3-Nano T2V trace(832x480x9f、4 步)在 VAE decode 中命中 8 个生产 shape 共 48 次可融合调用。这是 memory-bound 的拷贝/补零 kernel,不是 denoise kernel,因此预期模型级影响温和,应体现在 decode/total 而非 denoise。任务来自 KDA-Pilot `b200_diffusion_causal_conv3d_cat_pad__multi_shape`(BBuf/KDA-Pilot#132),由 MIT HAN Lab Kernel Design Agents 生成。

## 优化点

新增 `diffusion_causal_conv3d_cat_pad` JIT CUDA kernel:单 pass 遍历输出张量,按输出坐标反推来源(cache 或 x 或补零),用 16 字节向量化的 `uint4` pack 读(`SGLANG_LDG` + Pack union)与写,grid-stride 循环覆盖全部向量;进入条件用 TensorMatcher 静态校验 5D contiguous CUDA 张量及 padding 约束(depth_left ≥ cache_t、pad_d_right=0、宽/高对称 padding、输出元素数整除向量宽度、16 字节对齐)。注册为带 fake impl 的 eager SGLang custom op,使 `torch.compile` 追踪时看到不透明 op 而不会追踪 JIT 加载。调度侧默认先试 CUDA,JIT 加载/launch 失败则永久回退已有 Triton 实现。另将 Cosmos3 gen-layer 的 torch.compile 配置从 `dynamic=True` 改为 `dynamic=False`(静态 compile shape),保证 compile 路径可运行。

## 关键改动

```text
python/sglang/jit_kernel/csrc/diffusion/causal_conv3d_cat_pad.cuh: 新增向量化 cat+pad 拷贝 kernel(带 MIT HAN Lab 来源注释)
python/sglang/jit_kernel/diffusion/causal_conv3d_cat_pad.py: JIT 加载、can_use 守卫、custom op + fake impl
python/sglang/multimodal_gen/runtime/layers/parallel_conv.py: 统一入口,CUDA 失败后回退 Triton
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py: compile 配置 dynamic True→False
test/registered/jit/diffusion/test_causal_conv3d_cat_pad.py: 8 shape 正确性 + torch.compile(fullgraph=True) 测试
test/registered/jit/benchmark/diffusion/bench_causal_conv3d_cat_pad.py: 8 个 trace shape 的独立基准
```

回退机制核心:

```python
def fused_causal_conv3d_cat_pad(x, cache_x, padding):
    global _causal_conv3d_cat_pad_cuda_failed
    if (fused_causal_conv3d_cat_pad_cuda is not None
            and can_use_fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
            and not _causal_conv3d_cat_pad_cuda_failed):
        try:
            return fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
        except Exception:
            _causal_conv3d_cat_pad_cuda_failed = True
    return fused_causal_conv3d_cat_pad_triton(x, cache_x, padding)
```

## 性能声明

- Kernel 基准(8 个 Cosmos3 trace shape,bfloat16,marker 基准 200 iters):B200 按 trace 调用数加权 10.621→5.240 ms(**2.03x**,省 5.381 ms),单 shape 1.33x-2.18x;H100 加权 11.729→5.684 ms(**2.06x**,省 6.045 ms),单 shape 1.23x-2.20x。
- 模型 E2E(B200,`nvidia/Cosmos3-Nano` 原生 Cosmos3Pipeline,T2V 832x480x9f、4 步、seed 42,`enable_torch_compile=True`(mode default、fullgraph False、dynamic False),每实现独立进程、4 warmup、15 次取中位数):Triton fallback 181.521 ms → CUDA fast path 177.687 ms,**1.021x(+2.11%)**。
- 视觉等价:eager 输出对比 SSIM 全帧 1.000000、PSNR 无穷大。

PR 自述:预期收益是 kernel 组约 2x,整模型加速有意保持温和,因为该组只占 Cosmos3 T2V 运行时的一小部分。

## 验证与回归风险

新增测试:8 个 Cosmos3 shape 的 CUDA vs Triton 逐元素 `assert_close(atol=0, rtol=0)` 与 `torch.compile(fullgraph=True)` custom-op 测试,B200 与 H100 各 9 passed;另有 py_compile/ruff 检查。CI 状态:PR Test (Base) 沙漏(进行中/排队),PR Test (Extra) 失败。风险面:kernel 只进入 contiguous CUDA 张量且 padding 满足约束的调用,其余回退 Triton;`dynamic=False` 改动只影响 Cosmos3 `--enable-torch-compile` 路径的 shape 泛化(静态 shape 编译);CUDA 失败标志为进程级一次性熔断。

## 关联

同区域:已有 Triton 实现 `causal_conv3d_pad`(本 PR 的 fallback 与正确性参照);Wan VAE cat+pad 融合等 diffusion JIT kernel 系列(pack 未列编号)。来源任务:KDA-Pilot BBuf/KDA-Pilot#132。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
