# PR #34257: [JIT Kernel] Migrate per-token FP8 quantization from AOT to JIT

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-08-11 |
| 分类 | Quantization (tier: major) |
| Label | quant, sgl-kernel, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/34257 |

## 一句话结论

per-token FP8 E4M3 量化从 AOT wheel 迁到 JIT kernel,H200 几何均值 1.095x、B300 1.061x,输出 bit-exact。

## 动机

SGLang CUDA serving 的动态 per-token FP8 激活量化此前调用预编译的 `sgl_kernel` AOT wheel,使 CUDA 运行时对 AOT wheel 存在构建/分发依赖。目标是在完全保持现有输出缓冲契约、数值行为与公开兼容 API 的前提下,把该算子迁入 SGLang 自己的轻量 JIT kernel 基础设施,让 kernel 随运行环境(架构、dtype)按需编译,削减对 AOT wheel 的依赖面。

## 优化点

新增 JIT CUDA 实现 `per_token_quant_fp8.cuh`,保留 AOT 版既有的两种 dispatch 策略:warp kernel(每 warp 量化一个 token,`warp::reduce_max` 归约,适合大批量)与 CTA kernel(每 CTA 一个 token,`cta::reduce_max` 共享内存归约,适合小批量),launch 时按 `num_tokens >= sm_count * 2 * 8` 自动切换。向量宽度按 hidden_dim 是否被 `16/sizeof(T)` 整除选择 16B 或 4B 向量化。Python 侧经 `load_jit` 按输入 dtype 编译模块,SM90 上启用 `--use_fast_math`,其余架构用 precise_math。kernel 注册表把 `quantization.sgl_per_token_quant_fp8` 的 backend 从 AOT 改为 JIT,外面包一层 `register_custom_op`(声明 mutates_args)保持 torch.compile 兼容;AOT 的 `sgl_kernel.sgl_per_token_quant_fp8` 入口保留给外部消费者与 AOT/JIT 差分对比,MUSA 继续走 AOT。padded 输出契约保留:输出行数可多于输入,kernel 只写前 `num_tokens` 行。

## 关键改动

```text
python/sglang/kernels/jit/csrc/gemm/per_token_quant_fp8.cuh(新):
  warp/CTA 两种 dispatch 的 per-token FP8 E4M3 量化 kernel
python/sglang/kernels/ops/quantization/per_token_quant_fp8.py(新):
  load_jit 模块缓存(@cache_once)+ torch.compile 兼容 custom-op 封装
python/sglang/kernels/ops/quantization/__init__.py:
  KernelSpec backend AOT→JIT;fp8_kernel.py: CUDA 走 JIT,MUSA 拆分保留 AOT
test/registered/kernels/ops/quantization/test_per_token_quant_fp8.py(新):
  AOT vs JIT 逐 bit 差分;benchmark 直接对比 JIT/AOT
```

```cpp
// 按批量在 warp 与 CTA dispatch 间切换
const bool use_warp_kernel =
    num_tokens >= sm_count * 2 * kPerTokenQuantTokensPerCTA;  // 8 tokens/CTA
```

## 性能声明

注册的 AOT-vs-JIT benchmark 矩阵:FP16/BF16,token 数 1..7807,hidden 512..4096,共 72 配置。H200(SM90):JIT 在 58/72 配置更快,几何均值 1.095x,范围 0.885x-3.254x(最慢 FP16 (39,4096):2.925 us vs 2.588 us;最快 FP16 (7807,1368):10.965 us vs 35.684 us)。B300(SM103):48/72 更快,几何均值 1.061x,范围 0.864x-2.453x(最慢 BF16 (7807,512):4.852 us vs 4.194 us;最快 FP16 (7807,1368):8.743 us vs 21.442 us)。以上均为 kernel 级 microbenchmark,无端到端吞吐数字。

## 验证与回归风险

`test_per_token_quant_fp8.py` 在 H200(SM90)与 B300(SM103)各 32 tests 通过,对 FP16/BF16/FP32 输入、CTA/warp 两条 dispatch、非 2 次幂 hidden、零行与 FP8 中点值(raw-bit 语义)做 AOT-vs-JIT 逐 bit 相等断言;另覆盖 padded serving 输出、padded tail 不被改写、不支持 dtype 拒绝。B200 上验证 SM100 系 `redux.sync.max.f32` 归约路径,注册进 `base-b-kernel-unit-test-4-gpu-b200`。回归风险:hidden_dim 需被 4 整除(kernel 内 CHECK);legacy 的 dispatch 相关零行行为(scale=0 时 scale_inv=0 与否)必须逐 bit 保持,SM90 fast_math 与其他架构 precise_math 的编译差异使跨架构 bit-exact 依赖差分测试持续覆盖;MUSA 路径拆分后需各自 import 正确。

## 关联

同分类(Quantization/JIT kernel)其他 PR 编号 pack 未列出;同文件体系中保留的 MUSA AOT 路径与 per-token group quant(`sgl_per_token_group_quant_8bit_v2`)为相邻但未被本 PR 改动的算子,无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
