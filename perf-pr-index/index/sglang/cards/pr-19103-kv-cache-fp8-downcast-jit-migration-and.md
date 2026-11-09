# PR #19103: KV Cache FP8 Downcast: JIT Migration and Optimization to Near-HBM-Limited Performance on H200

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Johnsonms |
| 合并 | 2026-03-27 |
| 分类 | Quantization (tier: major) |
| Label | quant, sgl-kernel, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/19103 |

## 一句话结论

把 KV cache BF16/FP16->FP8 E4M3 下转 kernel `downcast_fp8` 迁入 JIT 框架并做向量化 + 2D grid + `__restrict__` 三段优化,H200 上峰值有效带宽 917 GB/s -> 约 4.2 TB/s(约 87% 名义 HBM 利用率),延迟最高降 61x。

## 动机

`downcast_fp8` 在单次 GPU pass 内完成 KV cache 张量的缩放、clamp 与 FP8 E4M3 转换。旧实现在 AOT sgl-kernel 构建路径中,迁移到 JIT 框架可降低构建复杂度(该部分 PR 明确声明性能中性、不计入提速归因)。迁移后 profiling 显示 scalar 实现仍有三个瓶颈:每条 memory instruction 搬运字节少、暴露的 warp 级并行度低、指针别名假设导致编译器代码生成保守。相关上下文为 #17865。

## 优化点

性能优化分三段推进,每段针对一个明确瓶颈:(1) 128-bit 向量化访存——`kVecSize = 16/sizeof(T)`,BF16/FP16 下每线程 2B 变 16B,每线程加载一个 K 向量、一个 V 向量,转换后向量化写 FP8,峰值带宽 917 -> 3646 GB/s;(2) 2D grid 暴露并行——把原先绑死在 `block = dim/kVecSize`(d=128 时仅 16 线程即 0.5 warp/block)的 launch 改为 `grid(input_num_tokens, ceil(head*dim/kVecSize/256))`、固定 256 线程 block,`head*dim` 的并行从内层循环搬到 CUDA grid,h=32,d=128 时每 token 活跃 warp 从 0.5 升到 16,越过 Little's-Law 饱和阈值(bytes-adjusted coverage 0.018x -> 4.51x),进入近 HBM-bound 区;(3) 对 kernel 指针参数加 `__restrict__`,让编译器把不变的 scale 值留在寄存器、去掉冗余 reload,大形状上再加最多 +17%。kernel 语义(scale、clamp 到平台相关 FP8 max、经 `loc/mult/offset` 的目的索引、BF16/FP16 输入)完整保留。

## 关键改动

```text
python/sglang/jit_kernel/csrc/elementwise/cast.cuh: 新增优化版 fused_downcast_kernel(向量化 + 2D grid + __restrict__)与 downcast_fp8 入口
python/sglang/jit_kernel/cast.py: JIT wrapper,按 dtype 用 @cache_once 缓存编译模块
python/sglang/jit_kernel/include/sgl_kernel/type.cuh: 新增 fp8_e4m3_t dtype trait 与平台相关 clamp 值(CUDA 448.0 / ROCm FNUZ 224.0)
python/sglang/jit_kernel/tests/test_cast.py: BF16/FP16 正确性、scale/clamp/loc scatter/mult-offset 与边界值测试
python/sglang/jit_kernel/benchmark/bench_cast.py: AOT vs JIT 延迟与有效带宽基准
sgl-kernel/csrc/elementwise/cast.cu: 移除旧 AOT 实现与注册路径(diff 中保留 FP8_E4M3_MAX 常量引用改动)
```

关键片段(cast.cuh,launch geometry 与向量化核心):

```cpp
constexpr int kVecSize = 16 / sizeof(T);
const int num_vecs = h * d / kVecSize;
dim3 grid(num_tokens, grid_y);
dim3 block(kBlockSize);   // 256 threads
// 每线程:k_vec.load / v_vec.load,clamp+cast 后 out_k.store / out_v.store
```

## 性能声明

硬件 H200(名义 HBM 4.8 TB/s),基准为 kernel 级 `bench_cast.py`,带宽按 K/V 一阶流量模型(bytes = 6 x tokens x head x dim)计算;所有 speedup 均相对 JIT scalar baseline:

- 阶段账本:scalar baseline 峰值约 917 GB/s(约 19% 利用率)-> 向量化 3646 GB/s(约 76%,kernel 延迟降 4.0x-8.8x)-> 2D grid 约 4.2 TB/s(约 87%,较向量化版再快 1.4x-6.8x)-> `__restrict__` 大形状再加最多 +17%
- 端到端延迟:sl=64,h=32,d=128 时 100.6 us -> 1.7 us(61x);sl=256,h=32,d=256 时 102.3 -> 3.7 us(28x);sl=1024,h=32,d=128 时 103.0 -> 6.0 us(17x);sl=2048,h=32,d=256 时 217.9 -> 25.5 us(9x)
- `__restrict__` 分形状带宽:(2048,32,128) 3456 -> 4028 GB/s(+17%);(1024,32,128) 3678 -> 4179 GB/s(+14%);极小形状无变化
- PR 未提供服务级 e2e 吞吐数字

## 验证与回归风险

`test_cast.py` 覆盖:BF16/FP16 基本功能(写入槽非零、未写槽保持零)、与 PyTorch fp8 参考的逐位对齐、非单位 scale、clamp 饱和(±448 边界及超出)、loc 非连续 scatter、mult/offset 目的索引、`static_cast` 转换路径在 ±448 边界附近的位模式。回归面:向量化假设 KV-cache 布局兼容 128-bit 访存(测试的 head dim 为向量大小倍数);`__restrict__` 假设输入/输出/scale/loc 不别名(与预期 API 用法一致);JIT 按 dtype 编译缓存,编译开销每 dtype 一次;ROCm 上 FNUZ/E4M3 两种 clamp 上限由宏区分。旧 AOT 路径移除,依赖 `sgl_kernel.downcast_fp8` 的调用方需改走 `sglang.jit_kernel.cast`。

## 关联

相关 PR:#17865(JIT 化方向上下文)。同分类 Quantization 其他 PR 编号 pack 未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
