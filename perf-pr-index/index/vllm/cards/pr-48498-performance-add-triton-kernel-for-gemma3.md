# PR #48498: [Performance] Add Triton kernel for Gemma3n sparse GELU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @BWAAEEEK |
| 合并 | 2026-09-14 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, rocm, intel-gpu, ready, torch.compile, nvidia, vllm-ir |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动覆盖 kernel、IR 注册、平台默认 priority、activation 层、测试与 benchmark |
| 链接 | https://github.com/vllm-project/vllm/pull/48498 |

## 一句话结论

为 Gemma3n 的 `gelu_and_mul_sparse` 激活新增融合 Triton provider 并接入 vLLM IR:A100 上 42/42 生产 shape 快于编译 native,几何均值 2.59x。

## 动机

Gemma3n 的稀疏激活 `GeluAndMulSparse`(逐行 mean/std 高斯阈值化 + tanh GELU + 门控乘)此前只有 native 实现或 Inductor 编译路径,由多个小 kernel 组成(mean、std、sub、relu、gelu、mul 各自 launch)。相关 IR 迁移 PR #41903 仅给该算子提供了 native 实现;本 PR 补上专用 Triton provider,并把该算子从 #41903 的迁移范围中排除以避免重复。

## 优化点

单 kernel 融合整段计算:一个 program 处理一行,载入 gate/up 两半,在寄存器内算 per-row mean 与 variance,构造 `gate - (mean + std*std_multiplier)` 的截断,ReLU 后做 tanh GELU 与门控乘。为逐位对齐 native PyTorch 的算子边界舍入,mean/std/cutoff/中间结果在 FP32 计算后按输入 dtype 做 roundtrip(`.to(input_dtype).to(tl.float32)`)。launch 配置用 reduction-width 启发式按 block size 选 1/2/4/8 warps(上限 8,生产维度 d>=2048 仍为 8),替代 review 前的固定 8 warp。集成层面:算子注册进 vLLM IR(`ir.ops.gelu_and_mul_sparse`),`GeluAndMulSparse` 层改走 IR dispatch;CUDA 平台默认 priority 设为 `["triton", "native"]`,ROCm/XPU 保持 `["native"]`;kernel 通过 `torch.library.triton_op` + `wrap_triton` 暴露,使 `torch.compile` 可追踪 launch 且 standalone AOT 依赖不丢。约束:CUDA、contiguous、三种 dtype、tanh 近似、intermediate size <= 32768,不满足自动回退 native。

## 关键改动

```text
vllm/kernels/triton/activation.py: 新增融合 kernel + triton_op 包装 + IR provider 注册(含 supports_args 分发谓词)
vllm/ir/ops/activation.py: 新增 gelu_and_mul_sparse IR 算子与 native 参考实现、输入生成器
vllm/model_executor/layers/activation.py: GeluAndMulSparse.forward_native 改调 ir.ops,std_multiplier 预先 .item()
vllm/platforms/cuda.py / rocm.py / xpu.py: 默认 IR op priority 分别加 gelu_and_mul_sparse 配置
tests/kernels/ir/test_activation.py 等: 精度/fallback/opcheck/IR lowering/CUDA Graph/standalone AOT 覆盖
```

关键舍入对齐代码:

```python
mean = mean.to(input_dtype).to(tl.float32)
std = tl.sqrt(variance).to(input_dtype).to(tl.float32)
scaled_std = (std * std_multiplier).to(input_dtype).to(tl.float32)
cutoff = (mean + scaled_std).to(input_dtype).to(tl.float32)
```

## 性能声明

BF16 kernel 基准(42 个 shape = token 1-2048 x intermediate 2048-32768,`do_bench_cudagraph`):A100-SXM4 40GB 上 42/42 快于 torch.compile 的 native,几何均值 2.59x、中位 2.49x、范围 1.15x-4.06x,对 eager native 几何均值 4.85x;B200 上 42/42 更快,范围 1.11x-4.06x。warp 启发式相对固定 8 warp:A100 小维度几何均值 +1.022x,B200 18 个组合 +1.134x。E2E:Gemma3n E2B,B200 短 prompt 中位延迟基本不变(172.79 vs 172.92 ms);8 prompt x 2048 input x 1 output 中位 150.14 → 147.81 ms(1.58%);GSM8K(1319 题,5-shot,greedy)accuracy 0.5262(native)vs 0.5307(Triton),输出吞吐 11,322.2 → 11,757.0 tok/s。A100 E2E 为 parity(mean 快 0.48%,Welch p=0.627,无统计显著回归)。测试环境:B200 SM100(PyTorch 2.11.0+cu130/CUDA 13.0)与 A100 SM80(2.11.0+cu129)。

## 验证与回归风险

双架构各 76 passed(sparse activation 74 + compiler 2),相邻 activation/IR/custom-op 回归 274 passed,Gemma3n 官方初始化测试 2 passed。精度按 dtype 设定容差(FP32 max err 1.43e-6、FP16/BF16 max err 0、零 mask 不匹配为 0);正文特别指出移除输入 dtype roundtrip 会在 A100 上导致 FP16 14/24、BF16 21/24 用例超容差并产生数百到数千个 sparse-mask 不匹配,该行为由测试钉住。风险面:仅 tanh 近似受支持(approximate="none" 回退 native);`d=32768` 时 A100 有 48 字节 spill 但仍是实测最快配置,作者声明无需按架构限制;TP>1 时统计量按 shard 本地计算(沿用原注释语义,未改变);`GeluAndMulSparse(0.0)` 仍显式报错引导用 `GeluAndMul`。

## 关联

同区域:PR #41903(四个 activation 家族的 IR 迁移,本 PR 要求其保留其余家族、排除 gelu_and_mul_sparse)、PR #32676;IR 基础设施相关见索引中 vllm-ir/torch.compile 条目(如 pr-23207)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
