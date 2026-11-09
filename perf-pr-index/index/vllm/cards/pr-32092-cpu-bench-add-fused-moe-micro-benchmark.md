# PR #32092: [cpu][bench] Add Fused MoE Micro Benchmark for CPU Backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @andikarachman |
| 合并 | 2026-01-12 |
| 分类 | MoE(major) |
| Label | performance, ready, cpu |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/32092 |

## 一句话结论

新增 CPU 后端 fused MoE kernel 微基准脚本,输出多档延迟统计与 TFLOP/s,补齐 CPU MoE 性能度量入口(修复 issue #31721)。

## 动机

CPU 后端的 `cpu_fused_moe` 缺少独立微基准(issue #31721),调 kernel/ISA 时缺少可比对的标准工具;脚本注释说明 ISA 选择与参数约定沿用 `test_cpu_fused_moe.py` 的模式。

## 优化点

新增 `benchmarks/kernels/cpu/benchmark_cpu_fused_moe.py`:随机生成 bf16 输入与 w13/w2 权重(按 hidden/intermediate 尺度缩放初始化),经 `cpu_prepack_moe_weight` 预打包(ISA 可选 amx/vec,依据 `torch._C._cpu._is_amx_tile_supported()` 决定可选集),router logits 经 softmax+topk 得到路由;5 次 warmup 后按 `time.perf_counter_ns` 测 iters(默认 20)次,打印 min/max/mean/std/median(ms)与 TFLOP/s(FLOPs 估算 `2 * batch * topk * (hidden * up_dim + intermediate * hidden)`,up_dim = 2*intermediate)。CLI 支持 batch-size/expert-num/hidden-size/intermediate-size/topk-num/use-bias/activation(silu|swigluoai)/isa/seed/iters;topk 默认 expert_num//2;CPU ops 不可用时打印明确错误并以非零码退出。

## 关键改动

```text
benchmarks/kernels/cpu/benchmark_cpu_fused_moe.py: 新增 CPU fused MoE 微基准(预打包权重、ISA/激活/bias 可配、TFLOP/s 输出)
```

## 性能声明

PR 正文仅以截图给出测试实例与结果,未给出文字数字;素材无法读取截图内容,具体数字 PR 未以文本注明。默认参数:batch 64、experts 8、hidden 2880、intermediate 2880、topk 4、bf16、silu。

## 验证与回归风险

工具型 PR,无单测;运行前提为 Linux x86_64 且编译含 AVX2/AVX512 的 vLLM CPU 扩展(脚本启动时校验)。回归面:无——仅新增 benchmark 脚本,不触碰运行时路径。

## 关联

修复 issue #31721;模式参照 `test_cpu_fused_moe.py`。素材未列出其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
