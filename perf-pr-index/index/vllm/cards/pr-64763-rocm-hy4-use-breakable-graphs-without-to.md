# PR #64763: [ROCm][HY4] Use breakable graphs without torch.compile

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zixi-qi |
| 合并 | 2026-11-04 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | documentation, rocm, torch.compile, ci/build |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 docs、config、model、tests |
| 链接 | https://github.com/vllm-project/vllm/pull/64763 |

## 一句话结论

HY4 AMD 骨干移除 torch.compile、默认改用 breakable HIP graphs,8×MI355X C8 decode 吞吐 +18.7%、TPOT −15.6%。

## 动机

编译会把 prefill 尺寸的 MXFP8 GEMM tile 固化进小 batch decode:编译后 MXFP8 linear kernel 只能使用 prefill 配置,decode 走小 batch 时 kernel 时间异常高。按 #57526 的方向让 HY4 与 NVIDIA 执行方式对齐——不编译,但保留 prefill/decode 的 graph capture,使 graph capture 按 batch 尺寸选择内核。

## 优化点

从 HYV4Model(AMD backbone)移除 @support_torch_compile;default_breakable_cudagraph_architectures 在 ROCm 下把 HYV4ForCausalLM 与 HYV4MTPModel 加入 breakable graph 默认集合,使其默认以 FULL_AND_PIECEWISE HIP graph 执行且运行时保留 shape 相关内核选择(含小 batch MXFP8 linear 配置)。回归测试重写为断言 backbone 不进编译(compile_mode=NONE)、graph_mode=FULL_AND_PIECEWISE、有 graph 捕获且生成正常;新增 docs/models/hy4.md 说明 ROCm 图执行(VLLM_USE_BREAKABLE_CUDAGRAPH=0 不恢复编译,调试用 --enforce-eager)。

## 关键改动

```text
vllm/models/hy_v4/amd/model.py: 移除 @support_torch_compile
vllm/config/vllm.py: ROCm default breakable 架构集合加入 HYV4ForCausalLM/HYV4MTPModel
tests/models/test_hyv4_rocm.py: 默认 graph 回归测试(在未改动 parent 上失败)
docs/models/hy4.md: 新增 ROCm HIP graph 执行文档
```

## 性能声明

tencent/Hy4-preview-FP8,上游 b91df5a9,8×MI355X,TP8/MRV2,BF16 激活/KV,无投机、无前缀缓存,compiled(6 次)/no-compile(3 次)取中位:C1 1024/256 +15.9%(36.77→42.62 tok/s);C8 4096/600 +18.7%(245.71→291.59 tok/s),TPOT 30.65→25.86 ms(−15.6%);C8 4096/16 +4.6%;C8 混合(128–6144/32–256)+14.0%,混合 TTFT +5.5%(555→586 ms)。_mxfp8_linear_kernel.kd 每 rank/step 时长 7.534→2.437 ms(同 411 次调用;强制 prefill tile 可复现 69% 损失)。GPQA-Diamond 91.92%→92.42%(+0.51pp)。

## 验证与回归风险

tests/models/test_hyv4_rocm.py 6 passed on AMD,新默认 graph 回归在未改 parent 上失败(可作为 fail-before 证据)。正文声明 GPQA 单点结果不足以证明整体精度持平。风险限于 ROCm HY4 路径:不再支持 torch.compile,VLLM_USE_BREAKABLE_CUDAGRAPH=0 无法恢复,需用 --enforce-eager 关闭 graph 调试;prefill/decode graph capture 均保留。

## 关联

跟随 #57526(breakable graphs 方向);pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
