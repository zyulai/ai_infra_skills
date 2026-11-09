# PR #57995: [Perf][MoE] Support fp8 combine in FlashInfer one-sided MoE all2all

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Ayu190505 |
| 合并 | 2026-10-02 |
| 分类 | MoE |
| Label | performance, ready, nvidia, verified |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/57995 |

## 一句话结论

FlashInfer one-sided MoE all2all 的 combine 段改走 FP8 传输,DEP8 serving 吞吐 +4.0%、median ITL -2.7%(opt-in env)。

## 动机

FlashInfer one-sided MoE all2all 的 combine(expert 输出回传)默认以 BF16 走 NVLink,带宽占用为每 token `hidden_size*2` 字节。FlashInfer 的 `MoeAlltoAll.combine` 已支持 `use_low_precision` 参数(FP8 量化传输,写入时量化,combine 段 NVLink 流量减半),但 vLLM 侧没有把开关接出来。

## 优化点

`FlashInferNVLinkOneSidedManager` 新增 `low_precision_combine` 字段与 `_resolve_low_precision_combine()`:env `VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE` 打开且安装的 FlashInfer `MoeAlltoAll.combine` 签名含 `use_low_precision`(用 `supports_kw` 反射检查,否则 warning 后回退 BF16)时启用;`combine_into` 在支持 `output` 的新路径上透传 `use_low_precision=self.low_precision_combine`。关键细节:workspace 仍按 BF16 combine payload 尺寸申请(hidden_size*2/ token),因为 FP8 内核是写时量化、不缩减 kernel 侧校验的 region 要求——注释明确"低精度传输不缩小该需求";FlashInfer < 0.6.16 无 `output` 与 `use_low_precision`,走旧路径不受影响。

## 关键改动

```text
vllm/distributed/device_communicators/all2all.py: low_precision_combine 解析 + combine_into 透传 use_low_precision
vllm/envs.py: 新增 VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE(默认 False)
tests/distributed/test_mnnvl_alltoall.py: 透传单测 + 真实 FP8 combine 多进程测试
```

```python
if not envs.VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE:
    return False
try:
    supported = supports_kw(
        MoeAlltoAll.combine, "use_low_precision", allow_var_kwargs=False
    )
except (TypeError, ValueError):
    supported = False
```

## 性能声明

测试条件:GB300 集群 A/B(开关 `VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE`),DeepSeek-R1-0528 NVFP4,8k input / 1k output。

- 微基准(`benchmarks/kernels/benchmark_moe_a2a_combine.py`,DeepSeek R1 shapes,按 tokens/rank 扫描):DEP8 每 dispatch+combine 小幅收益 2-8 µs(约 1-2%),DEP4 结果 mixed。
- nsys serving(DEP4 与 DEP8):combine kernel 在 decode 规模调用上约快 17%,但大 prefill 规模调用更慢——FP8 kernel 每线程 191 寄存器 vs BF16 96,疑似 prefill 规模下寄存器 spilling 所致。
- 无 profiler serving,DEP8(2 节点):FP8 较 BF16 输出吞吐 +4.0%,median ITL -2.7%;以 4 次 server 启动(bf16/fp8/fp8/bf16)平衡顺序,每次基准跑 3 遍。
- 附注:一次 fp8 run 在 profiler 停止后不久挂起;无 profiler 的 run 无挂起。

## 验证与回归风险

`tests/distributed/test_mnnvl_alltoall.py` 新增两类测试:fake kernel 断言 `use_low_precision` 逐字段透传;真实 FP8 combine 传输测试(world_size=2,`@requires_multi_gpu @requires_one_sided @requires_ptrace`,缺 `use_low_precision` 支持则 skip)——利用整数 1..top_k 在 fp8_e4m3 中可精确表示的性质,断言 combine 结果与 BF16 完全一致(workspace/dtype 不匹配会静默损坏而非报错)。风险面:精度上 combine 段改走 FP8 传输,数值经 FP8 量化(测试用的整数 payload 恰好无损,真实 hidden state 有损);prefill 规模 kernel 可能因寄存器压力变慢,故默认关闭、需显式 env 开启;依赖安装的 FlashInfer 版本能力。

## 关联

同区域(FlashInfer MoE A2A):索引内 PR #36022(FlashInfer MoE A2A kernel 引入,本 PR 的基座)、PR #20691(SM100 FlashInfer fused allreduce,同为 FlashInfer 通信优化);后继可关注 prefill 规模寄存器压力的 kernel 修复。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
