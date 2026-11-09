# PR #34302: [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @robertgshaw2-redhat |
| 合并 | 2026-02-23 |
| 分类 | Attention / Kernel |
| Label | performance, ready, ci/build, deepseek |
| Diff | diffstat 缺失(pack 未提供,正文 diff 另有 1 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/34302 |

## 一句话结论

为 DeepSeek-V3/Kimi-K2 的 MoE router GEMM 引入 TRTLLM 低延迟内核,小 batch(1-16 token)decode 下 B1 e2e 提速约 5.5%-6%。

## 动机

DSV3 类 MoE 每个 MoE 层都要先跑 router GEMM(hidden 7168 → 256/384 experts 的 logit),decode 阶段 batch 通常只有 1-16 token,这种瘦高形状走通用 GEMM 路径不划算。本 PR 从 sglang 的 sgl-kernel 移植 TRTLLM `dsv3RouterGemm` 低延迟内核(原始 PR:s-gl-project/sglang#7627),专攻 decode 小 batch。

## 优化点

新增 CUDA kernel `output = mat_a @ mat_b.T`:每个 block 负责一个 expert 列(B 按列主序直接取列),128 线程,`uint4` 向量载入 8 个 bf16 后转 fp32,用 PTX `fma.rn.f32x2` 与 butterfly `__shfl_xor_sync` 归约,跨 warp 用 shared memory 汇总,由 tid==0 写出。按 num_tokens(1-16)与 num_experts(256=DSV3、384=Kimi K2)、hidden=7168 全模板实例化,消除运行时分支。支持 PDL(Programmatic Dependent Launch):`griddepcontrol.wait` / `launch_dependents` + `cudaLaunchAttributeProgrammaticStreamSerialization`,由 `TRTLLM_ENABLE_PDL=1` 开启。Python 侧新增 `DeepSeekV2Gate` 替换原 `ReplicatedLinear`:仅当 CUDA + Hopper/Blackwell + experts/hidden 匹配且 `x.shape[0] <= 16` 时走专用 kernel,否则回落原路径;router logit dtype 在 monolithic expert 方法下用 fp32,否则 bf16。

## 关键改动

```text
csrc/moe/dsv3_router_gemm_entry.cu: 入口与形状校验(SM90-103、num_tokens 1-16、experts {256,384})
csrc/moe/dsv3_router_gemm_float_out.cu / dsv3_router_gemm_bf16_out.cu: fp32/bf16 输出两个 kernel(各 291 行)
vllm/model_executor/models/deepseek_v2.py: DeepSeekV2Gate 小 batch dispatch 到专用 kernel
CMakeLists.txt: 仅 CUDA 12+/SM90+ 架构集合编译(9.0a;10.x;11.0f)
```

```python
if self.allow_dsv3_router_gemm and x.shape[0] <= 16:
    return ops.dsv3_router_gemm(
        hidden_states=x, router_weight=self.weight, output_dtype=self.out_dtype), None
else:
    return super().forward(x)
```

## 性能声明

PR 正文:5.5% E2E speedup for Batch 1 Decode(标题称 6% B1)。`vllm bench serve` 对比(条件:input-len 2,output-len 100,max-concurrency 1,10 prompts,temperature 0;模型 PR 未注明,lane 为 `nvidia/DeepSeek-V3.1-NVFP4`):output token throughput 109.58 → 115.65 tok/s,request throughput 1.10 → 1.16 req/s,mean TPOT 9.00 → 8.50 ms,mean ITL 9.00 → 8.50 ms,TTFT 基本持平。硬件型号 PR 未注明(内核要求 SM90+,guard 支持 SM90-SM103)。gsm8k(limit 100,concurrency 10):flexible/strict exact_match 均 0.97。

## 验证与回归风险

验证:lm_eval gsm8k 精度回归 + `vllm bench serve` 单并发延迟基准。风险:内核形状硬编码(hidden 7168、experts 256/384、tokens 1-16),不匹配即回落 `ReplicatedLinear`;仅 CUDA(非 ROCm)且 SM90-103;`out_dtype` 依赖 expert quant_method 是否 monolithic 的 hack 注释(待 TRTLLM kernel 并入 MK framework 后移除);gate 权重仍为未量化 bf16(`quant_config=None`)。

## 关联

无(本批索引中无其他 DeepSeek router/MoE gate 前后序 PR;来源链:TRTLLM dsv3RouterGemm → sglang#7627 → 本 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
