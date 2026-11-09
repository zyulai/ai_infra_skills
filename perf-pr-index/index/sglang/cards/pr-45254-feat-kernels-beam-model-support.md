# PR #45254: [Feat] kernels: Beam model support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Jiminator |
| 合并 | 2026-10-26 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, dependencies, blackwell, jit-kernel |
| Diff | +1364/-34,21 files(PR 正文声明;pack diffstat 字段缺失,diff 本身可用但 71 个文件因超长被截断) |
| 链接 | https://github.com/sgl-project/sglang/pull/45254 |

## 一句话结论

为 Beam 模型引入 fused kernel 集(attention gate、sandwich norm、TP8 all-reduce+norm 融合、FP32 router GEMM/GEMV),其中 TP8 5-CTA cluster all-reduce 在 H200 上按 topk 快 4-16%。

## 动机

将大 PR #44747 拆分为可评审的 7 个部分,本 PR 是第 3 部分(kernels)。Beam 模型(reflection/Beam-1.0-501B-A23B)需要一组专属 kernel:带 activation-dtype 偏置与 sigmoid 舍入语义的 per-head attention gate、单 launch 的 post-norm + depth-scaled residual + 下一个 RMSNorm、以及 BF16 TP all-reduce 与 norm/residual 的融合,以支撑 Day-0 模型支持;另有 FP32 router gates 的数值精度需求。

## 优化点

四组 kernel:(1) `elementwise.beam_attention_gate`(Triton,注册 torch/triton 双 backend)——per-head gate `attention * sigmoid(gate + bias[head])`,逐处保留 eager 的 activation-dtype 舍入边界(FP16 用 `div_rn` 近似 sigmoid,torch.compile 后端加 `emulate_precision_casts`)。(2) `layernorm.beam_sandwich_residual_rmsnorm`——post-norm、两个 scaled residual 项、合并流、可选 next norm 在一次 launch 内完成,保留各自独立的 BF16 舍入边界,支持 PDL。(3) `communication.beam_all_reduce_norm`(JIT CUDA,csrc `beam_all_reduce.cuh`)——Beam 的 BF16 TP all-reduce 与 sandwich norm/residual 融合,跑在 CustomAllReduceV2 push plane 上(unicast/multicast 双模式,Lamport pos-zero 同步,PDL),TopK=8 时 FP32 求和 expert 输出再乘 routed_scale 舍入;TP8 且行数 <=16 时每行用 5-CTA cluster launch(否则 1 CTA/行),需新增 `CustomAllReduceV2.supports_push` 能力检查。(4) FP32 router gates:main 的 `tiny_gemm` N 变体加 FP32 weight 模板参数(dot_product 增加 bf16 激活 x FP32 权重的 FP32 累加),`router_gemv` 在 CUDA 上支持 FP32 权重(上限 8 行),更大 batch 用 `gemm.router_gemm_bf16x3_cublas`(按调用 `split_fp32_to_bf16x3` 拆权,BF16 tensor core 三分量求和)。既有调用方(BF16 tiny GEMM、gfx950 BF16 router GEMV)行为不变,JIT build key 加权重类型导致首次使用重编译一次。SM103 单 token BF16 MoE GEMV 与共享 SiLU helper 移至 draft #46401。

## 关键改动

```text
python/sglang/kernels/jit/csrc/distributed/beam_all_reduce.cuh: 新增 BeamAllReduceNorm(all-reduce + inv_rms + scaled residual + next norm,push plane)
python/sglang/kernels/ops/communication/beam_all_reduce.py: JIT wrapper;_cluster_size 按 TP8/行数选 5-CTA 或 1-CTA
python/sglang/kernels/ops/elementwise/beam_fused.py: 新增 fused_attention_gate Triton kernel 与 BeamAttentionGateOp
python/sglang/kernels/ops/layernorm/beam_fused.py: 新增 sandwich residual RMSNorm 融合 kernel
python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh 与 ops/gemm/tiny_gemm.py: tiny N GEMM 支持 FP32 权重
python/sglang/kernels/ops/gemm/router_gemv.py: FP32 router GEMV(CUDA)+ router_gemm_bf16x3_cublas
```

关键片段(cluster 选择策略):

```python
def _cluster_size(world_size: int, rows: int) -> int:
    # Measured on 8xH200: 5-CTA clusters are faster at TP8 up to 16 rows
    return 5 if world_size == 8 and rows <= 16 else 1
```

## 性能声明

TP8 cluster launch(每行 5-CTA cluster vs 1 CTA/行,per call;出自 PR "earlier revisions" 的历史验证部分,后续 review commit 未重跑):

- 8xH200:top-k 1 时快 4-10%,top-k 8 时快 8-16%(单行 6.76 vs 5.68 us)
- 8xGB300 跨两节点(MNNVL):top-k 8 快 3-11%;top-k 1 从快 2.2% 到慢 0.9%(五次扫描中位数),在噪声内
- TP2/TP4 保持 1 CTA/行(在 H200/GB300 上一样快或更快);TP8 超 16 行也保持 1 CTA/行;32 行 top-k 8 时 cluster 在 H200 快 1.4% 但在 GB300 慢 3.4%

H200 上 FP32 router:boundary norm + FP32 tiny GEMM 比更早的 PDL 顺序/拆分选择每层快 0.02-1.26 us(1-16 token),输出 bitwise 一致。e2e(与 #45256/#45257/#43216 组合):GB300 4 卡 NVFP4 checkpoint GSM8K 96.59%;8xH200 FP8 TP8 checkpoint GSM8K 96.21%(sgl-eval,1319 题,temperature 0)。

## 验证与回归风险

针对 review commit(`851658f9b8`,GB300/SM103,torch 2.14.1+cu130,Triton 3.8.0):196 个 tiny GEMM、5 个 router、13 个 fused-op parity 用例通过(新增 18 个数值用例,FP32 router 对 FP64、gate 精确舍入、BF16 norm 一 ULP、FP16 cancellation 界);collective 测试过 TP2/TP4/跨节点 TP8(MNNVL,unicast+multicast,graph replay 中变更 payload,TP8 两种 launch shape);42-tensor-per-rank probe 字节级零差异;memcheck TP2/TP4 到 64 行通过(TP8 的 memcheck 无声明,racecheck/synccheck 全矩阵通过,但依赖 `--report-api-errors no`,PR 自述并非干净的 CUDA API 验证);通用 custom-all-reduce TP4 套件 309 过 1 失败(该失败在未改动的 baseline 上可复现)。未在 H200/ROCm/SM120 上为 review commit 重跑。e2e 曾过 Beam Verifier 133/133。回归面:CustomAllReduceV2 push 路径新增能力检查与 cluster launch 分支;`--fmad=false` 编译选项;kernel 语义与 eager 的舍入边界逐位对齐是该设计的核心约束;CI 状态显示 Base/AMD ROCm 测试失败标记。

## 关联

系列拆分:前序 #45253(Beam CUDA Docker)、#45291(OpenAI SDK);后续 #45256(attention/memory/Beam model)、#45257(reasoning/tool calls/E2E test)、#45361(cuTile BF16x3 router GEMM)、#46401(SM103 GEMV draft);总参考 #44747;模型 E2E 测试在 #45257;#45361 直接 import 本 PR 的 `split_fp32_to_bf16x3`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
