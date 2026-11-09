# PR #62366: [Model][Perf] Fuse Beam's TP all-reduce with its sandwich norms and MoE finalize

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2026-10-22 |
| 分类 | MoE(major) |
| Label | documentation, new-model, speculative-decoding, ready, ci/build, nvidia, mrv2, dflash |
| Diff | diffstat 缺失(diff 全文已提供但裁剪了 10 个文件,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/62366 |

## 一句话结论

Beam 模型的 TP all-reduce 与 FP32 norm-residual-norm 尾板及 TRTLLM-Gen NvFP4 MoE top-k finalize 融合进一个 MNNVL kernel,hidden 8192 下 64 token 起比 JIT 版快 6-16%。

## 动机

Beam(新模型,GB200 MNNVL 场景)的 decode 图中,TP all-reduce 之后紧跟 sandwich 结构的 RMSNorm-residual-RMSNorm,TRTLLM-Gen NvFP4 MoE 层还需 top-k finalize;这些以独立 kernel 依次执行。此外 vLLM 自有 Lamport 原语在 GB200 上明显偏慢:其 all-gather 在 TP4 慢约 10%、TP8 慢 25-40%(64 tok × 5120:16.7 vs 11.9 µs),reduce-scatter+all-gather 对比 FlashInfer 单 launch two-shot 慢 1.3-1.8×,疑似逐个等待 peer 而非同时轮询所致。此前的融合 kernel 依赖 FlashInfer 运行时 JIT 构建。

## 优化点

新增 AOT 构建的 `csrc/beam/allreduce_norm.cu`(无运行时 JIT、模型目录无 CUDA 源),暴露 `torch.ops._C.beam_allreduce_norm` 与 `beam_allreduce`(纯 two-shot,大 batch 路径):one-shot/two-shot MNNVL Lamport all-reduce,在归约前可选执行 MoE top-k finalize(保持本地 BF16 rounding),归约后执行 FP32 norm-residual-norm 尾板(post norm、residual 加、pre-normed 输出、next norm 一体);线程块同时轮询所有 peer rank 而非逐个等待,按 rank 序求和保证确定性;小 batch 路径用常量索引避免 Lamport 状态溢出到 local memory,并用 programmatic stream serialization 与 cluster launch。FlashInfer 的 MNNVL Lamport 设备助手(`trtllm_mnnvl_allreduce.cuh`,v0.7.2)子集 vendor 进 `csrc/mnnvl_lamport.cuh`,Lamport 协议、初始化与 generation flags 归 vLLM 所有,构建不再依赖 FlashInfer 头文件与安装版本;multicast 内存仍用 FlashInfer `McastGPUBuffer`。Beam 借既有 `FusedMoEConfig.defer_moe_finalize()` 把 NvFP4 top-k finalize 推迟到 all-reduce 内做,不改共享 MoE 代码。仅 SM90/SM100 编译(`VLLM_ENABLE_BEAM_ALLREDUCE`),该文件 `--fmad=false` 保 FP32 边界数学不被收缩;op 未构建/multicast 不可用/形状不支持时退回标准 TP all-reduce。

## 关键改动

```text
csrc/beam/allreduce_norm.cu: beam_allreduce_norm / beam_allreduce(oneshot/finalizeScatter/finishTwoshot kernel 族)
csrc/mnnvl_lamport.cuh: vendor FlashInfer v0.7.2 MNNVL Lamport 助手子集
(其余 10 个文件——python 侧接线、workspace、测试等——在素材 diff 中被裁剪)
```

归约+尾板融合的 one-shot 节选:

```cuda
Vec value = loadInput<Finalize>(p, token, packedIdx);
// multicast 写入 peer payload,等全部 rank 就绪后按 rank 序确定性求和
value = reduceLocal<World, Rank>(value, local, token, p.ar.tokenDim, packedIdx);
epilogue<!SmallBatch, SmallBatch>(p, value, packedIdx, offset, ...);
```

## 性能声明

kernel 微基准(GB200,BF16,CUDA graph 内 20 次捕获,µs/次,取最慢 rank;"JIT"为同 kernel 的 FlashInfer-JIT 构建,"AOT"为本 PR):TP4 hidden 8192:256 tok 19.00→16.05,1024 tok 64.52→54.28,4096 tok 251.2→226.8;TP8 hidden 8192:256 tok 19.82→16.67,1024 tok 68.27→57.19,4096 tok 265.1→225.8(PR 称 hidden 8192、64 token 起快 6-16%);hidden 5120 与 1-8 token 基本持平。TP4 为 3 次中位数、TP8 为 2 次换序均值。端到端 before/after(对 #62074 标准 TP all-reduce,GB200 TP8 batch-1 decode 与 8×2K prefill)标注 TODO 未完成。

## 验证与回归风险

单测 GB200 4 卡:`test_beam_construction.py`/`test_beam_numerics.py`/`test_trtllm_nvfp4_moe.py` 共 178 passed / 3 skipped;TP8 跨两台 GB200(一个 MNNVL clique)以 `torchrun --nnodes=2` 跑 all-reduce 数值 worker(hidden 5120/8192、one/two-shot、大 batch、CUDA graph replay)通过;8-rank MoE finalize 用例未跑(测试为单节点)。FlashInfer 0.7.0.post1 环境(该头文件在 0.7.2 byte-identical)。风险面:仅覆盖 TP2/4/8、BF16、tokenDim 8 对齐;CI 中 GPU 用例在 CPU agent 上跳过;Hopper 未实测;hidden-8192 差距仅在 JIT 路径按 8192 token 尺寸的 workspace 上出现,原因未隔离。

## 关联

前序:#62074(已合并)、#62073(replicated-token EP 的 MoE finalize 延迟,扩展到 MegaMoE DP/EP);vendor 来源:FlashInfer `trtllm_mnnvl_allreduce.cuh` v0.7.2。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
