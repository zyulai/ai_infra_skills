# PR #47994: [Fix] moe: Revert the JIT rewrite of `moe_finalize_fuse_shared` (#47784)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47994](https://github.com/sgl-project/sglang/pull/47994) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-11-03 |
| labels | jit-kernel |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
#47784 的 JIT 重写内核在 b200 上致 GSM8K 掉点(0.95→0.90),整体回滚恢复旧内核,保留 math.cuh 辅助函数。

## 问题现象
`test_dsa_glm52_hisparse.py`(GLM-5.2-NVFP4,1P1D,HiSparse + MTP)在 8-gpu-b200 上 GSM8K 从 ~0.95 跌到 ~0.90,低于 0.935 阈值;该测试仅夜间跑,PR CI 未覆盖,故 #47784 合入时未被发现。仅在 8-gpu-b200 复现,8x B300 同提交通过(0.942-0.950)。bisect 确认 `ca819c11cd`(#47784)开始连续失败,回滚本分支恢复 0.950。

## 根因
正文以「Suspected cause」章节给出(作者自述为代码阅读结论,未经针对性运行确认):新内核在 `routing_ready=True`(默认值,`finalize_flashinfer_trtllm_deferred_output` 即传此值)时,在 `PDLWaitPrimary` 之前就加载 routing 张量**和 `shared_output` 行**;旧内核虽也在 `cudaGridDependencySynchronize()` 前暂存 routing 张量,但 `shared_output` 只在其后才读——这个提前读取是 #47784 唯一的顺序改动。`DeepseekV2MoE` 中 `shared_output` 由 shared expert 在 `alt_stream` 上产出,主 stream 在 finalize 启动前 `wait_stream` 汇合;怀疑在 CUDA graph capture 且启用 PDL 时,对 shared expert 最后一个内核的依赖被一并放松,finalize 会读到尚未写完的行(routing 张量来自同 stream 更早内核,提前读原本安全)。与症状吻合:不崩溃、仅掉几个点精度、只在一种 GPU 型号上因时序暴露竞态。

## 修复方案
按字节恢复 #47784(`ca819c11cd`)之前的实现:
- 新增(恢复)`python/sglang/kernels/jit/csrc/moe/moe_finalize_fuse_shared.cu`(417 行,fork 自 flashinfer finalizeKernel),删除 JIT 重写的 `moe_finalize_fuse_shared.cuh`;
- `python/sglang/kernels/ops/moe/moe_finalize_fuse_shared.py` 包装器恢复旧签名(去掉 `out`/`routing_ready` 参数与逐参数 JIT 模板);
- 删除新增的单元测试与 `bench_moe_finalize_fuse_shared.py`;
- 保留 #47784 加进 `sgl_kernel/math.cuh` 的 `fma_f32x2`/`mul_f32x2`/`add_f32x2`(`kimi_k3/attn_res/fused_tma.cuh` 已在使用)。
正文注明除回到旧 finalize 内核外无其他行为变化。
注:pack 的 diff 在 40000 字符处按文件粒度截断,截断部分(约含 math.cuh、单测等文件)未展示,未编造。

## 影响与回归风险
- 影响窗口:正文未说明;按 bisect,#47784(`ca819c11cd`)合入后至本 PR 前的提交受影响。
- 回归面:使用 `moe_finalize_fuse_shared` 的 MoE finalize 路径(trtllm deferred 输出,如 DeepSeek/GLM/Qwen-Kimi 系 MoE)恢复旧内核与旧包装器;`routing_ready`/`out` 参数消失,若有调用方已适配新签名需注意;`sgl_kernel/math.cuh` 新辅助函数保留,不影响其使用方。正文 Relanding 建议未来重做时把 `shared_output` 读取移到 PDL wait 之后、并补充 side-stream + CUDA graph + PDL 的测试。

## 关联
- https://github.com/sgl-project/sglang/pull/47784 (被回滚的 PR)
- bisect/验证运行:https://github.com/sgl-project/sglang/actions/runs/41541068672 、https://github.com/sgl-project/sglang/actions/runs/41548940818 等(正文多处链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
