# PR #53911: [Kernel] Fuse DeepEP-v2 decode globalize+align into one launch

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvircrn |
| 合并 | 2026-10-13 |
| 分类 | MoE(tier: major) |
| Label | ready, ci/build |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/53911 |

## 一句话结论

把 DeepEP-v2 INDEXED cudagraph-decode 的 globalize + moe_align_block_size + count_and_sort 三内核链融合为单次 CUDA launch,H200 EP32 decode 元数据准备最高提速 2.01x。

## 动机

DeepEP-v2 INDEXED cudagraph-decode 路径上,`_globalize_recv_topk_idx` + `moe_align_block_size` + `count_and_sort_expert_tokens` 三个内核串行执行,是每次 decode 的元数据准备开销;创建时无其他 PR 融合该链(gh 搜索验证)。

## 优化点

单内核 `fused_globalize_align_block_size`:`recv_topk_idx` 保存本地 expert id,内核按本地 id 分 bin(align/sort)同时把全局 id 原位回写;`num_recv` 直接在设备端从 `psum[-1]` 读取,不烘进 replay,CUDA-graph 安全;按本地专家数(而非全局)分配 sentinel/`expert_ids` 区域,EP 下缩小约 ep_size 倍;可选产出 Humming 大 EP batch 用的 scatter index。混合 launch:`effective_work ≤ 4096` 用单 block 非协作启动(`__syncthreads` 屏障),否则经 `cudaLaunchKernelEx` + cooperative 属性启动多 SM grid(`grid.sync()` 可被 graph 捕获)。选择条件:CUDA Humming INDEXED experts 且 DeepEP-v2 在 graph decode 中推迟 globalization(`FusedMoEKernelModularImpl` 对 `HummingIndexedExperts` 自动置 `prepare_finalize.defer_globalize`,跳过独立 globalize launch);其他路径保持原行为。

## 关键改动

```text
csrc/moe/fused_globalize_align_block_size.cu(新): 4 阶段融合内核(清零+sentinel → 分布式直方图 → block0 前缀+expert_ids → scatter+globalize)
vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py: prepare_humming_moe_kwargs 走融合路径
vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py: defer_globalize 跳过独立 globalize launch
vllm/model_executor/layers/fused_moe/fused_globalize_align_block_size.py(新): Python 包装与输出分配
vllm/model_executor/layers/fused_moe/modular_kernel.py: ExpertTokensMetadata.rank_expert_offset
```

## 性能声明

H200、EP32、INDEXED decode,融合 vs 三内核链(10 月 6 日 rebase 前测得):1 token 1.23x、8 token 1.14x、32 token 1.16x、64 token 1.36x、128 token 1.67x、256 token(n=8192)2.01x。作者注明 rebase 后未复测。

## 验证与回归风险

tests/kernels/moe/test_fused_globalize_align_block_size.py(新):与参考三内核链在 n/topk(1/6/8/16)/block_size(16–128)/EP rank/fill 分布网格上逐位一致,含 scatter index 与 CUDA graph 捕获/重放(两种 launch 路径)用例;`ruff`/`clang-format`/`git diff --check` 通过。风险与未完成项:rebase 后 `_moe_C` 重建、pytest 与模型评测(如 gsm8k,融合路径激活)待 CI;clang-format/mypy 钩子未在最终 base 重跑;`local_num_experts <= 1024` 限制;仅影响 CUDA Humming INDEXED 路径。

## 关联

同分类 MoE:#51808(在线 MXFP4 激活覆写)。区域:DeepEP-v2 / Humming INDEXED MoE decode。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
