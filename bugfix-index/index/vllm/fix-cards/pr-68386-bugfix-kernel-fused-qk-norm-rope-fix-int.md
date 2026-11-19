# PR #68386: [Bugfix][Kernel] fused_qk_norm_rope: fix int32 offset overflow and NeoX partial-rotary corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#68386](https://github.com/vllm-project/vllm/pull/68386) |
| 作者 | simon-veitner-redhat |
| 合并日期 | 2026-11-19 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
`fused_qk_norm_rope` 内核存在四处缺陷:QKV 行偏移用 32 位 int 计算会在大 `num_tokens` 时溢出(非法内存访问)、部分旋转的 NeoX 路径 warp 集合操作用全掩码但只有部分 lane 进入分支导致跨头 RMS 混叠、非 2 的幂 pairOffset 时 `laneId ^ pairOffset` 配对出错、以及 cp.async 等待后缺 warp barrier;逐项修复后 GLM-5.3-Flash 视觉编码器在 DP4+EP H200 上可正常启动。

## 问题现象
GLM-5.3-Flash 在 DP4 + EP、H200 上带视觉编码器启动时,`main` 在 encoder profiling(对 960,000 patches 跑 ViT)中触发 illegal memory access 而崩溃(`RuntimeError: Worker failed with error 'CUDA error: an illegal memory access was encountered'`),修复后 126 秒内正常就绪。内核层面:`token_heads_per_warp` 为 2/4/8 及 auto 时,部分旋转场景与单头内核相比 main 有 33-70% 元素不同、F=8 时输出 NaN;N=699,100 与 969,000 时直接 IMA。测试对照:main 上 `large_num_tokens` 2 例失败(IMA),`matches_reference` 16 例失败(F=4 NeoX 比率 0.5/0.25/0.375 及 F=1 NeoX 比率 0.375),修复后全部通过(66/66)。

## 根因
四个根因(均为正文披露,每条一个 commit):

1. **32 位偏移溢出**:两个内核都用 `tokenIdx * num_heads * head_dim`(int)计算行偏移,48 头 × 64 维时 token 699,051 的行起点超过 INT_MAX。
2. **部分 NeoX 损坏**:只有 `laneId < rotary_lanes` 的 lane 进入 NeoX 块,但块内 `__syncwarp` 与 `__shfl_xor_sync` 用全掩码;sm_70+ 上 `shfl.sync` 会等待掩码内其他 lane 执行同限定符、同掩码的 `shfl.sync`(PTX ISA 9.7.10.6),跳过分支的 lane 与下一个头的 `warpReduceSum` shuffle 相遇,把不同头的 RMS 求和混在一起。
3. **配对 lane 错误**:`laneId ^ pairOffset` 仅在 pairOffset 为 2 的幂时等价于 `laneId ± pairOffset`;rotary_dim 48/128 时 pairOffset 为 6,XOR 不再是简单加减。
4. **cp.async 可见性**:多头内核的 GPT-J 路径在 `cp_async_wait_group<0>()` 之后无 warp barrier 就读其他 lane 拷贝的 cos/sin(PTX ISA 9.7.10.28.3.3:cp.async 写在等待后仅对其自身线程可见)。

## 修复方案
1. 进入内核后一次性用 64 位数学把 `qkv` 移到 token 行:`qkv += static_cast<int64_t>(tokenIdx) * num_heads * head_dim;`,每头偏移保持行内相对 int(全部提升为 int64 的 #43166 方案实测有 4-6% 开销)。
2. 所有 lane 都进入 NeoX 分支,仅旋转 lane 写回结果(`if (laneId < rotary_lanes)` 包住 RoPE 混合),非旋转 lane 的 cos/sin 索引保持在 `rotary_dim / 2` 以下,读取在界内;原有 `__syncwarp` 现在全 lane 执行,同时理顺了 cp.async 拷贝的可见性。
3. 配对改为显式源 lane:`__shfl_sync(FINAL_MASK, elements[i], laneId < pairOffset ? laneId + pairOffset : laneId - pairOffset, 32)`;保留 width 32,因为该文件同时为 ROCm 构建,wave64 上源 lane 必须留在 32-lane 组内。
4. 多头内核在 `cp_async_wait_group<0>()` 后补一个 `__syncwarp()`。
测试:参考测试新增 `token_heads_per_warp=4` 与 rotary 比率 0.375;新增大 N 用例(约 4.5 GiB,空闲显存低于 12 GiB 跳过)覆盖 bug 1,运行于 H200 MIG 35GB Core Operation Kernels job。

## 影响与回归风险
- 输出正确性:GLM ViT(rot 64/32)与 Ling 形状(16q/4kv × 128, rot 64)在 F=1/2/4/8、N=4,096 与 700,000 下与 F=1 位级一致;全旋转与 GPT-J 输出与 main 位级一致。
- 性能:全旋转无减速;rot 32 下本 PR 反而明显更快(8,192 auto:91.1 µs → 50.5 µs,且 main 数值是错的)。
- 遗留与限制:ROCm 未在本 PR 测试;bug 2 与 #50903 的修法重叠,作者指出 #50903 把 shuffle 移出 guard 并去掉 `__syncwarp`,会留下 cos/sin 读取无序的问题,本 PR 采用不同方案;本 PR 取代过期的 #43166(bug 1)。
- 测试改动:`tests/kernels/core/test_fused_qk_norm_rope.py`(kernel +37/-21,tests +45/-1)。

## 关联
- Fixes #51049(NeoX 部分旋转损坏,Ling 形状复现);supersedes #43166(bug 1);overlaps #50903(bug 2,方案不同)。
- 涉及文件:`csrc/fused_qknorm_rope_kernel.cu`、`tests/kernels/core/test_fused_qk_norm_rope.py`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
