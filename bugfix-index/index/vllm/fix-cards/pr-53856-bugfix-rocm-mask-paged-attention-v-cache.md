# PR #53856: [Bugfix][ROCm] Mask paged attention V cache padding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53856](https://github.com/vllm-project/vllm/pull/53856) |
| 作者 | aoshen02 |
| 合并日期 | 2026-09-08 |
| labels | bug, rocm, ready |
| 热度 | 👍 2 · 💬 25 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论

ROCm paged attention 的向量化 V-cache 加载会读到末块未写槽位,NaN 经 0*NaN 传入输出;仅对越过 seq_len 的部分尾向量做掩码修复。

## 问题现象

ROCm paged attention 对超出 `seq_len` 的 token 做了 logit 掩码,但向量化的 V-cache 加载仍可能消费最后一个 block 中未写入的槽位。当这些槽位恰好含 NaN 时,后续 `0 * NaN` 累加把 NaN 传播进输出。作者直接对 kernel 做测试:给每条序列一个私有末块并把超出逻辑长度的槽位投毒(auto 路径用 cache dtype 的 NaN,FP8 用架构对应的 E4M3 NaN 表示),基线在 `auto` 与 FP8 两种 cache dtype 下均复现 `Mismatched elements: 10240 / 14336 (71.4%)、Greatest absolute difference: nan`。触发条件是序列逻辑长度未对齐 block 末尾且 padding 槽位为垃圾值;作者强调修复不依赖 sleep mode、分配器行为或随机物理块重叠。

## 根因

正文明确披露:分数掩码(score masking)对 V 的未写 padding 不够用,因为 `0 * NaN` 仍是 NaN——无效 K 值会在 softmax 前被剔除,所以 K 不需要同样处理,但 V 必须在矩阵累加前被清理。该 PR 沿用了早年 CUDA paged attention 的同一防护(#936 / #641)以及现行 Triton paged attention 的做法。

## 修复方案

保留向量化加载,只处理部分尾部(partial tail),`csrc/rocm/attention.cu` 三条路径:

1. gfx9 MFMA16:在既有消费循环中跳过完全无效的 V 向量、仅掩码跨越 `seq_len` 的那个向量;`auto` 与 FP8 路径保持编译期分离;`__launch_bounds__` 从 5 波改为 4 波,避免新增边界状态导致 scratch 溢出(实测该敏感场景从 96 VGPR/16B scratch 变为 104 VGPR/0 scratch)。
2. gfx9 MFMA4:整 warp 热路径不变;部分尾 warp 中跳过完全无效的 8-token 向量,仅掩码边界向量(新增 `SV_mfma_tail` 宏与 `mask_16b_v_cache_boundary`/`mask_8b_v_cache_boundary` 辅助函数)。
3. gfx11/gfx12:在 V 消费点统一调用共享的打包向量 padding 辅助函数 `mask_v_cache_padding`(核心思想如 diff 注释:"Score masking is insufficient for unwritten V padding: 0 * NaN is NaN")。

测试侧改 `tests/kernels/attention/test_attention.py`:块表保证每序列的最后一个物理块私有,并把越界槽位投毒为 NaN,同时保留非默认 K/V scale。修复后 `199 passed, 2 deselected`,覆盖 192 组 BF16/FP16、block 16/32、head 64/128、ALiBi、GQA 1/4/8 组合;gfx1100 实机验证修好全部 4 个既有 V-poison 失败。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,改动集中在 ROCm paged attention kernel 的 V 消费路径:完整块(无越界)走 `__builtin_expect` 预测的快速分支不变;部分尾处理带来约 2-3.5% 的直接 kernel 粒度开销(高精度 A/B 44 例中 0 例回退超 5%,最差 +3.48%,作者明确不声称零回退);MFMA16 的 launch bound 5→4 会改变占用/资源画像。gfx942 与 gfx1200 仅有编译验证,运行时验证在 gfx950 与 gfx1100。HunyuanOCR 等 gfx1100 工作负载在当前 GQA 门下不经由此 kernel,单独本 PR 对其字节无影响,叠加 #54210 放宽门控后进入修复路径。

## 关联

- https://github.com/vllm-project/vllm/pull/936 (早年 CUDA paged attention 同类防护)
- https://github.com/vllm-project/vllm/issues/641 (同上)
- https://github.com/vllm-project/vllm/pull/54210 (互补:放宽 gfx11 GQA 门控)
- https://github.com/vllm-project/vllm/issues/50603#issuecomment-5465499491 (HunyuanOCR 数据)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
