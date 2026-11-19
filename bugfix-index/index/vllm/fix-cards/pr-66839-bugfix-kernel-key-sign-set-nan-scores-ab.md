# PR #66839: [Bugfix][Kernel] Key sign-set NaN scores above +inf in the top-k kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66839](https://github.com/vllm-project/vllm/pull/66839) |
| 作者 | lucamotz |
| 合并日期 | 2026-11-17 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
CUDA top-k 内核中 FP16 分箱与 FP32 选键对带符号位的 NaN 排序不一致(`cvt.rn.f16.f32` 丢弃 NaN 符号),导致选错 top-k 索引、向 `sampled_topk` 输出行之后越界写入至少 64 KiB、以及 `cooperative_topk` 恢复路径非法内存访问;修复把带符号 NaN 的键排到 +inf 之上,与 `torch.topk` 的顺序一致。

## 问题现象
在 `main` 上,一行 score 中含足够多带符号位 NaN 时出现三类故障(作者在验证 #66840 时发现):fp32 `persistent_topk`(radix、FilteredTopK、sampled 路径)选错 top-k 索引;`sampled_topk` 向输出行 `sampled_topk` 之后写入至少 64 KiB(80 行 × 100000 列、top_k 2048、末行 8000 个 NaN 时,把输出后的全部 64 KiB 哨兵覆盖);`cooperative_topk`(1 行 4096 个 NaN)触发 illegal memory access。新增测试 `test_topk_nan_ordering_matches_torch` 在 `main` 上 12 个 fp32 `persistent_topk` 用例(16/48/128 行)失败,本 PR 72 例全过。

## 根因
正文披露根因:`cvt.rn.f16.f32` 会丢弃 NaN 的符号位——在 B200 上 0xFFC00000 转成 0x7FFF,bf16 路径的打包转换同样如此。于是 FP16 粗分箱把每个 NaN 都放到最顶端,而 `convert_to_uint32_v2` 却把带符号位的 NaN 键排在 -inf 之下(原实现按符号位翻转)。凡是"FP16 分箱 + FP32 键选择"的路径都会自相矛盾:

- `sampled_topk` 的精确回退选出一个低于所有有限键的截断值,几乎保留整行 → 越界写;
- `cooperative_topk` 的单键恢复把每个有限 score 都当成高于 pivot → 非法内存访问。

## 修复方案
改 `csrc/topk_histogram_4096.cuh` 的 `convert_to_uint32_v2`,用大小比较替换符号位测试:

```cuda
// cvt.rn.f16.f32 drops a NaN's sign, so the FP16 coarse bins put every NaN
// on top; key sign-set NaNs above +inf too.
return static_cast<int32_t>(bits) > static_cast<int32_t>(0xFF800000u)
           ? (bits | 0x80000000u)
           : ~bits;
```

非 NaN 键完全不变,两种符号的 NaN 都落在 +inf 键之上,即 `torch.topk` 使用的顺序。sm_100/sm_120 上部分键位点多一条指令。测试新增 `nan_sign` 参数,负号用例在 ROCm 上跳过(见影响)。测试中 bf16 转换会丢 NaN 符号位,故在 bf16 cast 之后按位写入 `-1`。

## 影响与回归风险
- 性能:B200 上对比 `main`,geomean 0.999(`--backend persistent`)与 1.000(默认后端),没有任何配置慢于 0.8%。
- 正确性:`main` 的 `test_top_k_per_row.py` 在本 PR 构建上 885 passed / 45 skipped,与 `main` 相同;逐行 top-k 行为对非 NaN 输入不变。
- 平台限制:ROCm 保留旧键(新测试用例在 ROCm 跳过),作者未验证 ROCm 的 FP16 转换对 NaN 符号的行为,即 ROCm 上该不一致仍可能存在,属遗留风险。
- 作者自述当前版本的测试在 bf16 cast 之后置位这一部分他本人未运行,由 H100 CI job 覆盖。

## 关联
- 作者在验证 #66840 时发现本问题。
- 涉及文件:`csrc/topk_histogram_4096.cuh`、`tests/kernels/test_top_k_per_row.py`;验证另覆盖 `csrc/{topk,cooperative_topk,sampler}.cu`(main `4e3463c874` 对照)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
