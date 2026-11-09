# PR #4231: fix per_token_group_quant_fp8 illegal memory when num_groups % 16 != 0

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4231](https://github.com/sgl-project/sglang/pull/4231) |
| 作者 | BBuf |
| 合并日期 | 2025-03-10 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
num_groups 非 16 倍数时 per_token_group_quant_fp8 kernel 越界;按整除性选每块组数并用 shuffle 归约重写修复。

## 问题现象
当 `num_groups % 16 != 0` 时,`per_token_group_quant_fp8` CUDA kernel 触发 illegal memory。正文未用文字描述现象,仅附 unit-test 截图(含 test_mla.py 相关用例)与 benchmark 截图作为验证材料;具体报错日志正文未给出。

## 根因
正文未披露根因,仅以截图展示测试结果。

(由 diff 推断)旧 kernel 固定 `groups_per_block = 16`、`block(256)` 线程(256 = 16 组 × 16 线程),grid 按 `(num_groups + 15) / 16` 向上取整;当 `num_groups` 不是 16 的倍数时,块内线程组与实际 group 数不再对齐,标题指明此时发生非法内存访问。新实现改为按 `num_groups` 的整除性(16/8/4/2/1)选择每块组数,使块内线程组与 group 边界严格对齐,不再有越界路径,这与"必须整除才不崩"的现象吻合。

## 修复方案
重写 `sgl-kernel/csrc/gemm/per_token_group_quant_fp8.cu` 中的量化 kernel:

1. `per_token_group_quant_fp8` 按 `num_groups % 16/8/4/2` 选择 `groups_per_block`,否则为 1,并通过 `LAUNCH_KERNEL` 宏按模板参数 `GROUPS_PER_BLOCK` 配置 grid 与 block:

```diff
+  int groups_per_block = 1;
+
+  if (num_groups % 16 == 0) {
+    groups_per_block = 16;
+  } else if (num_groups % 8 == 0) {
+    groups_per_block = 8;
+  } else if (num_groups % 4 == 0) {
+    groups_per_block = 4;
+  } else if (num_groups % 2 == 0) {
+    groups_per_block = 2;
+  }
```

2. `GroupReduceMax` 由共享内存数组(`s_absmax[16][17]` + `__syncthreads`)改为 warp shuffle(`__shfl_xor_sync`,mask 0xffff)。
3. 组内读取/写入改为 `flashinfer::vec_t` 向量化加载(.vec_size = 16 / sizeof(T)),由 lane 0 写出 scale。
4. 测试 `sgl-kernel/tests/test_per_token_group_quant_fp8.py` 参数扩展:batch_size 增至 `[1, 2, 4, 8, 16, 32, 64, 128]`,group_size 增至 `[16, 32, 64, 128, 256]`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:sgl-kernel 中所有经 `sgl_per_token_group_quant_fp8` 的 FP8 组量化调用路径(如 MLA/FP8 GEMM 的输入量化)。kernel 每组 absmax 归约与缩放语义保持一致,但归约实现(shuffle 替代共享内存)、访存模式(向量化)与 launch 配置(按整除性分档)均改变;非 16 倍数场景 grid/block 尺寸随之变化,属于本次修复的核心路径。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
