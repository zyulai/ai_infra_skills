# PR #22434: [CPU][sgl-kernel] `extend_attention_cpu` and `flash_attn_varlen_func`: fix `nan` for large seq

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22434](https://github.com/sgl-project/sglang/pull/22434) |
| 作者 | chunyuan-w |
| 合并日期 | 2026-04-17 |
| labels | sgl-kernel, run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
CPU attention 内核在长序列(>4096)下 causal mask 条件与负 `last_col` 处理错误,输出含 NaN 导致采样崩溃;改为按块边界掩码并钳位 `last_col`。

## 问题现象
向模型输入长序列(> 4096)时,`extend_attention_cpu` 与 `flash_attn_varlen_func` 内核输出 NaN,在采样阶段抛出:
```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```
(sampler.py `torch.multinomial` 处)。issue #20051 也报告了同一问题。修复前精度测试:`Mismatched elements: 281210 / 2560000 (11.0%)`,最大绝对差为 nan;另一用例 mismatch 7.0%。

## 根因
正文披露:当 max_len_extend > 4096 时内核选中 `BLOCK_M = 512`、`BLOCK_N = 768`,此时 `BLOCK_M > BLOCK_N/2`,首个 n-block 也可能包含未来 key(例:`m=512, num_keys=1024`,首块覆盖 keys [0,768),query row 0 位于绝对位置 512,keys 513..767 是未来 key)。原有掩码条件 `num_keys - n <= BLOCK_N` 只对最后一个 n-block 掩码,导致首块中 [513,767] 未被掩掉,结果非因果、数值错误;而对第二个 n-block,早期行可能整块为未来 key,`last_col` 为负,原代码会写到 `row_ptr` 之前(越界写),产生 NaN。

## 修复方案
改 `sgl-kernel/csrc/cpu/extend.cpp`(`extend_attention_kernel_impl`)与 `sgl-kernel/csrc/cpu/flash_attn.cpp`(`flash_attn_kernel_impl`、`flash_attn_varlen_kernel_impl`)三处 causal mask 逻辑:
1. 掩码条件由 `num_keys - n <= BLOCK_N` 改为 `n + n_size - 1 > m`(块的最后一个 key 晚于首个 query 位置即掩码);
2. `last_col = std::max(last_col, -1)`,整块为未来 key 时掩掉整行,消除负索引越界写。

```diff
-        if (num_keys - n <= BLOCK_N) {
+        if (n + n_size - 1 > m) {
           for (int row = 0; row < m_size; ++row) {
             int last_col = m + row - n;
+            last_col = std::max(last_col, -1);
```
并在 `test/srt/cpu/test_extend.py`、`test/srt/cpu/test_flash_attn.py` 新增大序列 causal mask 用例(非 varlen 与 varlen 两条路径)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(只说输入 > 4096 触发)。
- 回归面:CPU 后端 attention 内核的 causal mask 路径(extend.cpp 与 flash_attn.cpp 的两个 kernel 实现;`flash_attn.cpp` 中非 causal 分支因条件仍含 `causal &&` 不受影响)。对 `BLOCK_M <= BLOCK_N/2` 的旧配置,新条件与旧条件等价,行为不变;主要风险在掩码条件改动对其他 BLOCK 组合下边界行为的影响,已由新增大序列测试覆盖。

## 关联
- https://github.com/sgl-project/sglang/issues/20051

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
