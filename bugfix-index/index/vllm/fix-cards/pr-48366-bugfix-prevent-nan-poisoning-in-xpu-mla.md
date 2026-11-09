# PR #48366: [Bugfix] Prevent NaN poisoning in xpu_mla_sparse for fully-masked index chunks

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48366](https://github.com/vllm-project/vllm/pull/48366) |
| 作者 | nickus |
| 合并日期 | 2026-07-27 |
| labels | bug, intel-gpu, ready, v1 |
| 热度 | 👍 2 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
XPU 稀疏 MLA kernel 遇到行首 BLOCK_N 个索引全被 mask 时 `-inf-(-inf)=NaN` 永久污染累加器;运行最大值与被 mask logits 改用有限哨兵 -1e30 修复。

## 问题现象
修复 #48364:`_bf16_mla_sparse_kernel`(XPU_MLA_SPARSE 后端、DeepSeek-V4 XPU prefill 及 fp8 decode wrapper 背后的 XPU 稀疏 MLA kernel)在一行的前 `BLOCK_N`(=16)个 topk 索引条目全部被 mask 时,即使后面还有有效 key,输出也会被 NaN 污染。整行无任何有效 key 时同样返回 NaN(`0/0`),而库内参考实现 `reference_mla_sparse_prefill` 明确定义此时应输出零。现有测试未发现此 bug,因为旧测试把有效索引放在行首、用 -1 在尾部 padding——尾部 padding 是无害的,触发不了行首全 masked 的路径。

## 根因
PR 正文披露:
- running max(`e_max`)初始化为 `-inf`,首块全被 mask 时该块每个 logit 都被置为 `-inf`;
- 于是 `re_scale = exp2(-inf - (-inf)) = NaN`,`NaN * 0 = NaN`,永久污染 `acc` 与 `e_sum`,后续有效块也无法恢复该行;
- 整行无有效 key 时 `e_sum` 保持 0,输出为 `0/0 = NaN`,与参考实现的"零"约定不符。

## 修复方案
`vllm/v1/attention/ops/xpu_mla_sparse.py` 中 `_bf16_mla_sparse_kernel` 两行改动:running-max 初始化与被 mask 的 logits 由 `-float("inf")` 换成有限哨兵 `-1.0e30`(与 #47629 较新的 sparse-MLA kernel 采用同一方案):

```diff
-    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
+    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
```

```diff
-        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, -float("inf"))
+        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, -1.0e30)
```

正文论证该方案的等价性:混有有效/被 mask key 的 chunk 不受影响(fp32 下 `exp2(-1e30 - m)` 恰为 0.0);行首全 masked chunk 在首个有效 chunk 到达时被精确 rescale 掉;无有效 key 的行输出零,匹配参考语义。新增回归测试 `test_bf16_triton_sparse_mla_masked_chunks`(覆盖行首+行尾 masked chunk 与全 masked 行,经 Triton interpreter 双向验证:修复前 FAIL 产 NaN,修复后与参考在既有容差内一致,max |d| = 0.0075 为 bf16 存储舍入)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是 XPU 上索引含行首 -1 padding(全 masked 前导块)的稀疏 MLA prefill/decode 路径。
- 回归面:改动为纯数值路径的哨兵值替换(两行),无性能变化(正文声明 no perf change);对首块即含有效 key 的行,哨兵路径计算出的 `p`/`re_scale` bit 级一致(正文断言现有测试不受影响);无有效 key 的行 `lse`/`max_logits` 为大负有限值而非参考的 ±inf 占位(测试只比较输出),依赖这些占位值的下游需留意。

## 关联
- https://github.com/vllm-project/vllm/issues/48364
- https://github.com/vllm-project/vllm/pull/47629(较新 sparse-MLA kernel,同方案来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
