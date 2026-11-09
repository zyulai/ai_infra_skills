# PR #6852: [Bug Fix] Illegal memory access, FP8 Llama 3.1 405b

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6852](https://github.com/vllm-project/vllm/pull/6852) |
| 作者 | LucasWilkinson |
| 合并日期 | 2024-07-27 |
| labels | ready |
| 热度 | 👍 5 · 💬 5 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
batch size 非 gemm tile 整数倍时 Sm90 epilogue 读越界 scales buffer 导致非法内存访问;加谓词张量用 copy_if 屏蔽越界读。

## 问题现象
标题所示为 FP8 Llama 3.1 405b 的 illegal memory access;正文指向 issue [#6767](https://github.com/vllm-project/vllm/issues/6767) 与 [#6689 的评论](https://github.com/vllm-project/vllm/issues/6689#issuecomment-2247666348)等报告,并给出触发条件:batch size 不是 gemm tile size 的整数倍时,`Sm90ColOrScalarBroadcast` 会读到 scales buffer 末尾之外。正文未写具体报错日志。

## 根因
正文披露:`Sm90ColOrScalarBroadcast`(CUTLASS SM90 W8A8 epilogue 的列广播/标量广播)在 batch size 不是 gemm tile size 整数倍时,会越界读取 scales buffer——epilogue 按 tile 粒度做 `copy_aligned` 全量拷贝,没有对超出问题 M 维度的行做任何谓词保护。

## 修复方案
`csrc/quantization/cutlass_w8a8/broadcast_load_epilogue_c3x.hpp` 的 `Sm90ColOrScalarBroadcast`:构造与全局张量同形、按同样方式 partition 的 identity 张量 `tCcCol`,在 `begin()` 里生成谓词张量,把越界行过滤掉,`copy_aligned` 换成 `copy_if`:

```cpp
Tensor pred = make_tensor<bool>(shape(tCgCol));
CUTLASS_PRAGMA_UNROLL
for (int i = 0; i < size(pred); ++i) {
  pred(i) = get<0>(tCcCol(i)) < m;
}
...
copy_if(pred, filter(tCgCol), filter(tCrCol));
```

其中 `m` 取自 `problem_shape`,`ConsumerStoreCallbacks` 相应新增 `tCcCol`、`problem_shape` 参数。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:改动位于 SM90 CUTLASS W8A8 的 broadcast-load epilogue,所有经由 `Sm90ColOrScalarBroadcast` 加载 scale/zero-point 的 FP8/INT8 GEMM(SM90 路径)都会走到新谓词逻辑;对 tile 整除的常规 shape,谓词恒真,行为不变(正文未逐一列举受影响调用方)。正文未附性能对比与专门回归测试说明。

## 关联
- [#6767](https://github.com/vllm-project/vllm/issues/6767)
- [#6689(评论)](https://github.com/vllm-project/vllm/issues/6689#issuecomment-2247666348)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
