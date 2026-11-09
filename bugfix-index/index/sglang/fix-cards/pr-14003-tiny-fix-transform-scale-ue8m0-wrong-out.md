# PR #14003: Tiny fix transform_scale_ue8m0 wrong output in some scenarios

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14003](https://github.com/sgl-project/sglang/pull/14003) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-12-01 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`transform_scale_ue8m0` 在某些场景输出错误;改用从 DeepGEMM tests 拷贝的 torch 实现替换 DeepGEMM 库函数调用,并让启动期一致性检查固定走 torch 实现。

## 问题现象
正文未描述具体现象,Motivation 仅一句:「may dig further later if having time, but anyway this fixes the issue now」。从标题可知:`transform_scale_ue8m0` 在某些场景(scenarios)产生错误输出。该函数位于 `python/sglang/srt/layers/quantization/fp8_utils.py`,用于把 FP8 量化 scale 转换为 UE8M0 打包、TMA 对齐的张量布局(注释标明原实现 copy 自 DeepGEMM),影响走 DeepGEMM/UE8M0 scale 的 FP8 量化路径。

## 根因
正文未披露根因,作者明确表示之后再深挖。(由 diff 推断)问题出在直接调用 `deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor` 这一外部实现上——某些输入场景下它产出错误结果;修复思路是不再依赖该库函数,改用从 DeepGEMM tests 中拷贝的等价 torch 实现 `_get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl`(先把 FP32 scale 的指数位右移打包成 UE8M0 uint8,再填充对齐、按 int 重新解释、转置),并通过 `use_torch_impl` 开关切换。

## 修复方案
改 `python/sglang/srt/layers/quantization/fp8_utils.py`:
1. `transform_scale_ue8m0` 新增 `use_torch_impl: bool = False` 参数,按开关选择 torch 实现或原 DeepGEMM 库函数;
2. 新增 `_get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl`(注明 Copied from DeepGEMM tests);
3. `inverse_transform_scale_ue8m0` 的一致性检查改为 `transform_scale_ue8m0(sf_fp32, mn=mn, use_torch_impl=True)`,即反向变换校验固定使用 torch 实现,同时 assert 失败信息补充打印 `sf_fp32`。关键 diff 摘录:

```diff
-def transform_scale_ue8m0(sf, mn):
+def transform_scale_ue8m0(sf, mn, use_torch_impl: bool = False):
...
-    sf = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
+    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
...
-    sf_packed_recreated = transform_scale_ue8m0(sf_fp32, mn=mn)
+    sf_packed_recreated = transform_scale_ue8m0(sf_fp32, mn=mn, use_torch_impl=True)
```

## 影响与回归风险
- 回归面:仅 `fp8_utils.py` 的 UE8M0 scale 变换路径。默认 `use_torch_impl=False` 时 `transform_scale_ue8m0` 行为不变(仍走 DeepGEMM 库函数);唯一行为变化是 `inverse_transform_scale_ue8m0` 的启动期一致性检查改用 torch 实现重建再比对,且 assert 信息更详细。若 torch 实现与库实现在某些输入上不一致,该检查会在启动期报错(这正是检查目的)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
