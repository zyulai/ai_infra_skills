# PR #10928: [BugFix][Kernel]: fix illegal memory access in causal_conv1d when conv_states is None

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#10928](https://github.com/vllm-project/vllm/pull/10928) |
| 作者 | xffxff |
| 合并日期 | 2024-12-07 |
| labels | ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
causal_conv1d 在 conv_states 未提供时内核对 nullptr 写值导致非法内存访问,加空指针判断修复。

## 问题现象
Issue #10895 报告:在 H100 GPU 上,以特定输入形状调用 `causal_conv1d_fn` 且不提供 `conv_states` 时,触发 CUDA illegal memory access 错误。影响使用 causal_conv1d 内核(如 Mamba 类模型)且不传初始状态/conv_states 的调用路径。

## 根因
正文披露根因:调查发现该错误的根因是内核「attempt to assign a value to a `nullptr`」——当 `conv_states` 为 None 时,`causal_conv1d_fwd_kernel` 中处理 `final_state_position < 0 && seqlen > kWidth` 分支的代码仍会执行从 smem_exchange 加载并写值的逻辑,而该分支本应只在 `conv_states` 存在时才有意义,空指针被当作写入目标。

## 修复方案
改 `csrc/mamba/causal_conv1d/causal_conv1d.cu` 的 `causal_conv1d_fwd_kernel`:在进入「从 smem_exchange 加载数据以组装最终状态」的条件分支前增加 `conv_states != nullptr` 判断,`conv_states` 为空指针时跳过该分支,不再对 nullptr 写值。

关键 diff 摘录:
```diff
-        if (final_state_position < 0 && seqlen > kWidth){
+        if (conv_states != nullptr && final_state_position < 0 && seqlen > kWidth){
             input_t vals_load[kNElts] = {0};
```
同时在 `tests/kernels/test_causal_conv1d.py` 增加 `has_initial_state` 参数化(True/False),覆盖 `initial_states=None` 的调用路径并调整断言仅在传入初始状态时校验 final states。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:仅影响 CUDA 内核 `causal_conv1d_fwd_kernel` 的一个条件分支;`conv_states` 非空的原有行为不变(条件只是多了非空判断),`conv_states=None` 的路径从崩溃变为跳过最终状态写出。测试改动只覆盖 `tests/kernels/test_causal_conv1d.py`。

## 关联
[#10895](https://github.com/vllm-project/vllm/issues/10895)(正文引用的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
