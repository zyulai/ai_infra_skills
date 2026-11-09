# PR #7604: Fix quant kernel test errors and benchmark wrong output speeds

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7604](https://github.com/sgl-project/sglang/pull/7604) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-08-21 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
per-token-group quant 的测试/benchmark 用了重复的旧副本实现导致测试报错、测速失真;统一改为调用 fp8_kernel.py 正式实现并用 bench_kineto 计时。

## 问题现象
量化 kernel 的测试报错、benchmark 输出的速度不正确(来自标题)。正文 Motivation 仅写「also cleanup code a bit」,并注明依赖 #7603;未描述具体报错信息、触发条件或影响面。测试对象是 per-token-group 8bit 量化(fp8/int8)的 Triton 与 sgl-kernel 两条实现路径。

## 根因
正文未披露根因分析(Modifications 一节为空),只有标题点明「test errors」与「benchmark wrong output speeds」两件事。

从 diff 可以描述改动所暗示的机制:`sgl-kernel/tests/test_per_token_group_quant_8bit.py` 与 `sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py` 内各自维护了一份与生产实现重复的本地代码(本地 triton kernel、本地 `triton_per_token_group_quant_8bit`/`sglang_per_token_group_quant_8bit` 副本),测试对副本断言而非对真实实现;旧 benchmark 用 `triton.testing.do_bench` 对 Python 包装函数整体计时,且未覆盖 column-major scales / tma aligned / ue8m0 等生产参数组合;旧测试对 fp8 输出直接做 `torch.allclose` 浮点比较也不适配 8bit 值。(由 diff 推断)

## 修复方案
- `python/sglang/srt/layers/quantization/fp8_kernel.py`:新增按 `dst_dtype` 分发的 `per_token_group_quant_8bit` 与 `sglang_per_token_group_quant_8bit`,int8 走 `per_token_group_quant_int8`,其余走 fp8 路径(透传 `column_major_scales`/`scale_tma_aligned`/`scale_ue8m0`)。
- `python/sglang/srt/layers/quantization/utils.py`:新增 `assert_fp8_all_close` 助手,把 fp8 张量按 uint8 位视图比较:符号翻转必须为 0、大差异(≥2)必须为 0、微小差异(≥1)占比 < 0.005。
- `sgl-kernel/tests/test_per_token_group_quant_8bit.py`:删除全部本地副本实现,测试改为调用 fp8_kernel.py 的正式函数,参数化加入 `scale_ue8m0` 组合并按 Blackwell/对齐条件 skip,输入 dtype 从 float16 改为 bfloat16。
- `sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py`:删除副本,改用 `bench_kineto` 按 kernel 名(`_per_token_group_quant_fp8` / `per_token_group_quant_8bit_kernel`)计时,hidden_dim 覆盖 DeepSeek V3/R1 的 1536/7168/18432。

```python
def assert_fp8_all_close(a: torch.Tensor, b: torch.Tensor):
    ...
    count_diff_sign = ((a_u8 >= 0) & (b_u8 < 0)).sum().item()
    count_tiny_diff = (diff_u8 >= 1).sum().item()
    count_large_diff = (diff_u8 >= 2).sum().item()
    assert (
        (count_diff_sign == 0)
        and (count_tiny_diff / numel < 0.005)
        and (count_large_diff == 0)
    ), f"{count_diff_sign=} {count_tiny_diff=} {count_large_diff=} {numel=}"
```

## 影响与回归风险
- 回归面:影响 sgl-kernel 的 per-token-group quant 测试与 benchmark 脚本本身,以及 `fp8_kernel.py` 新增的两个分发入口;测试断言口径变化(allclose → fp8 位级容差)会让测试判定标准更严格,benchmark 数值口径从 do_bench 整体计时变为 kineto 内核计时,前后测速数字不可直接对比。生产推理路径代码无改动。

## 关联
- https://github.com/sgl-project/sglang/pull/7603 (本 PR 声明依赖的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
