# PR #32937: [CI] Fix MoE compile and DSA indexer regressions

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32937](https://github.com/sgl-project/sglang/pull/32937) |
| 作者 | mmangkad |
| 合并日期 | 2026-07-30 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 4 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论

#31888 引入两处回归:MoE 编译测试解包失败、DSA indexer 测试缺 mock 属性;按名取字段并补默认值修复。

## 问题现象

两个 CI 测试在 #31888 最后一次 base CI 运行(该 PR 被 force-merge 前)就已经真实失败:

- `test_torch_compile_moe.py` 失败,报错 `ValueError: too many values to unpack (expected 3)`。CI 链接:https://github.com/sgl-project/sglang/actions/runs/30411124003/job/90771215458
- `TestDSAIndexer.test_forward_decode_mode` 失败,报错 `AttributeError: 'ServerArgs' object has no attribute 'enable_two_batch_overlap'`。CI 链接:https://github.com/sgl-project/sglang/actions/runs/30411124003/job/90771215506

影响面为 CI 测试流程,两处都是 #31888 合入后暴露的回归,不是线上推理错误。

## 根因

按 PR 正文:

- `StandardDispatchOutput` 新增了第四个字段,而 `fused_moe_forward_native` 仍按三元组解包 `x, x_scale, topk_output = dispatch_output`,字段数不匹配导致 `ValueError: too many values to unpack (expected 3)`。
- DSA indexer 后端新增了 `enable_two_batch_overlap` 设置项,但测试里构造的 `ServerArgs` mock 没有该属性,访问即抛 `AttributeError`。

## 修复方案

改动两处:

- `python/sglang/srt/layers/moe/fused_moe_native.py`:不再按位置解包,改为按字段名读取所需成员:

  ```diff
  -    x, x_scale, topk_output = dispatch_output
  +    x = dispatch_output.hidden_states
  +    topk_output = dispatch_output.topk_output
  ```

- `test/registered/kernels/ops/attention/test_dsa_indexer.py`:给 `ServerArgs` mock 补上该设置的默认值 `False`:

  ```diff
                   "disaggregation_mode": "null",
  +                "enable_two_batch_overlap": False,
               },
  ```

## 影响与回归风险

- 影响窗口:正文未给出版本范围,仅说明回归来自 #31888 合入之后。
- 回归面:`fused_moe_forward_native` 的 native MoE 前向路径(torch.compile 场景)与 DSA indexer 内核测试。按名取字段后,新增字段(如第四个成员)不再参与解包,后续给 `StandardDispatchOutput` 加字段不会再破坏此调用点;mock 补默认值仅影响测试构造参数。

## 关联

- #31888(两处回归的来源 PR)
- CI 运行记录:[Run 30411124003 / job 90771215458](https://github.com/sgl-project/sglang/actions/runs/30411124003/job/90771215458)、[job 90771215506](https://github.com/sgl-project/sglang/actions/runs/30411124003/job/90771215506)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
