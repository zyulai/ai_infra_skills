# PR #20689: [diffusion] fix: fix sampling params incorrectly override in cli

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20689](https://github.com/sgl-project/sglang/pull/20689) |
| 作者 | mickqian |
| 合并日期 | 2026-03-17 |
| labels | run-ci, diffusion |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
diffusion CLI 的 sampling 参数默认值会错误覆盖模型子类默认值,改为 `argparse.SUPPRESS` 只透传用户显式传入的参数。

## 问题现象
正文未描述具体现象(Motivation/Modifications 等章节均为模板占位,无实际内容)。从修复后的回归测试可反推影响面:用户未在命令行显式传参时,基类 `SamplingParams` 的默认值会覆盖子类(模型级)默认值,例如 QwenImage 路径下 `negative_prompt` 应为 `" "`、`guidance_scale` 应为 4.0,却会被基类默认值取代(由 diff 推断)。

## 根因
正文未披露根因。由 diff 推断:`SamplingParams.add_cli_args` 为每个 CLI 参数都写了 `default=SamplingParams.xxx`(基类字段默认值),argparse 会把这些默认值无条件填进 namespace;随后 `get_cli_args` 无法区分「用户显式传入」与「argparse 填入的默认值」,导致未传参的字段也以基类默认值参与合并,压掉模型子类自定义默认值。

## 修复方案
改动位于 `python/sglang/multimodal_gen/configs/sample/sampling_params.py` 的 `add_cli_args`:
- 定义局部包装函数 `add_argument`,强制 `kwargs.setdefault("default", argparse.SUPPRESS)`;
- 所有 sampling 相关 CLI 参数(约 40 个:--data-type、--guidance-scale、--negative-prompt、--seed、分辨率开关、teacache、RIFE 插帧、超分等)改走该包装函数,移除逐个写死的 `default=SamplingParams.xxx`。

效果:用户未传的参数不会出现在 namespace 中,`get_cli_args` 只返回显式传入的字段,子类默认值得以保留;显式传入的值(即使等于基类默认值)仍可覆盖。

测试文件 `test_sampling_params.py` 重写为 `TestSamplingParamsCliArgs`,新增用例:空 CLI 解析结果为空 dict、显式参数被保留、QwenImage 默认值保留与显式覆盖基类默认值、`_merge_with_user_params` 对 explicit_fields 的处理等,替换原先针对 negative_prompt 的旧回归测试。

关键 diff 摘录:

```diff
+        def add_argument(*name_or_flags, **kwargs):
+            kwargs.setdefault("default", argparse.SUPPRESS)
+            return parser.add_argument(*name_or_flags, **kwargs)
+
+        add_argument("--data-type", type=str, nargs="+")
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:仅影响 multimodal_gen diffusion 路径中经 `SamplingParams.add_cli_args`/`get_cli_args` 构造参数的 CLI 入口。未传参字段现在会缺失于 kwargs 中,依赖「kwargs 必含全部字段」的外部代码行为可能改变(由 diff 推断);`--negative-prompt`、`--moba-config-path`、`--diffusers-kwargs` 原先显式 `default=None` 的参数现同样改为 SUPPRESS,行为一致性依赖下游对缺失键的默认处理。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
