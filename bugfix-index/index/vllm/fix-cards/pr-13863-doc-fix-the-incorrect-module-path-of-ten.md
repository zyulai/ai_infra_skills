# PR #13863: [Doc] fix the incorrect module path of tensorize_vllm_model

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#13863](https://github.com/vllm-project/vllm/pull/13863) |
| 作者 | tianyuzhou95 |
| 合并日期 | 2025-02-26 |
| labels | ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
tensorize_vllm_model 示例文档中模块路径写错,照抄命令无法运行;统一改为 examples.other。

## 问题现象
正文未描述具体现象,仅说明 tensorize_vllm_model 的模块路径不正确。由 diff 可见错误位于该示例脚本文件头部的 docstring:其中给出的运行命令均为 `python -m examples.offline_inference.tensorize_vllm_model ...`(由 diff 推断:该文件实际位于 `examples/other/` 目录下,照抄文档中的命令会因模块路径不存在而运行失败)。触发条件即用户按示例文档的说明复制运行序列化/反序列化命令。

## 根因
正文指出该错误由 commit aba8d6ee006b78149ac4514f460e4038b2d4f607 引入。具体机制正文未展开;由 diff 推断:该提交调整示例目录结构后,docstring 里的模块路径没有同步从 `examples.offline_inference.tensorize_vllm_model` 更新为 `examples.other.tensorize_vllm_model`,共涉及序列化、反序列化以及 `serialize --help` / `deserialize --help` 等多处说明。

## 修复方案
只改动 `examples/other/tensorize_vllm_model.py` 一个文件的 docstring,把其中 5 处错误的模块路径统一替换。关键 diff 摘录:

```diff
-python -m examples.offline_inference.tensorize_vllm_model \
+python -m examples.other.tensorize_vllm_model \
    --model facebook/opt-125m \
    serialize \
    --serialized-directory s3://my-bucket \
```

其余几处为反序列化示例及两处 `--help` 说明中的相同替换,均为纯文档文本修改,不涉及任何运行时逻辑。

## 影响与回归风险
- 影响窗口:自 commit aba8d6ee006b78149ac4514f460e4038b2d4f607 引入错误路径起,至本修复合并前(正文仅给出引入 commit,未给出版本号)。
- 回归面:仅修改示例脚本的 docstring 文本,不影响任何代码执行路径,回归风险极低;唯一影响是文档命令与实际模块路径恢复一致。

## 关联
正文提到的引入错误链接:
- https://github.com/vllm-project/vllm/commit/aba8d6ee006b78149ac4514f460e4038b2d4f607

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
