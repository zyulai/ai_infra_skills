# PR #54320: [Mypy] Fix mypy typing for Transformers models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54320](https://github.com/vllm-project/vllm/pull/54320) |
| 作者 | taneem-ibrahim |
| 合并日期 | 2026-09-17 |
| labels | ready, torch.compile, verified |
| 热度 | 👍 2 · 💬 30 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论

Transformers 建模后端目录此前被 mypy 排除、积压 144 个类型错误;修复类型与类结构后移除排除项,19 个源文件全绿。

## 问题现象

`vllm/model_executor/models/transformers` 目录被列在 `tools/pre_commit/mypy.py` 的排除名单里,mypy 完全不检查该后端。作者的复现步骤是:在 main 上手动删掉该排除项后运行 `pre-commit run mypy-3.12 --files $(git ls-files vllm/model_executor/models/transformers)`,输出 `Found 144 errors in 12 files (checked 19 source files)`;本 PR 分支上同样命令输出 `Success: no issues found in 19 source files`。影响面是静态类型检查的覆盖缺口,不是运行时故障;正文未描述任何线上症状。

## 根因

正文披露:该目录此前一直被 mypy 排除(#26533 的一部分),期间累积了 144 个类型错误。具体错误形态可从 diff 描述:`DynamicArgDims` 这类联合类型被内联重复书写、`type(self.model)` 直接传给 `%s` 格式化、`WeightsMapper` 先默认构造再原地 mutate 导致属性类型无法收窄、mixin 类未继承 `Base` 使 MRO/属性声明不可解析、若干 `Optional` 需收窄(用 `assert`/walrus)等(由 diff 推断)。

## 修复方案

1. `tools/pre_commit/mypy.py`:删除排除项 `r"vllm/model_executor/models/transformers"`。
2. `vllm/compilation/decorators.py`:新增 `DynamicArgDims: TypeAlias = dict[str, int | list[int] | dict[int, str]]` 并替换四处内联类型。
3. `transformers/` 目录内重构:各模型类不再直接继承 `Base`,改为 `CausalMixin(VllmModelForTextGeneration, Base)`、`LegacyMixin(Base)`、`MoEMixin(MixtureOfExperts, Base)` 等;`Base.__init__` 改调 `nn.Module.__init__(self)`;`hf_to_vllm_mapper` 改为一次性用完整参数构造、后续用 `|=` 合并,不再原地 mutate;`embedding_modules` 由 list 改为 dict `{"embed_tokens": "input_embeddings"}`;`fx_utils.py` 引入 `TypeIs` 并给 `is_leaf_call`/`is_fn`/`is_method`/`is_op` 收窄返回类型;`fusers/` 内以 `assert` 收窄 Optional、`merged_name` 由 ClassVar 改抽象属性;`LegacyMixin.forward` 增加 `**kwargs` 透传。

```diff
-    r"vllm/model_executor/models/[qQ]",
-    r"vllm/model_executor/models/transformers",
-    r"vllm/model_executor/models/[uU]",
+    r"vllm/model_executor/models/[qQ]",
+    r"vllm/model_executor/models/[uU]",
```

注:pack 中 diff 在 40000 字符处被截断,截断部分(应为其余 transformers 文件的类型修复)无法在此描述。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,除纯类型注解外含少量运行时语义调整:mixin 继承关系与 MRO 变化(mixin 现在继承 `Base`)、`Base.__init__` 初始化方式变化、`WeightsMapper` 的构造/合并时序变化(映射在 fuser 注册完成后才定型)、`LegacyMixin.forward` 开始透传 `**kwargs`、`TransformersRoutedExperts.get_expert_mapping` 签名扩展为可空参数、`MoERunner.forward` 加 `type: ignore[override]` 注释性改动。测试:全量 mypy-3.12 与 pre-commit 通过;fusers 测试 68 passed/1 skipped;`test_backend.py -k replace_` 7 passed/18 deselected。

## 关联

- https://github.com/vllm-project/vllm/issues/26533 (Fixes part of)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
