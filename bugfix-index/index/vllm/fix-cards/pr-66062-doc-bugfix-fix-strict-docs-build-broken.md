# PR #66062: [Doc][Bugfix] Fix strict docs build broken by subclassing mocked torch classes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66062](https://github.com/vllm-project/vllm/pull/66062) |
| 作者 | subhashpolisetti |
| 合并日期 | 2026-11-06 |
| labels | bug, documentation, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
无 torch 的 docs 构建里 mock 的 torch.Tensor 被继承成 mock 实例致 @dataclass 崩溃;让 mock 可作为真实基类被继承。

## 问题现象
main 分支的 strict docs 构建(`mkdocs build --strict`)自 #65235 起失败,报 `Failed to import vllm.entrypoints.cli.preload.PreloadSubcommand` 与 `TypeError: expected string or bytes-like object, got 'PydanticMagicMock'`;此后每次 main 构建都以同样方式失败。改动 docs 的 PR 也失败,不改 docs 的 PR 会跳过构建。

## 根因
正文披露:`TraceTensor` 继承自 `torch.Tensor`,而 torch 在 docs 构建环境中是被 mock 的。对 mock 取子类得到的不是类而是 mock 实例,随后 `@dataclass` 处理 `reload/bake.py` 中 `Copy.src: TraceTensor` 注解时失败。失败链条最终表现为 argparse docs 生成器(`generate_argparse.py`)导入所有 CLI 子命令时在 `vllm preload` 上炸掉。

## 修复方案
修改 `docs/mkdocs/gen_files/generate_argparse.py`:
1. 新增 `MockBase` 类:支持 `__class_getitem__`(下标,供 `cachetools.LRUCache[_K, _V]`)与 `__get_pydantic_core_schema__`(pydantic schema,返回 `core_schema.any_schema()`);
2. `PydanticMagicMock` 增加 `__mro_entries__`:任何继承自 mock 的类会得到一个真实的 `MockBase` 基类,从而成为真实类型;每次返回新 type 以允许两个 mock 同时作基类;

```diff
+    # A fresh type each time so two mocks can be bases of one class
+    def __mro_entries__(self, bases):
+        return (type("MockBase", (MockBase,), {}),)
```

3. `torch.nn` 的 mock 由 `MagicMock` 换成 `PydanticMagicMock(name="torch.nn", ...)`,使 `nn.*` 的子类也被覆盖。

验证:在 docs-only 环境(`requirements/docs.txt`,Python 3.12,无 torch)跑 `mkdocs build --strict`:main 上 exit 1(同样 TypeError),本 PR exit 0、所有 CLI 页面(含 `cli/preload`)构建成功;与上次修复相比页面集合完全一致,`cli/bench/serve` 仅随机 `--result-filename` 默认值与一个 choices set 的顺序不同。

## 影响与回归风险
- 影响窗口:正文未给出具体版本号;自 #65235 引入 `TraceTensor` 起,至本修复合并前。
- 回归面:仅 docs 生成脚本 `generate_argparse.py` 的 mock 基础设施;运行时代码不受影响(正文称生成的 CLI 与 engine-args 页面不变)。mock 行为变化只作用于无 torch 的 docs 构建环境;若未来有类同时继承两个 mock 基类,`__mro_entries__` 每次生成新 type 的设计已覆盖(由 diff 推断)。

## 关联
- #65235(引入失败回归的 PR)
- #66044(修复无关交叉引用警告,经查非重复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
