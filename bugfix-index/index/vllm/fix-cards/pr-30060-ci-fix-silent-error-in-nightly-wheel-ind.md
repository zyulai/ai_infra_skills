# PR #30060: [CI] fix silent error in nightly wheel index generation script, add generation time to HTML index

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30060](https://github.com/vllm-project/vllm/pull/30060) |
| 作者 | Harry-Chen |
| 合并日期 | 2025-12-05 |
| labels | ready, ci/build |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
#30008 的 wheel 索引脚本调用方式错误导致脚本从未运行、索引静默缺失;去掉 regex 依赖并直接用 `$PYTHON` 运行脚本修复,顺带在 HTML 索引加生成时间注释。

## 问题现象
nightly wheel 索引完全没有生成:由于调用命令写错,`generate-nightly-index.py` 根本没有被执行,但 wheel 本身仍照常上传,且 bash 不会报错(静默失败)。正文给出的日志显示:

```text
+ docker run --rm ... python:3-slim python3 pip install regex
python3: can't open file '/app/pip': [Errno 2] No such file or directory
+ echo 'Uploading indices to s3://vllm-wheels/<sha>/'
```

影响面是 nightly wheel 的 PyPI 风格索引发布(用户端索引内容缺失/过期)。

## 根因
PR 正文披露:#30008 引入的命令 `$PYTHON pip install regex && .buildkite/scripts/generate-nightly-index.py ...` 有三处错误:(1) 应该用 `$PYTHON -m pip` 而不是把 `pip` 当作 python 脚本参数传给解释器(于是报 can't open file '/app/pip');(2) `&&` 后半段脚本没有用同一个 python 解释器运行;(3) 前半段失败后 `&&` 短路,失败不会被 bash 捕获,形成静默错误。

## 修复方案
1. `.buildkite/scripts/upload-wheels.sh`:不再 `pip install regex`,改用 `sed -i 's/import regex as re/import re/g'` 把脚本里的 regex 导入临时替换回标准库 `re`(HACK 注释:regex 只是满足 pre-commit hook,这里不需要);随后直接 `$PYTHON .buildkite/scripts/generate-nightly-index.py ... --comment "commit $BUILDKITE_COMMIT"` 运行。
2. `.buildkite/scripts/generate-nightly-index.py`:新增 `--comment` CLI 参数,`INDEX_HTML_TEMPLATE` 头部插入 `<!-- {comment} -->`;为顶层与各 variant 的 index.html 生成 `Generated on <datetime.isoformat()> (commit ...)` 注释,标明生成时间与 commit。

```diff
-$PYTHON pip install regex && .buildkite/scripts/generate-nightly-index.py ...
+# HACK: we do not need regex module here, but it is required by pre-commit hook
+sed -i 's/import regex as re/import re/g' .buildkite/scripts/generate-nightly-index.py
+$PYTHON .buildkite/scripts/generate-nightly-index.py ... --comment "commit $BUILDKITE_COMMIT"
```

## 影响与回归风险
- 影响窗口:自 #30008 引入起,nightly wheel 索引一直未真正生成,直至本 PR。
- 回归面:改动仅涉及 Buildkite 发布脚本与索引生成器:所有生成的 index.html 会多出一条 HTML 注释(含时间与 commit);脚本在 CI 中被 sed 原地改写 import,若本地/其他环境直接运行仓库版本仍需 regex(pre-commit 需要)。对 wheel 上传本身无影响。

## 关联
- PR #30008(引入静默错误的前置改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
