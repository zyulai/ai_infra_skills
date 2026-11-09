# PR #14808: Revert "fix: checking if tokenizer is in cache before downloading from HF"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14808](https://github.com/sgl-project/sglang/pull/14808) |
| 作者 | yhyang201 |
| 合并日期 | 2025-12-10 |
| labels | high priority, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论
整体回退 #14698 引入的 CI 下 tokenizer 缓存复用逻辑,恢复原先的 HF 下载路径,消除该改动造成的回归。

## 问题现象
正文未描述具体现象,仅说明本 PR 为「Reverts sgl-project/sglang#14698」,即整体回退此前合入的"先检查本地缓存中是否已有 tokenizer、有则跳过从 HF 下载"的改动。从被删除代码看,该逻辑带 `is_in_ci()` 门槛、只在 CI 环境生效,作用于 `get_tokenizer` 与 `get_processor` 两条模型加载路径(由 diff 推断)。

## 根因
正文未披露根因,仅声明回退 #14698,未说明缓存检查逻辑具体触发了什么故障。(由 diff 推断) 被回退的机制是:在调用 `AutoTokenizer.from_pretrained` 前,先经 `_check_tokenizer_cache`/`find_local_tokenizer_snapshot_dir` 扫描自定义 cache_dir 与默认 HF cache 的 snapshot 目录,若按 allow_patterns(`*.json`、`*.model` 等)匹配到 tokenizer 文件,再经 symlink 存在性、`.incomplete` 残留文件与 JSON/SentencePiece 内容校验通过后,直接用本地路径替换 `tokenizer_name` 跳过下载。该改动在下载主路径前插入了一段复杂分支,虽然自带多层防御,但回退 PR 将其全部删除,意味着项目选择放弃这层缓存复用、回到无条件走正常下载流程的行为。

## 修复方案
直接 revert #14698,改动集中在 `python/sglang/srt/utils/hf_transformers_utils.py`:
- 删除 `_validate_tokenizer_file`、`find_local_tokenizer_snapshot_dir`、`_check_tokenizer_cache` 三个函数;
- 移除 `get_tokenizer` 与 `get_processor` 中对 `_check_tokenizer_cache` 的调用,`tokenizer_name` 不再被本地缓存路径替换;
- 清理不再使用的导入(`glob`、`huggingface_hub`、`find_local_repo_dir`、`is_in_ci`)。

diff 摘录(get_tokenizer 中移除缓存检查):
```diff
-    # Check if tokenizer files are already in local cache (CI only)
-    tokenizer_name = _check_tokenizer_cache(
-        tokenizer_name, kwargs.get("cache_dir"), tokenizer_revision
-    )
-
     try:
         tokenizer = AutoTokenizer.from_pretrained(
             tokenizer_name,
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `python/sglang/srt/utils/hf_transformers_utils.py` 一个文件。所有经 `get_tokenizer`/`get_processor` 加载 tokenizer/processor 的路径受影响:CI 中此前命中缓存可跳过下载,回退后一律走 `AutoTokenizer.from_pretrained` 的正常下载流程;被删代码原有 `is_in_ci()` 门槛,非 CI 环境行为基本不变(由 diff 推断)。CI 场景下对 HF 下载量与启动耗时可能回升。

## 关联
- Reverts sgl-project/sglang#14698

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
