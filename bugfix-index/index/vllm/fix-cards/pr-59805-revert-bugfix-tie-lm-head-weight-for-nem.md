# PR #59805: Revert "[Bugfix] Tie lm_head.weight for Nemotron Parse when checkpoint omits it" (#53020)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59805](https://github.com/vllm-project/vllm/pull/59805) |
| 作者 | Thangnguyenvn98 |
| 合并日期 | 2026-10-02 |
| labels | bug, multi-modality |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#53020 新增的 Nemotron-Parse-2.0 测试因 tokenizer 加载抛 DuplicatePatternError 使 CI 常红,按原样 revert。

## 问题现象
#53020 合并后,其新增的 `nvidia/NVIDIA-Nemotron-Parse-2.0` 测试用例在每次 CI 运行都失败,该 step 自合并起在 main 上持续红。失败发生在 2.0 checkpoint 的 tokenizer 加载阶段、任何模型代码运行之前:`TokenizerFast.from_file` 抛 `Exception: DuplicatePatternError: [225, 131, 156] at line 449472 column 1`(transformers/tokenization_utils_tokenizers.py:120 convert_to_native_format)。

## 根因
正文披露:新增测试用例无法在 CI 镜像所带的 tokenizer 库下运行(tokenizer 文件一加载即报错),因此该用例根本跑不到 tie_weights 逻辑;#53020 的 tie_weights 修复本身未被判定为错误,但要求重新落地时须附带能加载该 tokenizer 的方案,或去掉 2.0 测试用例。

## 修复方案
精确 revert #53020(声明为 exact inverse,`git diff 83cadd65d9 83cadd65d9^` 与本 diff 一致):
- 删除 `vllm/model_executor/models/nemotron_parse.py` `__init__` 中按 `tie_word_embeddings` 配置执行 `self.lm_head.tie_weights(self.decoder.embed_tokens)` 的分支;
- `tests/models/multimodal/generation/test_nemotron_parse.py` 参数化恢复为仅 `nvidia/NVIDIA-Nemotron-Parse-v1.2`。

验证:revert 后 step 回到 #53020 之前通过的状态(build 92354 pass,92561/92599 仅 `test_models[5-bfloat16-nvidia/NVIDIA-Nemotron-Parse-2.0]` 失败);ruff check 与 ruff format --check 通过。

## 影响与回归风险
- 影响窗口:正文说明自 #53020 合并起 CI 失败。
- 回归面:Nemotron Parse 的 lm_head 权重加载不再 tie;依赖 checkpoint 省略 lm_head.weight 的 2.0 类 compact export 在 vLLM 中回到不支持状态,直到修复配合 tokenizer 加载方案重新落地。

## 关联
[#53020](https://github.com/vllm-project/vllm/pull/53020)、[build 92354](https://buildkite.com/vllm/ci/builds/92354)、[build 92561](https://buildkite.com/vllm/ci/builds/92561)、[build 92599](https://buildkite.com/vllm/ci/builds/92599)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
