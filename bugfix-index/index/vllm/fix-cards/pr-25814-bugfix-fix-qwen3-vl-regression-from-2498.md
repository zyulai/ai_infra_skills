# PR #25814: [Bugfix] Fix Qwen3-VL regression from #24982

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25814](https://github.com/vllm-project/vllm/pull/25814) |
| 作者 | ywang96 |
| 合并日期 | 2025-09-28 |
| labels | ready, qwen |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
#24982 让 Qwen3-MoE 直接读 `hf_config`,破坏了 Qwen3-VL;四处改用 `hf_text_config` 修复,两模型恢复可加载。

## 问题现象
正文未描述具体报错信息,仅说明 #24982 造成回归:Qwen3-VL 需要通过 `get_text_config()` 访问 Qwen3-MoE 的 text config,而 #24982 之后相关代码直接读了 `hf_config`,导致 Qwen3-VL 出问题(修复前 Qwen3-VL 无法正常工作;作者本地验证修复后 Qwen3-VL 与 Qwen3-MoE 均可正常加载)。

## 根因
正文披露:根因是 #24982 的改动——Qwen3-VL 是多模态套壳模型,其 Qwen3-MoE 部分必须经由 `get_text_config()`/text config 获取文本侧配置;#24982 在 `qwen3_moe.py` 中把配置来源改为 `vllm_config.model_config.hf_config`(顶层 hf_config),Qwen3-VL 场景下这不再是 text config,因而引发回归。

## 修复方案
改 `vllm/model_executor/models/qwen3_moe.py` 四处配置获取,统一改用 `vllm_config.model_config.hf_text_config`:

```diff
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_text_config
```

涉及 `Qwen3MoeDecoderLayer.__init__`(原 `hf_config`)、`Qwen3MoeModel.__init__`(原 `hf_config.get_text_config()`)、`Qwen3MoeForCausalLM.__init__`(原 `hf_config`)等位置,`hf_text_config` 本身即等价于 text config。作者本地测试 Qwen3-VL 与 Qwen3-MoE 均正常加载。

## 影响与回归风险
- 影响窗口:#24982 合入至本 PR 之间,Qwen3-VL(及其它以 hf_config 顶层为多模态配置的模型)受影响;正文未给出具体版本号。
- 回归面:`qwen3_moe.py` 中 Qwen3Moe 系列(Qwen3-MoE 本体及被 Qwen3-VL 复用的路径)的配置读取;对纯文本 Qwen3-MoE 而言 `hf_text_config` 与 `hf_config` 等价,行为应不变。

## 关联
- #24982(引入回归的 PR)
- 正文未列出其他 issue/PR

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
