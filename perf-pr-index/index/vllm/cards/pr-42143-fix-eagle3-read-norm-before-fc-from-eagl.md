# PR #42143: fix(eagle3): read norm_before_fc from eagle_config for NVIDIA checkpoint

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ghost |
| 合并 | 2026-05-23 |
| 分类 | Speculative Decoding |
| Label | speculative-decoding, ready, llama, nvidia, verified |
| Diff | diffstat 缺失(pack 标注 +None/-None);diff 为 1 个文件的 4 行修改 |
| 链接 | https://github.com/vllm-project/vllm/pull/42143 |

## 一句话结论

修复 NVIDIA Eagle3 checkpoint 的 `norm_before_fc` 读取路径,恢复 FC 前 RMSNorm,挽回被拉低的 draft 接受率。

## 动机

NVIDIA 发布的 Eagle3 checkpoint(GPT-OSS 系列)把 `norm_before_fc` 存在嵌套的 `eagle_config` dict 里,而非顶层 config 字段。vLLM 此前只读顶层属性,导致这些 checkpoint 的 FC 前 RMSNorm 被静默跳过,drafter 输入分布与训练时不一致,Eagle3 投机解码的 token 接受率下降,直接损失投机解码加速比。

## 优化点

对齐同代码块中 `use_aux_hidden_state` 已有的读取模式:先从 `eagle_config` dict 取 `norm_before_fc`,取不到再回退顶层属性 `getattr(self.config, "norm_before_fc", False)`,并用 `bool()` 归一化;同时把 `eagle_config = getattr(...) or {}` 统一为空 dict 兜底,消除 `use_aux_hidden_state` 分支对 None 的特判。改动仅 4 行,无行为分支新增——只纠正了配置来源优先级。

## 关键改动

```text
vllm/model_executor/models/llama_eagle3.py: __init__ 中 norm_before_fc 的读取顺序改为 eagle_config 优先、顶层回退
```

```python
self.norm_before_fc = bool(
    eagle_config.get(
        "norm_before_fc", getattr(self.config, "norm_before_fc", False)
    )
)
```

## 性能声明

PR 未直接给出修复前后接受率对比数字。附带的 eagle3 收益基线(1x H200,Qwen3-32B + Qwen3-32B_eagle3,bf16,单请求 decode 512 tokens,`num_speculative_tokens=3`):baseline 51.3 tok/s → eagle3 67.1 tok/s(1.31x)。该 PR 的作用是让 checkpoint 配置错误的模型恢复到应有接受率,PR 未注明 NVIDIA GPT-OSS checkpoint 修复前后的具体吞吐对比。

## 验证与回归风险

测试计划含三项:greedy 输出与不开 eagle3 基线完全一致(4/4 prompts 通过,验证功能正确性而非 bit-level);单请求 decode 吞吐脚本;config 解析分支单测(嵌套 dict 优先、顶层回退两分支均断言),另验证 `get_eagle3_aux_layers_from_config` 不受影响。结果:4/4 greedy 匹配、config 单测全过、使用顶层 `norm_before_fc` 的模型无回归(fallback 路径未变)。风险面极小:仅 llama_eagle3 的配置读取,影响所有 Eagle3 drafter 的 norm 位置判定;若某 checkpoint 在两处都写了不同值,现在以嵌套 `eagle_config` 为准。

## 关联

同区域(Eagle3/投机解码配置修复):索引内 PR #25916(EAGLE-3 preamble 修正,同属"提升实际接受率"类修复)、PR #22872(multimodal EAGLE)、PR #46995(DSpark);Eagle3 drafter 实现文件同属 llama_eagle3 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
