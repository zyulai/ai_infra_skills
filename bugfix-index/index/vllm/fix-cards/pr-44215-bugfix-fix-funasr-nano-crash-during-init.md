# PR #44215: [Bugfix] Fix FunASR-Nano crash during initialization

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44215](https://github.com/vllm-project/vllm/pull/44215) |
| 作者 | SunskyXH |
| 合并日期 | 2026-06-08 |
| labels | bug, ready |
| 热度 | 👍 3 · 💬 1 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
FunASR 未实现 get_language_model 且真实 LM 藏于嵌套 decoder,初始化即抛 NotImplementedError;实现该接口返回 self.model.decoder 修复。

## 问题现象
在 vLLM >= 0.22 / current main 上服务 `FunASRForConditionalGeneration`(如 `allendou/Fun-ASR-Nano-2512-vllm`)时引擎初始化崩溃:
`NotImplementedError: No language model found in FunASRForConditionalGeneration! You should initialize it via '_mark_language_model', and make sure 'embed_input_ids' is implemented.`
修复后 `vllm serve allendou/Fun-ASR-Nano-2512-vllm` 应用启动正常。

## 根因
正文披露:PR #39805 在 `gpu_model_runner.load_model()` 中加入 `get_language_model()` 调用,用于把 VLM wrapper 解包成底层 MoE 语言模型以支持 EPLB;该调用对每个非 MoE 的 `SupportsMultiModal` 模型都会执行,且不受 `enable_eplb` 保护。FunASR 从未实现 `get_language_model()`,于是落到 `interfaces.py` 的默认实现——它只扫描直接子模块,而 FunASR 真正的语言模型位于嵌套的 `self.model.decoder`(`Qwen3Model`),不是直接子节点,因此在 vLLM 0.22.0 上报错。

## 修复方案
修改 `vllm/model_executor/models/funasr.py`:为 `FunASRForConditionalGeneration` 实现 `get_language_model()`,返回真实语言模型所在位置 `self.model.decoder`。

关键 diff 摘录:
```python
def get_language_model(self) -> torch.nn.Module:
    # Required as part of SupportsMultiModal interface.
    return self.model.decoder
```

## 影响与回归风险
- 影响窗口:vLLM >= 0.22 起(引入 #39805 的 get_language_model 调用后)受影响。
- 回归面:仅新增 FunASR 模型的接口方法;改变的是 EPLB 解包路径对 FunASR 的行为(现在能正确找到 decoder),其他模型与路径不受影响。

## 关联
- [PR #39805](https://github.com/vllm-project/vllm/pull/39805)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
