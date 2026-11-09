# PR #3373: Fix assertion failure in Qwen 1.5 with prefix caching enabled

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3373](https://github.com/vllm-project/vllm/pull/3373) |
| 作者 | chenxu2048 |
| 合并日期 | 2024-03-14 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
Qwen1.5/Qwen2 用 use_sliding_window 关闭滑窗但 get_sliding_window 仍非 None,前缀缓存断言失败;让该函数尊重该开关返回 None。

## 问题现象
启用 prefix caching 运行 Qwen 1.5(及 Qwen2)时断言失败。前缀缓存目前不支持 sliding window attention,model runner 会断言 `model_config.get_sliding_window()` 为 None;而 Qwen 系列默认已关闭滑窗,断言却仍然失败。

## 根因
正文披露:Qwen2/Qwen1.5 通过 `use_sliding_window` 字段来关闭 sliding window——默认关闭,但配置中仍带有 sliding_window 值,导致 `model_config.get_sliding_window()` 返回非 None,即使滑窗实际未启用,进而触发 model runner 的断言失败(修复 issue #3355)。

## 修复方案
修改 `vllm/config.py` 的 `get_sliding_window()`:当 hf_config 存在 `use_sliding_window` 字段且为 False 时直接返回 None,否则维持原有 `getattr(self.hf_config, "sliding_window", None)`。另新增 `tests/test_config.py` 验证 Qwen1.5(`use_sliding_window=False` 时返回 None、True 时返回窗口值)与 Mistral 的行为,并顺带修正两处 ROCm 报错信息中的引号转义与一处空行。

关键 diff 摘录:
```python
# Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
# addition to sliding window size. We check if that field is present
# and if it's False, return None.
if (hasattr(self.hf_config, "use_sliding_window")
        and not self.hf_config.use_sliding_window):
    return None
return getattr(self.hf_config, "sliding_window", None)
```

## 影响与回归风险
- 回归面:`get_sliding_window()` 是多处共享的配置读取入口;任何 hf_config 带 `use_sliding_window=False` 的模型,该函数现在正确返回 None——这正是 Qwen 系列的预期行为;其余模型(无该字段)返回值不变。

## 关联
- [#3355](https://github.com/vllm-project/vllm/issues/3355)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
