# PR #36715: [Doc] Fix grammatically incorrect error message in gpu_worker and xpu_worker

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36715](https://github.com/vllm-project/vllm/pull/36715) |
| 作者 | robinguo23 |
| 合并日期 | 2026-07-07 |
| labels | intel-gpu, ready, v1 |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
gpu_worker/xpu_worker 的 RuntimeError 文案 "Not support device type" 语法错误,统一改为 "Unsupported device type"。

## 问题现象
在 `vllm/v1/worker/gpu_worker.py` 与 `vllm/v1/worker/xpu_worker.py` 的 `init_device` 中,当请求的设备类型不受支持时抛出的 RuntimeError 文案为 `Not support device type: ...`,"Not support" 属于不合语法的英文(正确形容词形式应为 "Unsupported"),影响报错信息的可读性与用户排查体验。正文未描述更多触发场景;触发条件即 device 类型未被 worker 支持、走 `init_device` 的 else 分支。

## 根因
正文披露:报错字符串使用了语法错误的英文表达("Not support" is grammatically incorrect English),是当初撰写报错文案时的措辞错误,非功能性缺陷。

## 修复方案
仅改两处 RuntimeError 字符串,不改任何逻辑:
- `vllm/v1/worker/gpu_worker.py`:`raise RuntimeError(f"Not support device type: {self.device_config.device}")` → `raise RuntimeError(f"Unsupported device type: {self.device_config.device}")`
- `vllm/v1/worker/xpu_worker.py`:同样替换。
```python
-            raise RuntimeError(f"Not support device type: {self.device_config.device}")
+            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:正文明确"Error message string change only, no functional impact";唯一潜在影响是依赖旧错误文案做字符串匹配的脚本/测试会失效(正文未提及此类依赖)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
