# PR #37215: [Bugfix] Fix render server crash for quantized models on CPU-only hosts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37215](https://github.com/vllm-project/vllm/pull/37215) |
| 作者 | sagearc |
| 合并日期 | 2026-03-16 |
| labels | bug, frontend, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
CPU-only 主机上运行 `vllm launch render` 加载量化模型时 mxfp4 校验报错,render 路径将 quantization 置 None 绕过校验。

## 问题现象
在缺少原生 bfloat16 支持的 CPU-only 主机上,运行 `vllm launch render` 并指定量化模型(如使用 mxfp4 的 `openai/gpt-oss-120b`)会崩溃:系统自动检测出 mxfp4 量化并尝试校验,但硬件不支持 bfloat16 时 dtype 自动回退为 float16,而 float16 不是 mxfp4 支持的数据类型,启动时触发 Pydantic 校验错误。

## 根因
正文披露:mxfp4 量化被自动检测后进入校验流程;无 bfloat16 支持的硬件上 dtype 回退为 float16,与 mxfp4 不兼容导致启动期校验错误。而 render server 只做预处理/后处理(处理 token ID 与多模态占位符),不执行任何模型推理,根本不需要 quantization 配置——校验本可完全跳过。

## 修复方案
修改 `vllm/entrypoints/cli/launch.py` 的 `run_launch_fastapi`:在构造 VllmConfig 之前设置 `model_config.quantization = None`,使 VllmConfig 跳过量化 dtype/能力校验,不影响渲染功能。

关键 diff 摘录:
```python
# Render servers preprocess data only — no inference, no quantized kernels.
# Clear quantization so VllmConfig skips quant dtype/capability validation.
model_config.quantization = None
```

## 影响与回归风险
- 回归面:仅影响 render server 启动路径(`run_launch_fastapi`);推理服务路径不受影响。render 路径从此不再携带量化配置,若未来 render 侧需要量化信息则需另行处理。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
