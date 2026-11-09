# PR #53011: [Bugfix] Avoid eager torch.compile in Kimi K2.5 vision helper

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53011](https://github.com/vllm-project/vllm/pull/53011) |
| 作者 | saichowdary007 |
| 合并日期 | 2026-10-29 |
| labels | bug, ready, torch.compile, kv-connector, kimi |
| 热度 | 👍 3 · 💬 30 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论
Kimi K2.5 vision 的 `get_rope_shape` 被裸 `torch.compile` 装饰并经 eager 包装提前调用,抢在 vLLM 编译生命周期前初始化 Triton,造成 warm-cache 重载失败;删除装饰器与 eager 包装修复。

## 问题现象
Kimi K2.5 视觉编码器中,带 eager 包装的 `get_rope_shape` 在首次输入时就会被调用。此时会提前初始化 Triton,而 vLLM 的编译流程尚未配置好缓存目录,进而导致 warm-cache(暖缓存)重载失败。正文未描述具体报错日志与影响面大小,仅说明问题出在 `get_rope_shape` 的裸 `torch.compile` 装饰与 eager wrapper 的组合上。

## 根因
PR 正文披露:Kimi K2.5 vision encoder 的 `get_rope_shape` 被 bare `torch.compile` 装饰,而一个 eager wrapper 会在第一个输入 key 上调用它。这次提前调用可能在 vLLM 的 compilation lifecycle 配置好缓存目录之前就初始化 Triton,从而导致 warm-cache 重载失败。

## 修复方案
修改 `vllm/model_executor/models/kimi_k25_vit.py`:

1. 删除整个 eager 包装函数 `get_rope_shape_decorate`(内部用 set 记录首次调用并额外跑一次 `shape=(64, 64)`);
2. 删除 `get_rope_shape` 上的 `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend, ...)` 装饰器;
3. 顺带移除不再使用的 `from vllm.platforms import current_platform` 导入。

这样 `get_rope_shape` 保留为普通 helper,由调用方既有的编译路径接管,编译始终处于 vLLM 编译生命周期之内。关键 diff 摘录(被移除的装饰器):

```diff
-@get_rope_shape_decorate
-@torch.compile(
-    dynamic=True,
-    backend=current_platform.simple_compile_backend,
-    disable=current_platform.simple_compile_backend == "tpu",
-)
 def get_rope_shape(org, interpolation_mode, shape):
     return (
         F.interpolate(
```

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:改动仅触及 `kimi_k25_vit.py` 一个文件。`get_rope_shape` 不再自带 `torch.compile` 装饰,eager wrapper 的「首次额外跑 shape=(64,64)」预热行为也随之消失;依赖该函数自带编译的路径改由调用方编译路径覆盖。若调用方编译路径未覆盖此 helper,理论上可能影响该函数的编译优化,但功能语义不变。diff 还移除了 `current_platform` 导入,不影响其他模块。

## 关联
- Fixes #52965: https://github.com/vllm-project/vllm/issues/52965

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
