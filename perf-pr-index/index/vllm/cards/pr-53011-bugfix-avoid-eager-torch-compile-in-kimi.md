# PR #53011: [Bugfix] Avoid eager torch.compile in Kimi K2.5 vision helper

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @saichowdary007 |
| 合并 | 2026-10-29 |
| 分类 | CUDA Graph / Compiler \| tier: major |
| Label | bug, ready, torch.compile, kv-connector, kimi |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明（diff 完整呈现） |
| 链接 | https://github.com/vllm-project/vllm/pull/53011 |

## 一句话结论

移除 Kimi K2.5 vision helper 的 eager torch.compile 包装，避免 Triton 过早初始化致 warm-cache reload 失败。

## 动机

PR 正文 Problem 说明，Kimi K2.5 vision encoder 的 `get_rope_shape` 被裸 `torch.compile` 装饰，且 eager wrapper 会在第一个 input key 上调用它；这可能使 Triton 在 vLLM 的 compilation lifecycle 配置好 cache 目录之前被初始化，导致 warm-cache reload failures。正文标注 Fixes #52965。

## 优化点

正文 Solution：移除 eager wrapper 与裸 decorator，使编译保持在 vLLM compilation lifecycle 管理之下，`get_rope_shape` 退回到调用方既有编译路径中的普通 helper。结合 diff，被删除的是 `get_rope_shape_decorate` 包装函数（内部用 `_get_rope_shape_first_call_flag` 在首个 key 时以 `shape=(64, 64)` 预热调用一次）以及 `get_rope_shape` 上的 `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend, disable=current_platform.simple_compile_backend == "tpu")` 装饰器，并移除随之不再使用的 `current_platform` import。

## 关键改动

```text
vllm/model_executor/models/kimi_k25_vit.py: 删除 get_rope_shape_decorate eager 包装与 get_rope_shape 上的 @torch.compile 装饰，移除 current_platform import，保留普通 helper 由调用方编译路径处理
```

（本 PR diff 仅涉及以上 1 个文件。）

```text
-def get_rope_shape_decorate(func):
-    _get_rope_shape_first_call_flag = set()
-
-    def wrapper(org, interpolation_mode, shape):
-        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
-        if key not in _get_rope_shape_first_call_flag:
-            _get_rope_shape_first_call_flag.add(key)
-            _ = func(org, interpolation_mode, shape=(64, 64))
-        return func(org, interpolation_mode, shape)
```

## 性能声明

PR 正文未给出定量性能数据。正文为 bugfix 说明，无 benchmark 数字，pack 的 perf claims regex 未检测到；PR 未注明测试条件。

## 验证与回归风险

PR 引入/依赖的测试（正文 Testing）：针对 `vllm/model_executor/models/kimi_k25_vit.py` 执行 `.venv/bin/ruff check`、`ruff format --check`、`.venv/bin/python -m py_compile`、`pre-commit run --files`、`git diff --check`；正文说明完整模型测试套件需要仓库的 CUDA/PyTorch 环境，未在 Apple Silicon 主机上运行；AI 辅助做仓库探索与实现，提交者审阅改动并负责验证。PR 未注明回归风险；改动只影响 Kimi K2.5 vision encoder 的 `get_rope_shape` 编译时机，可能受影响路径为 vision encoder 的 torch.compile / CUDA Graph 捕获与 warm-cache reload；其他模型、硬件与精度正文未说明。

## 关联

正文标注 Fixes #52965（issue #52965）；未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
