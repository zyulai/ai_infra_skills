# PR #43530: Fix CuPy runtime deps and restore humming

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43530](https://github.com/vllm-project/vllm/pull/43530) |
| 作者 | mmangkad |
| 合并日期 | 2026-05-26 |
| labels | ready, ci/build, nvidia |
| 热度 | 👍 5 · 💬 3 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
cupy-cuda13x==14.1.0 在 torch custom-op 注册时经 inspect 拉 import pytest 致 nightly 运行时 pytest 失败;钉 cupy<14.1.0 并恢复 humming。

## 问题现象
`vllm-openai:nightly` 运行时镜像里的 `pytest` 失败。时间线:#43492 曾临时 revert #42540,因为当时以为 #43480 是 humming import 问题;但 #43528 显示移除 humming 后同样的 `pytest` 失败仍然发生。真正的触发链:`vllm-openai:nightly` 默认安装 KV connectors,`lmcache` 拉取刚发布的 `cupy-cuda13x==14.1.0`,而该 CuPy release 在 PyTorch custom-op registration 通过 `inspect` 遍历已加载模块时会从 `cupy.testing._random` import `pytest`。作者验证了在 pytest 被屏蔽的条件下 `torch -> cupy 14.0.1 -> humming` 的导入顺序可以成功。

## 根因
正文披露:共同根因是 `cupy-cuda13x==14.1.0` 这个新发布版本会在 PyTorch custom-op registration 用 `inspect` 遍历已加载模块的过程中 import `pytest`(来自 `cupy.testing._random`),导致运行时镜像内 pytest 挂掉;此前误判成 humming import 问题的 #43480 也是同一根因。

## 修复方案
四处改动:
1. `requirements/kv_connectors.txt`:新增约束,把 KV connector 安装的 CuPy 钉到坏版本之下(解析为 14.0.1):

```diff
 lmcache >= 0.3.9
+# CuPy 14.1.0 imports pytest from cupy.testing._random. Use <14.1.0
+# until a fixed newer release is verified for runtime images.
+cupy-cuda13x < 14.1.0
 nixl >= 1.1.0 # Required for disaggregated prefill
```

2. `requirements/cuda.txt`:新增 `humming-kernels[cu13]==0.1.2`,恢复 humming(此前被 #43492 临时移除)。
3. `docker/Dockerfile` 与 `setup.py`:CUDA 12 构建时把 `humming-kernels[cu13]` 重写为 `[cu12]`(顺带把原 sed 模式改为同时去掉版本号前的 `[cu13]` 写法)。
4. `vllm/model_executor/layers/quantization/humming.py`:顶层导入改为 `if current_platform.is_cuda():` 条件导入 humming 模块,删除 `assert_humming_available()` 断言。

## 影响与回归风险
- 影响窗口:受影响环境为默认安装 KV connectors 的 `vllm-openai:nightly` 镜像在 `cupy-cuda13x==14.1.0` 发布后拉到的组合;正文未给出受影响的 vllm 提交范围。
- 回归面:KV connector 依赖解析(cupy 被上限钉住);CUDA 12/13 构建的 humming-kernels 安装与 `setup.py`/Dockerfile 的 requirements 重写逻辑;`humming.py` 的导入策略——非 CUDA 平台不再导入 humming 符号,`HummingConfig` 构造不再断言可用性,引用 humming 类型处依赖 `TYPE_CHECKING` 导入。

## 关联
- [#43492](https://github.com/vllm-project/vllm/pull/43492)(临时 revert)
- [#42540](https://github.com/vllm-project/vllm/pull/42540)(被 revert 的 humming PR)
- [#43480](https://github.com/vllm-project/vllm/pull/43480)(误判为 humming import 问题的失败)
- [#43528](https://github.com/vllm-project/vllm/pull/43528)(证明移除 humming 后失败仍在)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
