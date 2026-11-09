# PR #24631: [Bugfix] Fix incorrect import of CacheConfig

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24631](https://github.com/vllm-project/vllm/pull/24631) |
| 作者 | DarkLight1337 |
| 合并日期 | 2025-09-11 |
| labels | ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
#21088 把 transformers 的已弃用 `CacheConfig` 误当成 vLLM 的导入,最新 transformers 下破坏 Whisper;改为从 `vllm.config` 导入。

## 问题现象
使用最新版本的 transformers 时,Whisper 模型会出错(报错来自错误的 `CacheConfig` 导入,CI 上可见失败日志,正文附有 Buildkite 失败链接)。影响面不止 Whisper 本身:任何把 Whisper 的 vLLM 实现当作 encoder 使用的模型同样会被破坏。正文未给出具体版本号与完整堆栈。

## 根因
PR 正文披露:前置 PR #21088 在 `vllm/attention/layers/cross_attention.py` 中错误地从 transformers 库导入已弃用(deprecated)的 `CacheConfig`,而不是从 vLLM 自己的配置模块导入;当用户环境安装最新版 transformers 时,该弃用符号不可用,导致导入失败、模型加载中断。

## 修复方案
改 `vllm/attention/layers/cross_attention.py` 的导入语句:删除 `from transformers import CacheConfig`,改为从 `vllm.config` 一并导入 `CacheConfig` 与 `VllmConfig`。

```diff
-from transformers import CacheConfig
 ...
-from vllm.config import VllmConfig
+from vllm.config import CacheConfig, VllmConfig
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围(由 #21088 引入)。
- 回归面:仅调整 cross_attention.py 的导入来源,不改运行时逻辑;受益路径是 Whisper 及以其 vLLM 实现为 encoder 的 encoder-decoder 模型在最新 transformers 环境下的加载。风险极低:若 vllm.config 中 CacheConfig 缺失则导入期即报错,容易在 CI 发现。

## 关联
- PR #21088(错误导入的引入者)
- CI 失败日志:[Buildkite build 30279](https://buildkite.com/vllm/ci/builds/30279/steps/canvas?sid=019936ef-4094-4ebf-85c3-047098b7a6ee)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
