# PR #12434: [Bugfix][Kernel] Fix perf regression caused by PR #12405

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#12434](https://github.com/vllm-project/vllm/pull/12434) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-01-26 |
| labels | ready, ci/build |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
#12405 引入 FA3 路径性能回退,实际回退到 FA2 执行;通过更新 CMakeLists.txt 中 vllm-flash-attn 的固定 commit 修复。

## 问题现象
PR #12405 合入后出现性能回退,表现为(FA3 场景)实际回退到 FA2 执行。正文以此测试命令验证:`VLLM_FLASH_ATTN_VERSION=3 python -m vllm.scripts serve meta-llama/Meta-Llama-3-8B`。正文未给出具体的性能数字或更细的症状描述。

## 根因
正文未披露详细根因,只说明回退由 PR #12405 引起,后果是「resulted in falling back on FA2」(落到 FA2 执行,损失 FA3 性能)。
(由 diff 推断)修复方式是更新依赖版本,暗示 #12405 对应的旧 vllm-flash-attn commit(9732b0ce...)在目标场景下不可用/被判定不支持,导致运行时选择退回 FA2;新的 flash-attention commit(d4e09037...)恢复了 FA3 的可用性。

## 修复方案
改 `CMakeLists.txt`:`FetchContent_Declare(vllm-flash-attn ...)` 的 `GIT_TAG` 从 `9732b0ce005d1e6216864788502d5570004678f5` 更新为 `d4e09037abf588af1ec47d0e966b237ee376876c`(即拉取 vllm-project/flash-attention 的新提交重新构建)。

```diff
   FetchContent_Declare(
           vllm-flash-attn
           GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
-          GIT_TAG 9732b0ce005d1e6216864788502d5570004678f5
+          GIT_TAG d4e09037abf588af1ec47d0e966b237ee376876c
```

## 影响与回归风险
- 影响窗口:PR #12405 合入之后、本 PR 之前的构建(正文未给出具体版本号)。
- 回归面:构建期改动——所有从源码编译 vLLM 并启用 vllm-flash-attn(FA3)的构建会拉取新的 flash-attention commit,需重新编译该依赖(`BINARY_DIR` 不跨 build type 共享);运行期无代码逻辑改动。若新 commit 引入上游变化,FA3 kernel 行为随之变化,属上游依赖升级的一般风险。

## 关联
- [PR #12405](https://github.com/vllm-project/vllm/pull/12405)(引入回归)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
