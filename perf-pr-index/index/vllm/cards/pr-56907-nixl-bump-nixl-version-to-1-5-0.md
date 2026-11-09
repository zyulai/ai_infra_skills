# PR #56907: [NIXL] Bump NIXL version to 1.5.0

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ovidiusm |
| 合并 | 2026-10-06 |
| 分类 | Overlap / Scheduler \| tier: major |
| Label | ready, ci/build, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/56907 |

## 一句话结论

将 KV connector 依赖 NIXL 由 1.4.1 升级到 1.5.0。

## 动机

PR 正文的 Purpose 仅写明 "Bump NIXL version to 1.5.0"，未进一步说明升级动因、修复的问题或目标收益，素材中也没有 issue 链接或背景描述，故具体动机 PR 未注明。从 diff 可见该依赖位于 `requirements/kv_connectors.txt`，与 lmcache、cupy-cuda13x、mooncake-transfer-engine 等一同管理 KV connector 依赖版本；文件内注释说明 CuPy 14.1.0 因破坏 runtime images 被排除，以及 CUDA 13 镜像会把 mooncake-transfer-engine 换成同版本的 cuda13 变体。

## 优化点

改动为单一依赖版本钉住（pin）的更新：把 `nixl == 1.4.1` 改为 `nixl == 1.5.0`。正文未描述任何算法、内核或调度层面的机制改动，也未说明新版本带来的具体行为变化，故优化点 PR 未注明；可确认的只有 KV connector 依赖清单中的版本号变更，diff 未触及任何 Python / CUDA 源码文件。同一文件的 diff 上下文还钉住 `lmcache >= 0.3.9`、`cupy-cuda13x != 14.1.0` 与 `mooncake-transfer-engine >= 0.3.12`，并说明 CUDA 12 构建下 `install-kv-connectors.sh` 会把 mooncake 换成同版本的 cuda13 变体，本 PR 均未改动这些条目；checklist 中 "The necessary documentation update" 一项也未勾选，正文未提到需要同步的文档。

## 关键改动

```text
requirements/kv_connectors.txt: 将 nixl 固定版本由 1.4.1 提升为 1.5.0（本 PR 唯一改动文件）
```

```diff
-nixl == 1.4.1
+nixl == 1.5.0
```

## 性能声明

PR 正文未给出定量性能数据，pack 的 "perf claims detected in body (regex, unverified)" 亦标注 "(none detected by regex)"。正文没有 benchmark 数字，也没有模型 / 硬件 / 精度 / workload 信息，测试条件 PR 未注明。

## 验证与回归风险

Test Plan 仅勾选 "vLLM CI" 与 "Internal testing in progress"（内部测试进行中）；Test Result 明确写 "Pending"，checklist 中 "pasting the results comparison before and after, or e2e results" 与文档更新项均未勾选。PR 未注明可能的回归风险与受影响路径；改动文件为 `requirements/kv_connectors.txt`，正文未对 KV connector 之外的路径做任何说明。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
