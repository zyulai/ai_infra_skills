# PR #13093: Revert "fix: display served_model_name in /v1/models"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13093](https://github.com/sgl-project/sglang/pull/13093) |
| 作者 | CatherineSue |
| 合并日期 | 2025-11-11 |
| labels | router, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #13063:router 侧展示 served_model_name 的改动未通过 CI lint 检查,整体还原 worker_registration 原逻辑。

## 问题现象
正文未描述功能性 bug 现象;说明回滚原因是原 PR #13063(fix: display served_model_name in /v1/models)没有通过 CI lint 检查("The PR did not pass CI lint check")。

## 根因
正文披露的回滚原因:#13063 未通过 CI lint。功能性根因正文未披露。从 diff 看,#13063 曾在 sgl-router 的 worker 注册路径新增 `served_model_name` 处理(由 diff 推断其内容):`ServerInfo` 增加 `served_model_name: Option<String>` 字段;`get_dp_info` 的 model_id 解析链改为 model_id → served_model_name → model_path;`DiscoverMetadataStep` 的 HTTP 分支向 labels 写入 `served_model_name`;本回滚将这些全部移除并恢复原代码格式。

## 修复方案
`sgl-router/src/core/workflow/steps/worker_registration.rs`:

- 删除 `ServerInfo.served_model_name` 字段
- `get_dp_info` 的 model_id 恢复为 `model_id → model_path` 的回退链(恢复 `.filter(|s| !s.is_empty())` 空串过滤)
- `DiscoverMetadataStep` 的 HTTP 分支恢复只插入 `model_path` label,并还原 #13063 改动前的 match/缩进格式

## 影响与回归风险
- 影响窗口:正文未说明(#13063 合并至本回滚之间)。
- 回归面:sgl-router 的 worker 注册/元发现路径——router 不再读取或透出 `served_model_name`;/v1/models 的 served_model_name 展示若确属真实需求则回归(正文未评价);Rust 代码格式恢复原状以通过 lint。

## 关联
- [#13063](https://github.com/sgl-project/sglang/pull/13063)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
