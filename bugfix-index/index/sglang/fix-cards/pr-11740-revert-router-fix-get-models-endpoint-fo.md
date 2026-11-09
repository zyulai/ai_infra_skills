# PR #11740: Revert "[router] fix get_models endpoint for openai router (#11687)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11740](https://github.com/sgl-project/sglang/pull/11740) |
| 作者 | key4ng |
| 合并日期 | 2025-10-17 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论

整体回滚 #11687:移除 RouterManager 在单 router 模式下把 `/v1/models` 委托给 OpenAI router 的逻辑及其测试,恢复直接读 worker_registry。

## 问题现象

正文未描述具体现象。正文仅一句 "This reverts commit 0975ba99bcf38c34c28e738c14e9df0abb9cb10a.",Motivation/Modifications 等模板段均未填写。

## 根因

正文未披露根因。(由 diff 推断)被回滚的 #11687 曾在 `RouterManager::get_models` 中加入委托逻辑:非 `enable_igw`(单 router 模式)时先查找 default_router,若存在则把请求转给该 router(尤其 OpenAI 模式,由其代理 `/v1/models`),否则才回退到 `worker_registry.get_models()` 聚合;本 revert 将该方法恢复为直接使用 `worker_registry`,并删除了对应的委托行为测试 `test_router_manager_delegates_models_to_openai_router`。

## 修复方案

按 revert 提交恢复两个文件:

`sgl-router/src/routers/router_manager.rs` 的 `get_models`:删除单 router 模式下读取 `default_router` 并调用 `router.get_models(req).await` 的委托分支,签名改回 `async fn get_models(&self, _req: Request<Body>)`,方法体恢复为直接 `let models = self.worker_registry.get_models();` 及其后续的空列表处理:

```diff
-    async fn get_models(&self, req: Request<Body>) -> Response {
-        // In single-router mode, delegate to the router (especially for OpenAI mode)
-        if !self.enable_igw {
-            ...delegation...
-        }
+    async fn get_models(&self, _req: Request<Body>) -> Response {
         let models = self.worker_registry.get_models();
```

`sgl-router/tests/test_openai_routing.rs`:删除整个 `test_router_manager_delegates_models_to_openai_router` 测试(含 mock OpenAI server、OpenAI 模式 RouterManager 构造与 `/v1/models` 断言)。

## 影响与回归风险

- 影响窗口:含 #11687 改动的版本受影响(依据正文 revert 声明;正文未给出具体版本号)。
- 回归面:改动触及 sgl-router 的 `/v1/models` 端点处理与 Rust 测试。恢复后 OpenAI 模式下单 router 场景不再委托,`get_models` 一律走 worker_registry 聚合——#11687 原本要解决的 openai router `/v1/models` 问题会重新存在;其他路由模式的该端点行为不变。

## 关联

- [#11687](https://github.com/sgl-project/sglang/pull/11687)(标题中引用的被回滚 PR;正文未另行列出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
