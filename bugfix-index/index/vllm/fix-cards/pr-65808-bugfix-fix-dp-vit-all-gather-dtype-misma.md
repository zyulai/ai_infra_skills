# PR #65808: [Bugfix] Fix DP ViT all-gather dtype mismatch when the first parameter is fp32

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65808](https://github.com/vllm-project/vllm/pull/65808) |
| 作者 | micah-wil |
| 合并日期 | 2026-11-06 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

DP ViT 占位 embedding 取首个参数 dtype,fp32 patch embedding 与 bf16 输出不一致致 all-gather 挂死;改用模型 `dtype` 属性修复。

## 问题现象

使用 `--mm-encoder-tp-mode data` 时,`amd/MiniMax-M3-MXFP4` 以 `-tp 4` 启动会在 encoder profile run 之后**挂死**,无法继续启动并提供图像请求服务;修复后可正常启动并服务 image requests。触发条件:部分 rank 收不到图像,它们会 all-gather 占位 embedding,而这些占位张量的 dtype 与持有图像 rank 的输出 dtype 不一致,集合通信无法完成。

## 根因

PR 正文披露:在 `--mm-encoder-tp-mode data` 下,没有图像的 rank 会 all-gather 占位 embedding,其 dtype 取自 `next(vision_model.parameters())`(首个参数);自 #64111 起 MiniMax-M3 的 patch embedding 保持 fp32,而 vision tower 输出是 bf16。于是有图像的 rank gather 的是 bf16、无图像的 rank 提供的是 fp32,all-gather dtype 不匹配导致启动挂死。其他模型不受影响:它们的 `dtype` 属性要么与首个参数一致,要么未定义。

## 修复方案

改 `vllm/model_executor/models/vision.py` 的 `run_dp_sharded_mrope_vision_model`:优先使用 vision model 的 `dtype` 属性,没有该属性时回退到首个参数的 dtype,保证占位 embedding 与有图像 rank 的输出 dtype 一致:

```diff
-    embed_dtype = next(vision_model.parameters()).dtype
+    embed_dtype = (
+        getattr(vision_model, "dtype", None) or next(vision_model.parameters()).dtype
+    )
```

## 影响与回归风险

- 影响窗口:由 #64111 引入 fp32 patch embedding(MiniMax-M3)之后、`--mm-encoder-tp-mode data` 多 rank 部分的提交范围受影响。
- 回归面:仅 `run_dp_sharded_mrope_vision_model` 的占位 embedding dtype 选择;优先取 `dtype` 属性,对 `dtype` 与首参数一致或无 `dtype` 属性的模型行为不变。持续覆盖来自 nightly MiniMax-M3 评测(带 `--mm-encoder-tp-mode data` 运行)。

## 关联

- https://github.com/vllm-project/vllm/pull/64111

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
