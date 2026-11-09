# PR #24559: [BugFix][Multi Modal] Fix TensorSchema shape mismatch in Molmo

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24559](https://github.com/vllm-project/vllm/pull/24559) |
| 作者 | wwl2755 |
| 合并日期 | 2025-09-10 |
| labels | ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

Molmo 图像输入的 crop 数在不同图片间是动态的,TensorSchema 却按固定维校验导致 shape mismatch;将 nc 维声明为动态修复。

## 问题现象

正文对现象描述较简短:不同图像的 crop 数量各不相同,需要将其作为动态维度处理。可推断的触发场景(结合修复内容,由 diff 推断):向 Molmo 模型(allenai/Molmo-7B-D-0924)发送包含图像的请求时,MolmoImageInputs 的 TensorSchema 形状校验因 crop 数维度不匹配而报错。作者用 api_server 启动 Molmo-7B-D-0924 并以 vllm bench serve 在 VisionArena-Chat 数据集上跑 10 条请求验证,修复后 10 条请求全部成功(吞吐 8.70 req/s,Mean TTFT 695.62 ms);正文未贴出修复前的报错日志。

## 根因

正文披露的根因:Molmo 的 crop 数量在不同图像之间是动态变化的,模型输入定义必须相应地做成动态。具体到失败机制,diff 显示 `MolmoImageInputs` 的 `TensorShape` 原先把 `nc`(Number of crops)当作与其他维一样需严格匹配的维度声明,`images`、`image_masks`、`feat_is_patch` 三个字段均如此,实际输入 crop 数不一便会触发 shape mismatch 校验失败(由 diff 推断)。

## 修复方案

只改 `vllm/model_executor/models/molmo.py` 的 `MolmoImageInputs` TensorSchema 声明:

1. `images` 与 `image_masks` 的 `TensorShape` 增加 `dynamic_dims={"nc"}`,并在注释中说明 "Number of crops may vary per batch and image, so pass it as a list."(crop 数可逐 batch、逐图像变化,故以 list 传入);维度文档同步标注 nc 为 dynamic。
2. `feat_is_patch` 除加 `dynamic_dims={"nc"}` 外,其形状维度由 `("bn", "nc", "np")` 改为 `("bn", "nc", "tp")`,新增维度说明 `tp: Token sequence positions`,并对齐了注释排版。

关键 diff 摘录:

```diff
-                      TensorShape("bn", "nc", "np", "pd")]
+                      TensorShape("bn", "nc", "np", "pd", dynamic_dims={"nc"})]
+    # Number of crops may vary per batch and image, so pass it as a list.
```

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,改动仅涉及 Molmo 模型输入的类型/形状声明,不触及运行时计算逻辑;受影响路径是使用 MolmoImageInputs 校验的多模态输入预处理。`feat_is_patch` 的维度名由 `np` 改为 `tp` 改变了校验预期,若上游构造的 feat_is_patch 尺寸与 token 序列长度不一致,校验会显式报错。其余模型的 TensorSchema 不受影响。

## 关联

- https://github.com/vllm-project/vllm/issues/24544 (Fix)
- https://github.com/vllm-project/vllm/pull/22022 (Related)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
