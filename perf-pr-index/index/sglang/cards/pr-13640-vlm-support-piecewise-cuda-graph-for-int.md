# PR #13640: [VLM] Support Piecewise CUDA Graph for InternVL

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-11-21 |
| 分类 | CUDA Graph / Compiler | tier: major |
| Label | performance, Multi-modal, run-ci, vlm |
| Diff | diffstat 缺失(素材未提供统计数字);diff 含 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/13640 |

## 一句话结论

让 InternVL 接入 external multimodal 预处理与 Piecewise CUDA Graph,补齐该 VLM 的预填充加速路径。

## 动机

InternVL 此前无法启用 Piecewise CUDA Graph:其 forward 依赖 `general_mm_embed_routine`,图像特征嵌入发生在语言模型 forward 内部,与 piecewise CUDA graph runner 的捕获/回放机制不兼容。PR 是 #13055(Qwen-VL 同类支持)的后续,用于落实 #12838 提出的需求,把 InternVL 加入支持列表。

## 优化点

机制上做了两件事:(1) 把 InternVLChatModel 加入 `should_use_external_mm_preprocess` 名单,图像特征嵌入从模型 forward 内移到 model_runner 的 `forward_extend` 阶段由 `external_mm_preprocess_routine` 完成;新增 `resolve_external_mm_data_embedding_funcs()`,若模型在 `__init__` 里显式暴露 `external_mm_data_embedding_funcs` 映射(如 `{Modality.IMAGE: get_image_feature}`)则采用之,否则回退 routine 内部默认逻辑(Qwen2.5-VL 路径)。(2) 重写 `InternVLChatModel.forward`:不再走 `general_mm_embed_routine`,直接取 `forward_batch.input_embeds`(extend 场景由外部预处理预嵌入)传入 language model;decode 场景仍传 input_ids。注释说明该双入口设计是为兼容 model_runner 直调与 piece_wise_cuda_graph_runner 捕获回放两条调用路径。另外把 `self.model` 指到 `self.language_model.model`,配合 graph runner 的模块定位。

## 关键改动

```text
python/sglang/srt/models/internvl.py: 暴露 embedding funcs 映射,forward 改为直接调用 language model
python/sglang/srt/managers/mm_utils.py: InternVL 加入 external preprocess 名单,新增 resolve_external_mm_data_embedding_funcs
python/sglang/srt/model_executor/model_runner.py: forward_extend 解析并传入 data_embedding_funcs
test/srt/test_piecewise_cuda_graph.py: 新增 InternVL2.5-8B 的 piecewise CUDA graph 准确性测试
test/srt/run_suite.py: test_piecewise_cuda_graph.py 预估耗时 750→850
```

关键片段(internvl.py):

```python
self.external_mm_data_embedding_funcs = {
    Modality.IMAGE: self.get_image_feature,
}
self.model = self.language_model.model
...
input_embeds = forward_batch.input_embeds
hidden_states = self.language_model(
    input_ids=input_ids,
    forward_batch=forward_batch,
    input_embeds=input_embeds,
    positions=positions,
)
return hidden_states
```

## 性能声明

PR 正文未提供加速数字(Benchmarking 章节留空),仅给准确率验证:OpenGVLab/InternVL2_5-26B,TP=2,`--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler eager --piecewise-cuda-graph-max-tokens 8192 --disable-radix-cache`,lmms_eval mmmu_val(900 题):mmmu_acc 0.5122,作者称"The accuracy result is as expected"。

## 验证与回归风险

新增 `TestPiecewiseCudaGraphInternVL25`:以 InternVL2_5-8B 启动 server(`--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler eager --disable-radix-cache`),跑 2000 例 mgsm_en 评测并断言 score ≥ 0.70。回归风险:改动了 `general_mm_embed_routine` 之外的 VLM 嵌入路径,InternVL 的 extend 输入从 input_ids 换成预嵌入 input_embeds,需保证 non-piecewise 路径同样正确;`resolve_external_mm_data_embedding_funcs` 对未来暴露同名属性但语义不同的模型可能产生误匹配(Qwen3_VL 被列为 TODO);mm_utils 名单机制按类名字符串匹配,派生模型类名不在名单内则不生效。

## 关联

前序:#13055(Qwen-VL piecewise CUDA Graph,本 PR 为其 follow-up)、#12838(需求来源)。同区域:#12996(DeepSeek piecewise CUDA Graph)、#10062(piecewise CUDA graph + torch.compile)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
