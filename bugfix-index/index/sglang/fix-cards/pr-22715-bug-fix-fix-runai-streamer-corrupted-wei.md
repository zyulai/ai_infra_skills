# PR #22715: [Bug Fix] Fix RunAI streamer: corrupted weights, missing quant init, and broken URIs for multimodal models

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22715](https://github.com/sgl-project/sglang/pull/22715) |
| 作者 | junliu-mde |
| 合并日期 | 2026-05-07 |
| labels | deepseek, run-ci |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
RunAI streamer 配多 GPU 量化/多模态模型有三个独立 bug(漏传 quant_config、跨 batch 复用 buffer 导致权重损坏、get_processor 不解析对象存储 URI),按根因分四个 commit 修复。

## 问题现象
作者在 8×H200 集群上测试 #17948(Runai Model Streamer 从对象存储直接加载)时发现:用 `az://` 路径、TP=8 加载 Kimi-K2.5(WNA16 Marlin MoE)与 DeepSeek-V3-0324(BF16 MoE)都失败。三个独立 bug:1) 量化模型以 BF16 初始化,导致 OOM 或推理错误——DeepSeek-V3-0324(FP8)在 `_initialize_model()` 期间 OOM;2) Kimi-K2.5 server 能启动,但所有 chat 回复都是 `!`(token 0,来自 stale buffer 的 all-zero logits);3) 多模态模型从对象存储加载时,`get_processor()` 直接崩溃。

## 根因
正文披露三个独立根因:1) `RunaiModelStreamerLoader` 初始化模型时漏传 `quant_config`,量化模型被当作 BF16 分配权重;2) RunAI 分布式 streamer 跨 batch 复用 GPU staging buffer,而两条代码路径(Kimi-K2.5 的 list 累积、DeepSeek MLA 的 `cached_a_proj`)把引用保留到下一个 batch,引用背后的 buffer 被改写,权重静默损坏;3) 与 `get_config()`/`get_tokenizer()` 不同,`get_processor()` 把原始 `az://`/`s3://`/`gs://` URI 直接传给 HuggingFace,不解析对象存储路径。

## 修复方案
按正文,四个原子 commit 对应四个根因:1) `loader.py`:`RunaiModelStreamerLoader.load_model()` 给 `_initialize_model()` 传 `quant_config`,与其他 loader 一致;2) `deepseek_weight_loader.py`:`cached_a_proj` 缓存的 tensor `.clone()`,使其跨 batch 存活;3) `kimi_k25.py`:语言权重 `load_weights()` 从「累积后处理」重构为 streaming pattern,只 clone 需要批处理的小 vision 权重;4) `hf_transformers_utils.py`:给 `get_processor()` 加 `is_runai_obj_uri()`/`ObjectStorageModel.get_path()` 归一化。diff 可见部分为 `hf_transformers/` 重构:common.py 新增公共 helper,config/tokenizer/processor 三处统一调用:

```python
def resolve_runai_obj_uri(model_name_or_path: str) -> str:
    if is_runai_obj_uri(model_name_or_path):
        return ObjectStorageModel.get_path(model_name_or_path)
    return model_name_or_path
```

注:diff 摘录未包含 loader.py、deepseek_weight_loader.py、kimi_k25.py 三处改动,相关修复以正文表格为准。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:RunAI streamer 的模型初始化路径(量化模型);deepseek 与 kimi_k25 的权重加载逻辑(kimi 的 load_weights 结构性重构);`get_config()`/`get_tokenizer()`/`get_processor()` 三处 URI 解析统一为公共 helper。修复后:DeepSeek-V3-0324 FP8 约 386s 加载成功(8 GB streamer 内存上限)且推理正确;Kimi-K2.5 多轮对话、reasoning 与 vision 响应正常。

## 关联
- https://github.com/sgl-project/sglang/issues/22701
- https://github.com/sgl-project/sglang/pull/17948

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
