# PR #7916: [Bugfix] Fix phi3v incorrect image_idx when using async engine

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7916](https://github.com/vllm-project/vllm/pull/7916) |
| 作者 | Isotr0py |
| 合并日期 | 2024-08-27 |
| labels | ready |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
phi3v 输入处理器在无 prompt 时 `image_idx` 错为空列表,异步引擎下多图占位全失配;改为按图片数生成序号。

## 问题现象
正文描述:不正确的 `image_idx` 会破坏 phi3-vision 模型的服务部署——具体是那些不把 prompt 直接传给 `input_processor` 的请求路径,即异步引擎(async engine)场景。bug 由 #7718 报告。正文未描述具体日志或复现步骤。

## 根因
正文未披露根因(PR 描述为模板未填写,仅一句话指向 #7718)。(由 diff 推断)`input_processor_for_phi3v` 中,当 `llm_inputs` 里没有 `prompt`(异步服务请求只带 token_ids)时,代码走 `image_idx = []` 分支,即认为没有任何图片占位序号;而 prompt 路径会从 prompt 文本里用正则提取 `<|image_(\d+)\|>` 序号。对异步请求,token_ids 里实际含有多图占位符,却拿到空 `image_idx`,导致占位符与 `image_data` 无法配对。diff 注释还说明其假设:「for async server request, we assume prompt and its token_ids is always in correct format. And num_image_tags == len(image_data) always True.」

## 修复方案
改 `vllm/model_executor/models/phi3v.py` 的 `input_processor_for_phi3v`:prompt 为 None 时不再把 `image_idx` 置空,而是按图片数量生成连续序号:

```python
if prompt is None:
    # for async server request, we assume prompt and its token_ids is always
    # in correct format. And num_image_tags == len(image_data) always True.
    image_idx = range(1, len(image_data) + 1)
    new_prompt = None
else:
    image_idx = sorted(map(int, re.findall(r"<\|image_(\d+)\|>+", prompt)))
```

即异步路径下按 1..N 为每张图分配 `image_idx`,与 prompt 路径的正则提取结果语义一致。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:仅改动 `vllm/model_executor/models/phi3v.py` 的输入处理器中 `prompt is None` 分支,影响通过异步引擎提交、不带原始 prompt 的 phi3v 多模态请求;带 prompt 的路径逻辑不变。修复依赖「token_ids 格式正确、图片占位数量等于 `len(image_data)`」这一假设,若请求违反该假设,行为与修复前不同(修复前空列表同样会失配)。

## 关联
- #7718(bug 报告)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
