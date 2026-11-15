# PR #57796: [Frontend][Multimodal] Accept features together with content_parts in /generate

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57796](https://github.com/vllm-project/vllm/pull/57796) |
| 作者 | aoshen02 |
| 合并日期 | 2026-11-12 |
| labels | bug, documentation, frontend, ready, multi-modality |
| 热度 | 👍 0 · 💬 52 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
/generate 强制 content_parts 与 features 互斥,多轮多模态 RL 两头不适用;放开组合,features 前缀保留、仅渲染其后段落。

## 问题现象
多轮多模态 RL 场景(`/inference/v1/generate`,token-in-token-out):到第 N 轮,prompt 已包含此前已展开的图像(来自上一响应的 `prompt_token_ids` / `mm_placeholders`)加一张带紧凑 placeholder 的新图。原有两个互斥输入都不适用:只给 `content_parts` 时 processor 匹配到的前面的 placeholder tokens 落在早期展开区间内,除非调用方先折叠所有早期 run,prompt 会被静默破坏;只给 `features` 时新图没有 processed data。main 上同时给两者被拒(`content_parts and features are mutually exclusive`)。测试:Qwen3-VL-2B-Instruct on 1x GB200(`vllm/vllm-openai` nightly `18f8f96` + 本 diff)6 passed;无改动时 `test_features_with_content_parts` 失败(400 Bad Request)。

## 根因
正文披露:`GenerateRequest` 的 model validator 把 `content_parts` 与 `features` 强制互斥;本 PR 早期版本曾试图在 processor 里探测 expansion,与维护者讨论后改为让两个字段协作——`features` 携带已展开的早期条目及显式 range,`content_parts` 只承载新条目。

## 修复方案
(1) `vllm/entrypoints/scale_out/token_in_token_out/protocol.py` 删除 `_check_mm_fields_exclusive` 校验,字段 docstring 改为 content_parts 的 placeholder 必须位于最后一个 features range 之后;(2) `serving.py` `serve_tokens`:两字段同给时计算最后一个 features range 的终点 `start = max((p.offset + p.length for rs in feature_ranges for p in rs), default=0)`,只把 `request.token_ids[start:]` 交给 renderer(matcher 看不到展开区间),再把渲染结果与 features 前缀拼接——`token_ids = token_ids[:start] + rendered["prompt_token_ids"]`,mm_placeholders 用 `replace(p, offset=p.offset + start)` 平移后 append,mm_kwargs/mm_hashes 同步 extend;渲染抛 `PromptUpdateMismatchError` 且带 features 时转成 `VLLMValidationError`(400,提示 "content_parts placeholders must come after the last features range");(3) `vllm/exceptions.py` 新增 `PromptUpdateMismatchError` 类;(4) `vllm/multimodal/processing/processor.py` 把未匹配条目的 `assert`(500)改为 raise `PromptUpdateMismatchError`。新增测试 `test_features_with_content_parts`(cached 两种参数化)、`test_content_parts_before_features_is_rejected`(400)。

## 影响与回归风险
- 影响窗口:正文未给出。
- 回归面:`/inference/v1/generate` 的协议校验与 multimodal 处理路径;只用单一字段的请求行为不变(正文声明);新 placeholder 位于最后一个 features range 之前时请求得到 400(原为互斥校验拒绝);文档明确媒体插入在 prompt 开头的模型(PaliGemma、BLIP-2、Molmo)会把新条目紧跟 range 放置而不报错,不应组合两字段;`--mm-processor-kwargs`/`--media-io-kwargs` 会把 `uuid` 与它们一起 hash、不再是缓存 key(文档提示改用 render 响应取 features);条目被缓存逐出时请求失败,需用 `kwargs_data` 重发;processor 的未匹配条目由 500 assert 变为 400 校验错误。

## 关联
- [#53169](https://github.com/vllm-project/vllm/pull/53169)(重处理 raw images 缩 payload,不组合两输入)
- [#54548](https://github.com/vllm-project/vllm/pull/54548)(is_embed round trip,orthogonal;未合并前 features 条目照旧丢失 is_embed)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
