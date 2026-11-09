# PR #45399: [Perf] glm-5.3: Encode GLM-5.3 Python chat template prompts segment by segment

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Dovis01 |
| 合并 | 2026-10-24 |
| 分类 | KV Cache / Memory (tier: major,classify.py 给出;实际为 tokenizer/preprocess 路径优化) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,7 个文件,多为测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/45399 |

## 一句话结论

GLM-5.3 聊天模板按段交出 prompt 并分段编码 + LRU 缓存,agent 型 8x3 工具请求 render+encode 快 6.5x(热路径 32.6x),E2E TTFT 最多降 23%。

## 动机

GLM-5.3 Python 聊天模板仍把 prompt 渲染成一个字符串,`parallel_prompt_encode` 需要重新扫描文本找 added token 切分点,且只在 32,768 字符(`SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS`)以上才启用并行,短 prompt 走单次 tokenizer 调用;每个请求都全量重编码,而 function-calling/agent 负载里 system prompt、工具列表、前几轮对话在请求间大量重复。渲染器其实早就知道每个 turn 特殊 token 的位置,却没有交给编码器利用。

## 优化点

四个组件:(1) `glm53_chat_template.py` 的 `render_glm53_segments`/`render_glm53_flash_segments` 返回拼接前的段列表,`[gMASK]`、`<sop>`、`<|system|>`、`<|user|>`、`<|assistant|>`、`<|observation|>` 与生成 prompt 的 `<think>` 各自成段,join 后与原渲染逐字节一致;(2) `PythonChatTemplate.render_segments` 与新增 `render_chat_segments` 共享一条渲染路径,`template_kwargs` 含 `apply_chat_template` 自身参数(`chat_template`、`continue_final_message`、`documents` 等)时返回 None 走原路径;(3) `encode_prompt_segments`:整段 added token 若不可能被其他 added token 跨越匹配(`_cuts_at`,每 token 只查一次)即为切点,切点间文本不论长度都走一次 `encode_batch` 且不再扫描整条 prompt,ids 存入 per-tokenizer LRU(总预算 1M token、单条上限 128K token),未缓存且超过 min-chars 的长文本仍用 `split_prompt` 并行,其余情形一律回落 `parallel_prompt_encode`;(4) `serving_chat.py` 在 Python 模板激活时取 segment 路径,Jinja 与其他模型不变。

## 关键改动

```text
python/sglang/srt/parser/glm53_chat_template.py: _render 返回段列表,render_glm53(_flash) 由 join 封装。
python/sglang/srt/parser/python_chat_template.py: PythonChatTemplate.render_segments、render_chat_segments。
python/sglang/srt/entrypoints/openai/parallel_prompt_encode.py: _SegmentEncoder(_cuts_at 切点 + LRU 缓存)与 encode_prompt_segments。
python/sglang/srt/entrypoints/openai/serving_chat.py: _render_and_encode_chat_template 接入 segment 路径。
test/registered/unit/entrypoints/openai/test_parallel_prompt_encode.py: 新增 TestPromptSegments(fuzz 1500 组段列表)。
test/registered/unit/parser/test_glm53_chat_template.py、test_serving_chat.py: 模板段与 serving 接线测试。
```

## 性能声明

render+encode 每请求中位耗时(GLM-5.3-Flash tokenizer,末条消息每次不同;cold 无缓存,warm 工具列表与前几轮已缓存):agent 8 tools、8 轮 x 3 并行调用(25,053 字符/5,276 token)main 7.596 ms -> 本 PR cold 1.160 ms(6.5x)、warm 0.233 ms(32.6x);agent 18 轮 warm 1.559 -> 0.357 ms(4.4x);128 工具(82,925 字符)main 2.835 -> warm 1.003 ms(2.8x,cold 持平因 main 已超阈值并行);8 工具 1 轮 warm 0.083 ms(8.8x)。E2E TTFT p50(GLM-5.3-Flash FP8,TP4/EP4,4xB200,EAGLE MTP 5/1/6 adaptive,流式 max_tokens 512、temp 1.0,radix 前缀缓存热、每请求唯一 nonce 使整请求 render cache miss):128 工具 c=1 185.3 -> 159.1 ms(-14.1%),c=16 550.3 -> 422.9 ms(-23.1%);agent 18 轮 -15.3%/-14.3%;8 工具(606 token)TTFT 基本不变。环境:Intel Xeon 6960P,Python 3.12.15,transformers 5.19.0,tokenizers 0.23.2。

## 验证与回归风险

验证:真实 tokenizer(GLM-5.3 与 GLM-5.3-Flash)上 6,608 次 token-id 对比 0 不匹配(含特殊 token 拼写、切点旁的部分拼写、unicode、150K 字符文档、128 工具 prompt、缓存二次编码;600 例另与 Jinja 对比);在线 A/B 24 场景(工具模式、乱序工具结果、图片、流式、/v1/responses、/v1/messages、400 错误)prompt_token_ids/状态/输出完全一致;新增单测覆盖 id 一致性、无阈值单次编码、长文切分、可重叠 token 不作切点、回退条件与缓存预算。CI:Base 通过,Extra 与 AMD ROCm 10 标红。风险:(1) cut point 判定 `_cuts_at` 依赖 added token 集合的静态分析,新增 normalized added token 或 Metaspace 预分词器走回退路径;(2) LRU 缓存按文本精确匹配,每进程 1M token 预算;(3) 改动位于所有 Python-chat-template 请求的热路径,回退逻辑(test 覆盖)是正确性兜底。

## 关联

前序:PR #45154(GLM-5.3 模板/parallel_prompt_encode 既有工作,本 PR 为其 follow-up);同区域 tokenizer/encode 优化:PR #17387(model gateway special token 搜索优化)、PR #12001(TypeBasedDispatcher)。本批次内无其他相关卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
