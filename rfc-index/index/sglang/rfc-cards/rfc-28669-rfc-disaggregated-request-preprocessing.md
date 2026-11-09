# RFC #28669: [RFC] Disaggregated request preprocessing

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28669](https://github.com/sgl-project/sglang/issues/28669) |
| 作者 | sagearc |
| 创建日期 | 2026-06-18 |
| 状态 | ⚪ closed(关闭于 2026-09-20) |
| labels | (none) |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Disaggregated Serving |

## 一句话结论
把 SGLang 推理用的请求预处理路径做成轻量独立服务,让路由与模型服务拿到与真实执行一致的 generate request。

## 提案动机
模型执行前,服务系统常需知道请求对模型呈现的形态:路由器要在请求到达 GPU 前放置与定量工作(请求多长、前缀是否已缓存、媒体是否已处理);disaggregated pipeline 希望请求只预处理一次、把 token 交给模型,而非每一跳重新 re-tokenize。prepared request 必须与模型实际运行的一致,否则 cache lookup miss、token 计数出错(#26555、#27386、llm-d-router#1072)。本提案把这一"请求视图"在执行前暴露出来,让路由与模型服务不依赖近似。

## 提出的方案
- 把 serving 入口使用的同一条预处理路径(chat template、tools、reasoning、tokenization、media 身份与 token 位置)暴露为 CPU-light 的预处理专用服务,不启动完整引擎/不加载模型;返回完整 generate request(token ids + request metadata + generate 路径所需多模态字段),而非只有 token ids。
- HTTP 上以镜像 serving API 的 /render 端点暴露:/v1/chat/completions/render、/v1/completions/render、/v1/responses/render、/v1/messages/render,请求格式与对应 serving 端点一致;CLI 形态为 sglang serve renderer 或 sglang renderer。
- 现有 /tokenize 不够:只返回 token ids 与数量、无 mm hash/位置、不支持 /responses 与 /messages、且仍需启动引擎并加载模型。
- 多模态:预处理输出 tokenized request、media 身份及每个媒体项在 token 序列中的位置,使 encoder 分离部署可一致路由与复用 encoder 工作;重型媒体数据在独立服务还是 SGLang server 侧生成留作实现选择,关键是路由与执行用同一媒体视图。
- 不做全后端通用预处理服务器:输出必须匹配 SGLang 内部模型输入(chat template、special tokens、多模态占位、内部格式随 backend/模型而异),通用层只能近似。
- 分期:1) 先把预处理收敛为 serving 入口共享的同一路径(行为不变,本身即有价值);2) 再以 opt-in 方式暴露独立预处理服务(精神上跟随 #27606)。

## 优化点与收益
正文未给出量化数字。收益为正确性与一致性:路由/预处理与模型执行共享同一请求视图,避免 cache miss 与 token 计数错误,并在多跳管线中消除重复 re-tokenize。

## 现状与落地
已关闭(2026-09-20)。正文为提案(QA + Phasing),是否实现需查关联 PR。

## 关联
- vLLM #35213(独立/disaggregated preprocessing 讨论)
- vLLM #36102(独立预处理 CLI 入口)
- vLLM #36166(独立预处理实现)
- vLLM #35684(返回含多模态信息的 prepared generate request)
- SGLang #27606(早期 standalone 预处理入口)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
