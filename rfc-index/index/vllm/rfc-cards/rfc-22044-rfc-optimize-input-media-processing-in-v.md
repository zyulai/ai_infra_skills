# RFC #22044: [RFC]: Optimize Input Media Processing in vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22044](https://github.com/vllm-project/vllm/issues/22044) |
| 作者 | huachenheli |
| 创建日期 | 2025-08-01 |
| 状态 | ⚪ closed(关闭于 2025-12-30) |
| labels | RFC, multi-modality, stale |
| 热度 | 👍 5 · 💬 9 |
| 分类 | Other |

## 一句话结论
提议给多模态输入引入用户提供的 uuid 标识与 URL 级哈希,让重复媒体命中缓存时跳过下载、解码与全量哈希三步开销。

## 提案动机
当前媒体输入(image/video/audio/embeddings)在媒体被重复使用时(多轮对话或不同请求用同一媒体)存在低效:处理流程固定为三步,1) MediaConnector 下载远程 URL 输入,产生网络延迟;2) 把下载字节加载为 Python 对象(如 PIL.Image、音视频解码),产生 CPU 延迟;3) BaseMultiModalProcessor 用媒体全部字节做哈希到 ProcessingCache 查缓存,产生 CPU 延迟。三步都在任何缓存检查之前执行,因此即使最终缓存命中,重复媒体仍要付出全部代价;长视频解码与按全字节哈希可能慢至毫秒到几十毫秒。对重复发送的媒体,目标是三步全部绕过。

## 提出的方案
- 在媒体输入中增加用户提供的 `uuid` 字段作为媒体标识与哈希键,例如 `{"type": "image_url", "image_url": {"url": image_url}, "uuid": "abcde"}`、`{"type": "image_embeds", "image_embeds": "xyz", "uuid": "abcde"}`。
- 带了 uuid 且输入为空也应被允许(即客户端预期缓存命中的场景)。
- 未设置 uuid 的 URL 输入只对 URL 做哈希而不是对全部字节哈希;其他输入回退到对字节全量哈希。
- 输入处理时先查 ProcessingCache,若媒体已缓存则完全绕过 MediaConnector。

## 优化点与收益
正文未给出量化数字。收益为重复媒体场景下跳过网络下载、解码加载与全字节哈希三步;正文提到长视频解码与全字节哈希本身可能慢至「毫秒到几十毫秒」,重复媒体在缓存命中后这些开销可全部省去。

## 现状与落地
已关闭(2025-12-30)。正文提出的两个潜在问题:一是与 OAI API 的兼容性——OAI API 不支持用户提供的 uuid,该字段会加大两者差异,但因是附加字段影响应较小;二是缓存查找与缓存读取之间的竞态(查找后、读取前可能发生逐出),提议的兜底是让该单个请求失败(此点标注需确认)。关闭后的最终落地情况需查关联 PR。

## 关联
- 关联 issue: https://github.com/vllm-project/vllm/issues/21113(正文称可与之互补)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
