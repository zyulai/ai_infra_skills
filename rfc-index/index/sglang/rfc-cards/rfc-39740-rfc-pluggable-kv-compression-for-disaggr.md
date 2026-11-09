# RFC #39740: [RFC] Pluggable KV Compression for Disaggregated Serving

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39740](https://github.com/sgl-project/sglang/issues/39740) |
| 作者 | luoroger37 |
| 创建日期 | 2026-09-16 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Disaggregated Serving |

## 一句话结论
为 Mooncake RDMA 的 P/D 传输提出可选 KV 压缩路径,用小型内部后端接口解耦压缩算法与传输管线,默认关闭,先验证集成边界。

## 提案动机
P/D serving 中 KV 传输会延迟 prefill→decode 交接;压缩能减少传输字节,但增加计算、内存流量、同步与临时存储,是否有益取决于实际 KV 分布、有效带宽与并发模型执行。把每个压缩算法直接接入传输路径会重复 page extraction、buffer 管理、长度发布、恢复与失败处理;一个小的共享接口可让机制留在 SGLang、数据变换独立替换与评估,其价值必须由真实后端证明而非来自通用性假设。

## 提出的方案
- `KVCompressionBackend` Protocol(`supports` / `buffer_requirements` / `compress_into` / `decompress_into`)+ `CompressionOperation.poll()`;`BatchSpec` / `SupportResult` / `BufferRequirements` 描述块数、尺寸、对齐与元数据要求;buffer 由调用方持有;`OperationSnapshot` 将 outcome 与 quiescent(不再访问所给 buffer)分离
- layout adapter 把选中 KV 组织为单层、单本地分片内独立可恢复的 block(一 block 可含多 page);后端只变换 packed block,不遍历缓存树、不注册 RDMA、不解释请求状态;P/D adapter 持有传输描述符、buffer lease 与恢复完成
- 表示元数据(UNCOMPRESSED/COMPRESSED、格式与版本、块边界、编码/解码长度)与传输 envelope 分离;仅当压缩后完整 wire size 更小时才选压缩,不可压块走 `UNCOMPRESSED` 同一 lease/恢复生命周期
- 传输执行复用现有 Mooncake engine 与 P/D work queue(有界注册字节池);restore-pending 映射 `KVPoll.Transferring`、失败映射 `Failed`,恢复完成前不报告成功
- 资源生命周期与失败处理:源 KV 保护到最后一次读;无法确认 quiescence 时隔离受影响 buffer,预算耗尽则停止相关 admission 或失败会话;部分恢复永不发布有效缓存条目
- 初始后端为 GPU LZ4(nvCOMP,固定依赖版本);首个验证目标:Qwen3-8B、BF16 KV、1 P worker + 1 D worker、TP/PP/CP/DCP=1;默认关闭、nvCOMP 为可选依赖;压缩 payload 复用与 HiCache 集成为后续独立扩展

## 优化点与收益
正文明确「No compression ratio or serving speedup has been established」。示意计算(照抄,作者注明非测量或可压缩性预测):Qwen3-8B(36 层、8 KV heads、head dim 128)BF16 K+V 每 valid token 144 KiB,8,192 tokens TP=1 约 1.125 GiB;若传输字节减少 20%,理想网络节省约 9.66 ms @200 Gb/s 或 4.83 ms @400 Gb/s,且额外关键路径成本必须小于实际节省。

## 现状与落地
仍开放:设计提案,评审点为集成边界与验证计划;压缩默认关闭,禁用路径不分配压缩 buffer。(注:pack 正文在约 2 万字符处截断,其后内容未纳入本卡。)

## 关联
- Qwen3-8B config.json(huggingface)
- nvCOMP 文档(docs.nvidia.com)
- 集成基于 upstream commit 444b29c93225

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
