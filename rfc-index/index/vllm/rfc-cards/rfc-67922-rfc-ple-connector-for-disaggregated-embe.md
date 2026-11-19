# RFC #67922: [RFC]: PLE Connector for Disaggregated Embedding Serving

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | RulinJuice |
| 创建日期 | 2026-11-16 |
| 状态 | 🟢 open |
| labels | kv-connector, quantization |
| 热度 | 👍 2 · 💬 2 |
| 主题分类 | Disaggregated Serving |
| 链接 | [#67922](https://github.com/vllm-project/vllm/issues/67922) |

## 一句话结论
提出 PLE Connector:把 disaggregated 部署中的 vLLM prefill 实例接入一个共享 PLE(per-layer embedding)查表服务,在 prefill 执行前异步准备好 embedding 行,避免在 prefill 实例间复制超大 N-gram embedding 表,并让查表故障无需重启推理 worker 即可恢复。

## 提案动机
Qwen 模型使用很大的 N-gram embedding 表:开放权重的 Qwen3.8-Flash-Next 包含 51B embedding 参数,Qwen4-Max 报告 328B(其发布文案)。按 FP8 存储(每参数 1 字节)估算,这两张表分别约需 51 GB 与 328 GB(还不含量化 scale、元数据与运行时缓冲)。在 prefill 实例间复制这些表会成倍放大存储成本,即使 offload 到主机内存也一样。规模化之后,查表失败的影响应有限,恢复不应要求重启推理 worker。为此,提案设计一个 PLE Connector,把 vLLM prefill 实例连接到共享查表服务,并在 prefill 执行之前异步准备 embedding 行。

## 方案
RFC 面向 disaggregated prefill/decode 部署中的 prefill 实例,要点如下:

- **共享查表服务**:由一个独立服务持有版本化、不可变的 PLE 表,为多个 prefill 实例执行稀疏行查找;它可对表分片或复制,并独立于推理算力扩缩。vLLM 内部,PLE Connector 通过 provider 接口支持本地或远程查找,负责准备工作、重试与一个有界(bounded)结果池;provider 按请求的 row-ID 顺序返回原始行,模型特定的哈希、反量化、投影与门控(gating)仍留在 vLLM 侧。采用远程查找时,prefill 实例本地只保留已准备好的行。
- **高可用**:查表服务器故障时,connector 重试查找即可恢复,无需重启推理 worker——简化大规模生产部署的 failover。
- **计算准入前准备(prepare before compute admission)**:PLE Connector 运行在 EngineCore,用已知的 prefill token ID 在后台准备所需行:Scheduler 请求准备 prompt 所需行 → connector 后台查找(期间调度其他就绪请求)→ 主机行就绪后 connector 通知 scheduler → scheduler 复查覆盖率与调度预算 → 准入并下发 chunk → worker 对该 chunk 异步 H2D,可与早层计算重叠,到 PLE 消费点再等待拷贝;后续 chunk 复用已准备好的主机行。prefix cache 复用后,connector 会准备完整剩余 prompt 后缀以便跨 chunk 复用;replay 或抢占发生后需要重新检查覆盖率。
- **正确性与资源约束**:provider 启动时绑定表版本与布局,重试必须保持该身份;所有请求的行都写完后结果才可见;准备工作带字节预算与 deadline(含重试),未恢复的失败作为请求错误向上传播;完成或取消时,结果在未完成的 provider 写入与所有 worker 拷贝结束前保持分配;device staging 缓冲同样受保护,直到前一个 GPU 消费者完成。这些要求与存储和传输后端无关。

## 优化点与收益
- 省存储:避免在 prefill 实例间复制约 51 GB(Qwen3.8-Flash-Next,FP8 估算)乃至 328 GB(Qwen4-Max)量级的 N-gram embedding 表,共享服务可分片/复制并独立扩缩。
- 高可用:查表服务器故障由 connector 重试兜底,推理 worker 不重启。
- 延迟友好:计算准入前在 EngineCore 后台异步准备行,H2D 可与早层计算重叠,后续 chunk 复用已准备的主机行。
- 正文未给出端到端性能量化数据(pack 未提供)。

## 现状落地
- 状态 open(2026-11-16 创建,👍 2 · 💬 2);正文未给出已合并实现或已提交的实现 PR。
- 相关工作:#59958(PLE disk offload)与 #66941(next-chunk prefetch)提供行读取与重叠的相关机制;#64002(shared host tables)让 worker 映射共享表存储并自行执行查找;#67539(cross-engine Engram pool)把表分片分布到各推理引擎——本 RFC 的差异在于由独立查表服务持有表,并把 gather 后的行返回给 prefill 实例。
- 计划扩展:把远程 PLE 查找扩展到 decode 实例(重点关注每 token 延迟);在远程查表服务器上用 GPU 加速行 gather 与结果传输。
- 征询反馈:是否应为 vLLM 暴露 PLE provider 接口(本地/远程表查找、重试由 connector 处理);EngineCore 内、计算准入前做准备是否为合适的集成点。CC @ZJY0516 @youkaichao。

---
> 本卡由 RFC 正文自动素材 + LLM 撰写,未经人工核实。
