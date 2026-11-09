# RFC #43235: [RFC] NSA is architecturally incompatible with NVIDIA consumer/workstation GPUs — MLA is the only viable sparse attention path

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43235](https://github.com/vllm-project/vllm/issues/43235) |
| 作者 | piweb-labs |
| 创建日期 | 2026-05-20 |
| 状态 | ⚪ closed(关闭于 2026-09-19) |
| labels | RFC, stale |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
论证 NSA 在 NVIDIA 消费级/工作站 GPU(SM120)上存在架构性不兼容,提议社区转向 MLA 作为唯一可达生产级可靠性的稀疏注意力路径。

## 提案动机
作者在 SM120(RTX PRO 6000 Blackwell,8×96GB,PCIe Gen5)上大量生产测试后认定 NSA 无法可靠用于生产服务——不是 bug 而是架构不兼容:1) NSA 每步 decode 需做动态稀疏索引(per-layer、per-token 对全 KV 做 topk),SM100 有专用 FP8 block-scaled GEMM 单元而 SM120 没有,软件回退既慢又有正确性问题;2) PCIe 拓扑让 NSA+MTP 极不可靠——b12x PCIe oneshot allreduce 在 CUDA graph/no-copy 模式下有 buffer 复用竞争,已知修复(pcie_oneshot.cu 加完成 barrier)代价约 1 tok/s 且并发下仍不保证正确,MTP + TP>1 + 并发>1 会死锁(#41402、#41404 均未解决即关闭);3) SM120 不是 SM100:SM100 的 FP8 block-scaled GEMM kernel 在 SM120 上崩溃或输出垃圾(#26211、#35566),SM120 上 FP8 KV 比 BF16 还慢,SGLang 的 NVFP4 KV PR(#21601)仍带严重 bug 未合并;4) NSA 正确性脆弱:voipmonitor/GLM-5.1 参考镜像仅 93.3% 准确率(28/30),需含 3 个补丁源文件的自定义 b12x overlay 加自定义 vLLM 树;5) 社区所有 NSA+MTP Docker 镜像要么挂起要么输出混乱。MLA 无这些问题:结构化(低秩投影)压缩 KV、标准 attention 路径、兼容既有 FP8/BF16 KV 后端、MTP 集成直接、多数 workload 压缩率相当或更好。7+ 天测试所有 NSA+MTP 镜像无一生产可用。

## 提出的方案
1) 停止在 SM120 GPU 上投入 NSA kernel 开发;2) 把 MLA attention 后端作为 vLLM 主要稀疏/高效注意力路径优先支持,含 SM120 一等支持:原生 FP8 KV cache 的 MLA decode、MLA + MTP 投机解码集成、MLA prefix caching 与 chunked prefill;3) 文档标注 SM120 上的 NSA 为实验性/best-effort,免得其他团队重蹈 7 天弯路;4) 重新评估 b12x NSA indexer 集成是否应留在 vLLM,或移入独立实验包以免动摇主分支。

## 优化点与收益
正文未给出 MLA 的量化性能对比;引用数字均为问题证据:NSA 参考镜像准确率 93.3%(28/30)、b12x 修复代价约 1 tok/s、7+ 天无一镜像生产可用。

## 现状与落地
已关闭(2026-09-19),labels 含 stale;正文未说明提议是否被采纳,落地情况需查关联 PR。

## 关联
- #41402、#41404、#26211、#35566、SGLang #21601

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
