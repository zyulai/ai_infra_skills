# RFC #30928: [RFC] Position-Independent KV Cache Reuse for Agentic/RAG Workloads

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30928](https://github.com/sgl-project/sglang/issues/30928) |
| 作者 | krakhit |
| 创建日期 | 2026-07-12 |
| 状态 | ⚪ closed (2026-10-13) |
| labels | inactive |
| 热度 | 👍 4 · 💬 13 |
| 分类 | KV Cache / Memory |

## 一句话结论
讨论位置无关 KV cache 复用——相同内容出现在不同偏移也能命中——梳理三种前沿方案并征询社区是否值得在 SGLang 落地。

## 提案动机
RadixAttention 只在前缀字节相同且处于相同绝对偏移时复用 KV cache。Agentic/RAG 流量中,共享系统提示、工具 schema、检索文档与多轮历史常把相同内容放到不同偏移上,内容未变也必然 cache miss。作者对照当前 sglang 代码确认:RoPE 在写入时烘焙进 K、radix cache 哈希链与位置耦合,现有架构不经新工作无法支持上述任何方案。

## 提出的方案
- Irminsul(arXiv:2605.05696,面向 sglang 但暂无代码):利用 MLA 将 KV 分解为位置无关 latent `c_KV` 与小尺寸可旋转切片 `k_r`,在新偏移复用只需对后者做闭式 delta-rotation;MLA 专属,机制可能推广到任意 partial-RoPE(rotary_dim < head_dim)模型
- MiniPIC(arXiv:2606.13126,vLLM/IBM):存未旋转 K、把 RoPE 推迟到 attention 读取时;更通用但触及所有 attention backend、与 fused write-path 内核冲突;其首个实现(IBM/vllm PR #88)因并发复用同一 span 的精度问题被放弃并换用另一策略
- COMB(arXiv:2602.01519):在模型内训练专用 encoder 组件,属第三类路线(改模型而非只改推理引擎)
- 作者倾向:窄化、opt-in 的 delta-rotation 实现,限 MLA/partial-RoPE 模型,以 `--radix-cache-backend` 变体做 feature flag;并征询是作为贡献还是外部实验性扩展更合适

## 优化点与收益
Irminsul 与 MiniPIC 正文未给出量化数字。引用 COMB 论文数字(照抄):51-94% TTFT reduction、3x throughput,测试于 DeepSeek-V2-Lite-Chat。

## 现状与落地
已关闭(2026-10-13)。正文为调研与征询(是否已有人评估、是否欢迎贡献),作者自述「Wanted to check if this is on anyone's radar before digging further」;落地情况需查关联 PR。

## 关联
- Irminsul: arXiv:2605.05696
- MiniPIC: arXiv:2606.13126(代码 IBM/vllm,PR #88)
- COMB: arXiv:2602.01519(代码 github.com/shijuzhao/Comb)
- 相关 issue:#27574 / #24656

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
