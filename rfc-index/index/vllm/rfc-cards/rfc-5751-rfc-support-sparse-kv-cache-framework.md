# RFC #5751: [RFC]: Support sparse KV cache framework

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5751](https://github.com/vllm-project/vllm/issues/5751) |
| 作者 | chizhang118 |
| 创建日期 | 2024-06-21 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 13 · 💬 19 |
| 分类 | KV Cache / Memory |

## 一句话结论
提议通用稀疏 KV cache 框架:以"部分缓存淘汰"为主路线,按优先级排名周期性压缩 KV cache,降低显存占用。

## 提案动机
KV cache 在大模型推理中占用大量 GPU 显存,缩小其体积是重要优化方向。作者梳理三类研究路线:FastDecode(把 KV cache 的计算与存储全部 offload 到 CPU);基于量化的压缩(GEAR、Mixed Precision,减小单 token KV 体积,但会产生需额外存储的 residual/outlier 矩阵);部分 KV cache 淘汰(H2O、SnapKV、LESS、Adaptive Compression、Scissorhands、Dynamic Memory Compression、StreamingLLM,减少存储的 token 数)。作者此前权衡过:量化 vLLM 已实现;类 GEAR 方案需为每次 token 生成 outlier/residual、代价高、不普适;KV cache 累积+附录方案要求模型用同一方法训练、不普适。最终选择部分淘汰路线:所调研的稀疏 KV cache 方法中六种基于淘汰,普适且适合抽象成框架集成进 vLLM。

## 提出的方案
- 参数:可选 flag `--sparse-kv-cache-type` 指定稀疏类型(默认 'auto' 即不启用,可选如 H2O 的 attention scores);淘汰压缩比(正文示例:淘汰 20% 以实现 80% 的 KV cache 使用量削减),并据此计算每 n 步重建一次 KV cache;
- 工作流:首次 decoding 除计算全部 token 的 KV 外,同时计算并保留 token 对的优先级排名信息(如 H2O 的 attention scores);vLLM 每次调度检查是否满 n 步,需要压缩时按优先级排名分配一个或多个新 KV block、修改输入位置信息,由 block manager 把相应 KV block 从原 sequence group 搬到新 block,随后递减原 block 引用计数、可能直接释放;循环往复直到生成结束;
- 改动面:`core/scheduler.py`(是否触发稀疏动作的判断)、`core/block_manager_v1.py`(block table 映射更新与块分配/释放)、`worker/model_runner.py`(稀疏化后位置更新并向模型传递 blocks_to_sparse_copy)、模型文件如 `models/opt.py`(指明哪些 KV 应被过滤)、`csrc/attention/attention_kernels.cu` 与 `csrc/cache_kernels.cu`(计算 attention score 选取"重要" token、支持 sparse_cache_copy);
- 配套产出:配套 PR #5752 与 Google Docs 设计文档。

## 优化点与收益
目标是降低 KV cache 显存占用、腾出空间支持更长上下文或更大 batch;收益框架化:同一套淘汰框架可承载 H2O、SnapKV、StreamingLLM 等多种算法。正文示例参数为"淘汰 20% 实现 80% 的 KV cache 使用量削减",为说明性配置而非实测结果,正文未给出量化数字。

## 现状与落地
state=open:已有配套实现 PR(#5752)与设计文档,处于征求意见阶段(作者 CC 了多位核心维护者,询问设计是否合理),等待评审。

## 关联
- https://github.com/vllm-project/vllm/pull/5752 (配套 PR)
- https://docs.google.com/document/d/13_cpb31P9VOmPGa_tZ70s7z1vXGP_UenXf1WVuIppCk/ (设计文档)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
