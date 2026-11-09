# RFC #10942: [RFC]: Dynamic KV Cache compression based on  vLLM framework

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#10942](https://github.com/vllm-project/vllm/issues/10942) |
| 作者 | Zachary-ai-engineer |
| 创建日期 | 2024-12-06 |
| 状态 | ⚪ closed(关闭于 2025-05-10) |
| labels | RFC, stale |
| 热度 | 👍 7 · 💬 4 |
| 分类 | KV Cache / Memory |

## 一句话结论
基于 KV 稀疏化在不同层做动态 KV cache 压缩,以约 1% 精度损失换取 1.5x 推理加速。

## 提案动机
作者梳理近一年 KV sparsity 领域的学术论文(H2O、SnapKV、PyramidKV),把 KV 稀疏化应用到模型的不同层:通过剪枝策略消除得分较低的 KV pair,保留得分较高且位置更近的 KV pair。这一方法同时减少内存占用、计算与 I/O 开销,最终加速推理。正文同时给出 Novita AI 基于 vLLM(从 v0.6.2 fork)的 KV-Compress 实验结果作为支撑,并称 Novita AI 也支持张量并行(如 Llama3-70B 跑多卡),因故暂不开放源码,希望向社区贡献技术与想法。

## 提出的方案
- **压缩机制**:用不同层设置不同滑动窗口长度来控制 KV cache 压缩率;实验主要设置 1024、1280、1536 三种滑动窗口长度,并在不同的稀疏化层数(22/26/31 层)之间交叉测试。
- **改动范围**(正文列出的主要修改文件):
  - Flash attention:基于 Flash attention 做 sparse scoring,同时保证 kernel 性能损失小于 1%;
  - Paged attention 与 reshape_and_cache:基于 Paged attention 做 sparse scoring,并在 prefill 与 decode 阶段同步稀疏打分;
  - Block_manager 及其他内存管理与张量准备相关函数。
- **测试环境**:所有 KV-Compress 实验基于 fork 自 v0.6.2 的 vLLM 集成,CUDA graph mode,block size 16;RTX 4090 / Llama-3.1-8B-Instruct 实验使用默认 GPU memory utilization 0.9、`max-model-length` 32k。基线为 vLLM-0.6.2 与 Novita AI(Pyramid KV Cache compression based on vLLM framework)。

## 优化点与收益
(数字均照抄正文)
- **吞吐**:在输入/输出长度 5000/500(正文称实际应用中最常见配置)、TTFT 须小于 2s 的条件下进行批量对比测试,「yielded a 1.5x inference speedup for vLLM」(条件:上述 v0.6.2 fork / RTX 4090 / Llama-3.1-8B-Instruct 设置)。
- **MMLU Pro**:不同 KV 稀疏层与窗口长度表现不同,「taking the acceleration ratio into consideration, the overall accuracy can be guaranteed to be above 98%」;表数据如 31 层、窗口 1024 时 0.431(vLLM-Base 为 0.4517)。
- **LongBench**:滑动窗口 1024 时「accuracy loss was about 1.03%」(Avg:49.5 → 48.99)。
- vLLM-0.6.2 不支持:Chunked-prefill、Prefix caching、FlashInfer 等 non-FlashAttention 后端、Speculative Decoding。

## 现状与落地
已关闭(2025-05-10),labels 含 stale;正文为经验分享加提案(源码未开放),未写明是否进入 vLLM 主线,落地情况需查关联 PR。

## 关联
- [设计文档:How KV Sparsity Achieves 1.5x Acceleration for vLLM](https://blogs.novita.ai/how-kv-sparsity-achieves-1-5x-acceleration-for-vllm/)(正文附带的 Novita AI 博客)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
