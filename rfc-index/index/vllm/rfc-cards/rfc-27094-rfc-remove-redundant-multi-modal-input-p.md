# RFC #27094: [RFC]: Remove redundant multi-modal input preprocessing during disaggregated inference

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27094](https://github.com/vllm-project/vllm/issues/27094) |
| 作者 | MerHS |
| 创建日期 | 2025-10-17 |
| 状态 | ⚪ closed(关闭于 2026-06-16) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议在 E/P/D 分解推理中以 image_meta 元数据取代在所有 worker 间重复传输并预处理的多模态输入,消除占 TTFT 近半的重复预处理。

## 提案动机
E/P/D 分解推理(#25233)中,多模态输入的预处理在推理路径的所有 worker 上重复执行。实测:单请求 1024x1024 图 + 单文本 token,Qwen/Qwen2.5-VL-3B-Instruct,3x RTX3090(E→P PCIe、P→D NVLink),vllm bench serve 10 请求端到端 Mean TTFT 1014ms,其中 Preprocess 143ms ×3,细分为 request parsing 14.5ms、base64→PIL 28ms、tokenize 4ms、多模态 hash 15ms、HF 多模态预处理(Qwen2.5-VL 主预处理)75ms 等。作者指出预处理器占了推理时间的一半(约 500ms);对照 HydraInfer(arxiv:2505.12658)为 Qwen2-VL-7B + TextVQA 设的 1 秒 TTFT SLO,当前 EPD 分解难以达标。

## 提出的方案
- 在多模态请求类型中新增 image_meta(受 PyTorch meta device 启发):含 mm_hash、pixel_values(type/shape/dtype)、image_grid_thw 等元数据;encoder 之后请求只携带元数据(如 Qwen2-VL 示例)
- encoder worker 的结果携带 image_meta(类似 PD 分解中的 kv_transfer_params),router 用其替换原请求中的图像内容,下游 P/D worker 不再重复预处理
- 即便 image_meta 在 PD worker 侧产生空 dummy 张量,仍需保留以计算多模态 token 长度或向后兼容
- 若实现绕过 scheduler/device worker 的 hash-based 多模态输入处理,mm_hash 可省去;但多模态 cache 不得把生成的 dummy 张量注册进缓存

## 优化点与收益
按实测,重复预处理占端到端 TTFT(1014ms)约一半(约 500ms,即 143ms ×3);消除后 E/P/D 各 worker 均省去该段 CPU 开销。正文未给出改动后的实测数字。

## 现状与落地
已关闭(2026-06-16),labels 含 stale;正文未说明是否实现,落地情况需查关联 PR。

## 关联
- #25233(E/P/D disaggregation)、#26376(Q4 2025 roadmap)、#23887、arxiv:2505.12658

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
