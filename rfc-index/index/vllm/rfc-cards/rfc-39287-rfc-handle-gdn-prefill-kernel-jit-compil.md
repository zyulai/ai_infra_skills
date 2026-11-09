# RFC #39287: [RFC]: Handle GDN prefill kernel JIT compilation failures - seeking community input

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39287](https://github.com/vllm-project/vllm/issues/39287) |
| 作者 | Alex-ai-future |
| 创建日期 | 2026-04-08 |
| 状态 | ⚪ closed(关闭于 2026-08-10) |
| labels | RFC, stale |
| 热度 | 👍 4 · 💬 2 |
| 分类 | CI / Infra / Process |

## 一句话结论
解决 FlashInfer GDN kernel JIT 编译在 H100 上吃掉 150-200GB 内存导致 OOM/挂起的问题,增加预警与回退。

## 提案动机
**症状**:H100 用户加载 Qwen3.5/Qwen3-Next 模型时,在 FlashInfer GDN kernel JIT 编译期间出现 OOM 或挂起,内存消耗达 150-200GB。**根因**:默认编译使用全部 CPU 核心(64-128 核),每任务约 3GB;按公式「编译内存 = MAX_JOBS × ~3GB/任务」,64 核 × 3GB = 192GB、128 核 × 3GB = 384GB,叠加模型(72B 约 140GB)、KV cache(20-50GB)与系统开销(10-20GB),总需求约 362-402GB。**影响面**:约 7% 的 H100 用户(Qwen3.5 + 无预编译 cubin)。只有 Compute Capability 9.0+ 的卡(H100/H800、B200、RTX 5090)会触发 FlashInfer GDN;A100(8.0)及更早架构自动回退 Triton/FLA,不受影响。PyTorch JIT 为编译速度默认最大并行,历史背景是 CUDA 扩展编译的标准行为,多数用户因不用 GDN 模型、有预编译 cubin或架构较旧而未察觉。

## 提出的方案
采用「Approach A + 增强」,在 `vllm/model_executor/layers/mamba/gdn_linear_attn.py`(约 +100 行)实现四项:
- **内存预警**:编译前展示估算内存需求(如编译 ~192GB、模型 ~70-140GB、KV Cache ~10-50GB、合计 ~292GB,并附系统总/可用 RAM 与 CPU 核数),提示可 `export MAX_JOBS=1/8`、`pip install flashinfer-cubin` 或 `--gdn-prefill-backend triton`。
- **错误处理**:对 FlashInfer 导入与 kernel 执行包 try-except,给出带修复建议的详细报错(如找不到 `-lcuda` 时建议 Triton 后端、检查 `LD_LIBRARY_PATH`、限制 `MAX_JOBS`/`NVCC_THREADS`)。
- **自动回退**:FlashInfer 不可用时回退 Triton/FLA,避免硬失败。
- **预检查**:初始化时检测 FlashInfer 可用性,尽早失败。
改动集中在 `fi_chunk_gated_delta_rule()` 与 `ChunkGatedDeltaRule.__init__()`。遗留可选项:文档加 Known Issues、诊断脚本 `test_flashinfer_jit.py`、默认 `MAX_JOBS=2`(已讨论、暂缓)。

## 优化点与收益
收益为可用性与用户体验:修复前,用户启动即 64 并发编译、内存爬至 192GB,被 OOM killer 终止或挂起 20 分钟且无清晰报错;修复后,启动前预警(如「需 ~292GB,仅有 256GB」),用户可自主选择 `export MAX_JOBS=2`(编译内存降至约 6GB)、安装 flashinfer-cubin(免编译)或 `--gdn-prefill-backend triton`,实现成功启动。正文未给出性能类量化数字(上述均为内存/耗时描述)。

## 现状与落地
已关闭(2026-08-10)。正文写明「Solution Implemented」,即修复方案已实现(文件改动表已列出);但正文 Related/TL;DR 中 PR 号均为占位符「#XXXXX」,无法据此定位实际 PR,落地情况需查关联 PR。用户的即时规避手段:`export MAX_JOBS=2` 或 `--gdn-prefill-backend triton`。

## 关联
- 正文未列出有效链接(正文 Related 一节的 PR 为占位符 #XXXXX;受影响模型为 Qwen3.5 / Qwen3-Next 系列,平台为 H100、H800、B200、RTX 5090 等 CC 9.0+ 设备)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
