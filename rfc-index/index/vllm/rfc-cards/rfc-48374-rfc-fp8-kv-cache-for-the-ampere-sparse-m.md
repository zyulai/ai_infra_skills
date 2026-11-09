# RFC #48374: [RFC]: fp8 KV cache for the Ampere sparse-MLA path (TRITON_MLA_SPARSE) via software dequant

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48374](https://github.com/vllm-project/vllm/issues/48374) |
| 作者 | nickus |
| 创建日期 | 2026-07-11 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Quantization |

## 一句话结论
提议在 Ampere 的 TRITON_MLA_SPARSE 路径上以"fp8 作为存储 + 软件反量化"支持 fp8 KV cache,换取约 1.756× KV 容量,并坦承 decode 有代价。

## 提案动机
TRITON_MLA_SPARSE(#47629)只支持 bf16 KV,这在 24GB 显卡上是硬约束。官方排除 fp8 的理由是 Triton 无法在 Ampere 处理原生 fp8e4nv——这对"原生 fp8"成立,对"fp8 作为存储"不相关:#47629 自己加的 mqa_logits_triton.py 已在 SM80 上把 FP8 按 uint8 加载并用 LUT 解码;XPU 路径同样做在图反量化后复用 BF16 sparse MLA kernel。先例已在树内出现两次,且与 #43914/#47060 的"原生 FP8 需 SM89+"不冲突——软件反量化是不同机制。

## 提出的方案
- 在 _sparse_mla_compute_tile 内加 IS_FP8: tl.constexpr 分支(该函数是 _sparse_mla_kernel_final 与 _sparse_mla_kernel_split 加载 KV 的唯一通路),fp8 免费继承现有 split-KV、merge kernel、autotune 与 masking
- 标准 656 字节 fp8_ds_mla 页按 raw uint8 加载,以位数学解码,对全部 256 个字节值与 torch.float8_e4m3fn bit-exact;缓存布局不变,既有 C++ concat_and_cache_ds_mla writer 在 sm_86 原样可用
- patch 224 行(one flag / one branch / one dispatcher arg);backend dtype 声明、get_kv_cache_shape(576→656)、dtype canonicalize 等接线还需约 40 行(out-of-tree 已跑通);引擎级端到端启动与 logits 对比尚未做
- 备选方案均已考虑:原生 fp8e4nv 在 Ampere 不可行;整页反量化到 bf16 会付出整份 gathered KV 的物化成本,丢掉 24GB 卡上带宽节省的意义;维持 bf16 并缩小模型是现状(也是 GLM-5.2 级 DSA 模型在消费级 Ampere 长上下文不可用的原因)

## 优化点与收益
RTX 3090(sm_86)实测:KV pool 容量 1.756×(656 B vs 1152 B/token),计入 indexer 自身 K-cache 后约 1.63×;decode 对 fp32 参考 cosine 0.999996;decode 速度 vs bf16(同 kernel、topk=2048):bs=8 为 0.92×,bs=32 为 0.45×(即 bs=32 下慢约 2.2×)。作者明确这是容量换速度:24GB 卡上 1.76× 容量可能值得,A100 上不值得,故建议保持 opt-in;推测 autotune 配置表(BLOCK_N 仅 16/32)按 bf16 的 1152B 行调优,fp8 需要更大 block,但未实测,仅作假设标注。

## 现状与落地
🟢 open;本质是征求意见:项目是否要该特性、kernel-only PR 先行还是连同 backend 接线、对软件反量化方案是否有异议;作者愿在开 PR 前补引擎级 boot + logits parity 与配置表测量。

## 关联
- #47629(TRITON_MLA_SPARSE)、#43914、#47060
- 顺带发现并已单独立案的 bug:#48364 / #48366
- patch: https://github.com/nickus/vllm-fp8kv/blob/main/patches/triton_mla_sparse_fp8.patch

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
