# PR #40860: [Feat] DeepSeek V4 Rebased

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ivanium（Co-authored by Bugen Zhao, Giancarlo Delfin, Jie Li, Kaichao You, Roy Wang, Woosuk Kwon, Yifan Qiao, Yongye Zhu, Zhewen Li, Zijing Liu, Zixi Qi） |
| 合并 | 2026-04-27 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | documentation, new-model, frontend, speculative-decoding, ci/build, v1, tool-calling, deepseek, cpu, gpt-oss, kv-connector, nvidia, ready-run-all-tests, verified |
| Diff | diffstat 缺失（素材记录为 +None/-None across None files）；diff 共 145+ 文件，144 个源码文件因超长被剔除，仅保留部分 |
| 链接 | https://github.com/vllm-project/vllm/pull/40860 |

## 一句话结论

为 DeepSeek V4 增加 vLLM 支持：引入 MLAP 专用的 QNorm+RoPE+KV 量化插入水平融合 CUDA kernel 与 softplus-sqrt MoE 门控 kernel，减少逐 token 的 kernel 启动与访存开销。

## 动机

PR 正文极简：本卡是 #40760 的 rebase 版本，为 vLLM 接入 DeepSeek V4（roadmap 见 issue #40902），无独立动机描述。从 diff 可推断动机：DeepSeek V4 的 MLA 注意力路径原本需要多个独立 kernel（Q 侧 RMSNorm、GPT-J RoPE、KV 侧 RoPE、UE8M0 FP8 量化、paged cache 插入），逐 token 开销大，故做水平融合。

## 优化点

核心是新增 `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`：单个 CUDA kernel、单一 grid 内按 head-slot 分派 Q 与 KV 两类工作——Q 分支做无权重的 per-head RMSNorm + GPT-J 交错 RoPE（全程 fp32 数学，仅最终落盘转 bf16）；KV 分支做 RoPE 后对 NoPE 部分做 UE8M0 FP8 逐块（64 元素）量化并直接写入 paged cache（每 token 448 fp8 + 128 bf16 + 8 字节 scale）。kernel 硬编码 DeepseekV4 注意力形状（HEAD_DIM=512/ROPE_DIM=64/NOPE_DIM=448），使用 128-bit 向量化 load/store，SM90+ 上启用 PDL（programmatic dependent launch，`cudaGridDependencySynchronize`/`cudaTriggerProgrammaticLaunchCompletion`）重叠上游 kernel。另新增 `topk_softplus_sqrt_kernels.cu`：把 sigmoid、max、argmax 融合进单个 TopK gating kernel，并支持 hash MoE 查表路径（tid2eid）。

## 关键改动

```text
csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu: 新增，MLAP Q/KV 融合 kernel（RMSNorm+RoPE+UE8M0 FP8 量化+cache 插入，PDL）
csrc/moe/topk_softplus_sqrt_kernels.cu: 新增，softplus-sqrt TopK MoE 门控融合 kernel（支持 bf16/fp16/fp32 输入与 hash MoE）
csrc/layernorm_kernels.cu: RMSNorm/fused_add_rms_norm 改为 fp32 中间计算（x*s_variance*w）消除中间 bf16 舍入
csrc/layernorm_quant_kernels.cu: fused add rms norm + static FP8 quant 对齐非融合路径的 scalar_t 舍入精度
csrc/cpu/pos_encoding.cpp: CPU rotary_embedding 增加 rope_dim_offset/inverse 参数（CPU 上不支持并报错）
```

## 性能声明

PR 正文与 Test Plan/Test Result 均为空，无任何 benchmark 数字。素材正则未检出性能声明。该 PR 的性能收益只能从融合 kernel 结构推断，PR 未注明测试条件。

## 验证与回归风险

label 含 ready-run-all-tests，但正文未给出具体测试命令或结果。风险面广：layernorm_kernels.cu 与 layernorm_quant_kernels.cu 是共享路径，fp32 中间计算改动影响所有使用 RMSNorm/fused-add-RMSNorm + static FP8 量化的模型（diff 注释称此改动是为对齐非融合复合路径在 E4M3 量化 tie 边界的精度，即更保守而非更激进）；CPU rotary_embedding 签名变更涉及 CPU 路径；融合 kernel 要求 sm_80+（pre-Ampere bf16 路径编译为 no-op 并在 host 侧显式报错）。144 个文件被剔除，无法核实其余改动（模型实现、前端、spec decode、kv-connector 等）。

## 关联

前序：#40760（本卡为其 rebase 版本）；roadmap issue：#40902。同为 DeepSeek V3.2/V4 相关路径的优化可参考分类 Model-specific Fast Path 下其他 PR（素材未列出具体编号，无法进一步确认）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
