# RFC #59016: [RFC]: Software-dequant fp8 KV cache for MLA on Ampere (sm80/sm86) — consolidate the existing pieces, a 1M-ctx field case, and a validation offer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59016](https://github.com/vllm-project/vllm/issues/59016) |
| 作者 | ming616 |
| 创建日期 | 2026-09-28 |
| 状态 | 🟢 open |
| labels | kimi |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Quantization |

## 一句话结论
汇总 sm80/sm86 上 MLA 软件反量化 fp8 KV cache 的五块在途工作,请求维护者裁决合并路径,并承诺以生产机队做验证。

## 提案动机
MLA 模型(DeepSeek V4.x、Kimi-K3、GLM-5.3)在 Ampere 上目前只能 bf16-KV:`TritonMLAImpl` 对 SM89 以下拒绝任何 `--kv-cache-dtype fp8*`,尽管 `fp8_ds_mla` 布局是 storage-only(查询保持 bf16、页在读取时反量化,无需原生 fp8 运算)。代价是每 token 每层 1152 B 而非 656 B,即 1.76× KV 容量被搁置。三条相关线程各自停滞,本文请求维护者给出合并路径并 volunteer 生产验证站点。

## 提出的方案
- 已有未合并部件:#47629(sm80 TRITON_MLA_SPARSE)、#48374(sparse-path fp8 KV RFC,sm_86 端到端验证)、#52202(dense TritonMLA decode 草案)、#55173(可移植 Triton fp8e4nv 转换,A100 验证)、#55184/#55177(DSV4 pre-sm90 omnibus,A100 验证)。
- 提议顺序:先落 #55173 作公共基础;#47629 落地后基于 #48374 的已验证工作收 sparse-path fp8 KV PR;dense 评估 #52202 草案。
- 全部 gated 于显式 `--kv-cache-dtype fp8_e4m3`/`fp8_ds_mla`,bf16 保持默认;warmup/autotune 须按 cache dtype 区分。

## 优化点与收益
生产案例(GLM-5.3-Flash,4× A800 sm80,max-model-len 1M,需 fork 供服):bf16 基线(当周验证)930K 上下文 needle recall 5/5、decode 70 tok/s at 930K context、KV/token 11.6 KiB;KV 预算 24.7 GiB/卡 → 2.25M tokens → 最多 2 路满长 1M 并发(2.15×),fp8 KV 下同预算约 3.7×,4× 满长在现有余量内可达。

## 现状与落地
open,等待维护者回答三个问题(原则上是否接受、部件排序、#52202 形态);作者提供常设验证服务:teacher-forced parity、930K–1M needle recall、吞吐/并发扫描、跨会话状态污染 canary,并承诺无论结果公开数字。

## 关联
#47629、#48374、#52202、#55173、#55184/#55177

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
