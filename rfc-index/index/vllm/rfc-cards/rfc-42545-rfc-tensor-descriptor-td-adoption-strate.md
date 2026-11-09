# RFC #42545: [RFC]: Tensor descriptor (TD) adoption strategy for vLLM Triton kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42545](https://github.com/vllm-project/vllm/issues/42545) |
| 作者 | afierka-intel |
| 创建日期 | 2026-05-13 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 7 · 💬 25 |
| 分类 | Attention / Kernel |

## 一句话结论
提出 vLLM Triton 内核采用 Tensor Descriptor(TD)的统一策略:单一 env var opt-in、平台感知默认与毕业标准。

## 提案动机
vLLM 的 Triton 内核目前经裸指针算术读写数据(`tl.load(ptr + offset, mask=…)`);Triton 上游正转向结构化内存访问 API `tl.make_tensor_descriptor`(TD),`tl.make_block_ptr` 已被官方废弃(2026-03-06 起发运行时弃用警告,见 triton#9667)。TD 是通往架构加速内存路径的途径:Hopper 上的 TMA、Intel Xe 上的 2D block load。vLLM 主线共 245 个 `@triton.jit` 函数,其中约 50 个可在 Intel XPU 调用,是近期范围;策略原则上覆盖全部 245 个。试点已在 `triton_unified_attention.py`(PR #40327,TRITON_ATTN 后端)落地验证。

## 提出的方案
- 单一环境变量 `VLLM_TRITON_USE_TD`,三态语义:不设=auto(Intel XPU 默认开、其他平台关)、`1`=强制开、`0`=强制关;试点先以 `VLLM_TRITON_ATTN_USE_TD` 落地,后续统一改名;不设子系统子开关
- 内核级 constexpr 双门:`USE_TD`(K/V cache tile 读取)与 `USE_TD_QO`(Q 读取与输出写出);禁用路径编译期死代码消除(生成 IR 中无分支),零回归设计
- paged KV 采用混合方案:block table 查找仍用裸指针,块内用 TD 加载;前置条件:`BLOCK_SIZE % TILE_SIZE == 0`、Triton ≥ 3.4.0(XPU 侧 triton-xpu ≥ 3.6.0),不满足时回落裸指针路径
- 明确非目标:不做无条件全量转换(B200 证据不支持);每个转换 PR 与每次默认开启翻转都须满足毕业标准:数值等价、≥3 个模型上 E2E 吞吐不低于裸指针路径、autotune 编译耗时不超过基线 2×。近期目标:TRITON_ATTN 其余约 22 个内核、MLA 约 9 个、LoRA 约 16 个

## 优化点与收益
- Intel Battlemage(B60/B70,TRITON_ATTN 后端,`VLLM_TRITON_ATTN_USE_TD=1`,吞吐中位数 req/s、3 次运行):Qwen3-4B-Thinking-2507 random-decode 3.51→6.41(+83%)、random-prefill 3.50→6.41(+83%)、sharegpt 5.02→8.52(+70% R1;+38–39% R2/R3 中位数);Llama-3.1-8B-Instruct random-decode 2.83→4.38(+55%)、random-prefill 2.83→4.37(+55%)、sharegpt 5.42→7.48(+38%)
- B200:TD 当前落后裸指针 –33% 至 –60% E2E(归因 per-tile descriptor 重建开销与未为 TMA 调优的 autotune 配置,非 API 结构性问题);NVIDIA 平台默认关闭,用户无成本
- Qwen2-7B(GQA 7:1):TD=0 与 TD=1 吞吐一致(random-decode 2.28 req/s),无非 2 的幂 GQA 回归;Hopper 预期 lowering 到 TMA,正文未附基准
- 默认(flash-attn)后端同轮波动 ±1.6%,TD 效果与 Triton 后端隔离

## 现状与落地
open,反馈期两周。试点 PR #40327 已落地;正文列出 4 个待维护者决策的开放问题(双路径长期设计、env var 设计与改名、后续转换的证据门槛、Triton 版本下限)。Intel Battlemage 已满足全部毕业标准。注:pack 正文在 20k 字符处截断(截于开放问题第 4 条),本卡基于现有内容撰写。

## 关联
- 试点 PR #40327:https://github.com/vllm-project/vllm/pull/40327
- 关联 RFC #40458:https://github.com/vllm-project/vllm/issues/40458
- Triton 弃用 PR:https://github.com/triton-lang/triton/pull/9667

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
