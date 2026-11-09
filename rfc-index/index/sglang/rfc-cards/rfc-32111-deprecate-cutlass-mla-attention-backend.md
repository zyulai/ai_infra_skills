# RFC #32111: Deprecate CUTLASS MLA attention backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32111](https://github.com/sgl-project/sglang/issues/32111) |
| 作者 | b8zhong |
| 创建日期 | 2026-07-22 |
| 状态 | ⚪ closed (2026-09-10) |
| labels | RFC |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
提议废弃 --attention-backend cutlass_mla,改用 trtllm_mla 或 tokenspeed_mla 做 MLA 解码。

## 提案动机
降低 MLA attention backend 路径的复杂度,清理长期无维护的后端选项。

## 提出的方案
- 废弃 Cutlass MLA:它是 decode-only MLA kernel,只 gated 到 SM 10.0(B200),且已在 SM 10.3(GB300)上禁用。
- 可移除的理由:CutlassMLABackend 只 override forward_decode,prefill、extend、verify、draft-extend 全部落到其父类 FlashInferMLABackend;锁定 page_size=128;阻止 tree speculative decode;约 10 个月无功能性改动(唯一近期改动是在更新的 Blackwell 硬件上禁用它);没有任何 CI workflow 覆盖它。
- 替代:推荐 trtllm_mla 或 tokenspeed_mla 作为受支持的 MLA 解码后端,正文原话「Their perf. is much better」。
- 反馈期:3 周。

## 优化点与收益
正文未给出量化数字;仅定性称替代后端(trtllm_mla/tokenspeed_mla)性能好得多。收益形式为减少 MLA 后端路径复杂度。

## 现状与落地
已关闭(2026-09-10)。正文未写明废弃是否已执行、由哪个 PR 承接,落地情况需查关联 PR。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
