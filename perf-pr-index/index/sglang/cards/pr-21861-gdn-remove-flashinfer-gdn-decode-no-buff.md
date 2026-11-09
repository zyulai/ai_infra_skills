# PR #21861: [GDN] Remove FlashInfer GDN decode + no_buffer guard and default to FlashInfer on SM100+

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-04-08 |
| 分类 | Speculative Decoding (tier: major,classify.py 给出;实际改动为 GDN linear attention decode backend 调度) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,仅 server_args.py 1 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/21861 |

## 一句话结论

解除 FlashInfer GDN decode 与 no_buffer 调度的组合禁令,并在 SM100+ 默认启用 FlashInfer GDN decode,conc=1024 时 TPOT 降 4.5%、吞吐升 4.7%。

## 动机

GDN(gated delta net)线性注意力的 FlashInfer decode kernel 此前因 bf16 decode kernel 中负 padding 索引导致 OOB 访问、精度退化,被 `raise ValueError` 禁止与 `--mamba-scheduler-strategy no_buffer` 组合(issue #20791)。根因已在 FlashInfer v0.6.7 修复(flashinfer-ai/flashinfer#2810 的 padding index guard),且 PR #21422 已把 FlashInfer 升级到 v0.6.7,禁令失去依据,可以移除并进一步把 FlashInfer 设为 SM100+ 的默认 decode 后端。

## 优化点

纯调度层改动(单文件):(1) 删除 `_handle_mamba_radix_cache` 中拦截 `linear_attn_decode_backend == "flashinfer"` + `mamba_scheduler_strategy == "no_buffer"` 的 ValueError;(2) 在 `_handle_linear_attn_backend` 增加默认选路:用户未显式指定 decode backend、SM100+、`mamba_ssm_dtype == "bfloat16"`(FlashInfer GDN 在 SM100+ 要求 bf16 state)且未开投机算法时,自动将 `linear_attn_decode_backend` 置为 `flashinfer`;显式排除 MTP 投机解码场景,因为 FlashInfer GDN MTP verify 尚不支持 SM100+。SM90 的 float32 state 路径与原有约束保持不变。

## 关键改动

```text
python/sglang/srt/server_args.py: 唯一改动文件——删除 no_buffer+flashinfer 禁令;新增 SM100+ 默认选路逻辑。
```

默认选路核心:

```python
if (
    self.linear_attn_decode_backend is None
    and is_sm100_supported()
    and self.mamba_ssm_dtype == "bfloat16"
    and self.speculative_algorithm is None
):
    self.linear_attn_decode_backend = "flashinfer"
```

## 性能声明

sa-bench,ISL=1024、OSL=1024,NVFP4,4xGB200,`--linear-attn-decode-backend flashinfer` 对比默认 triton:TPOT 全并发改善,conc=2 为 5.68 -> 5.61 ms(-1.3%),conc=256 为 23.60 -> 23.04 ms(-2.4%),conc=1024 为 53.50 -> 51.08 ms(-4.5%);输出吞吐相应 +0.5% 至 +4.7%(conc=1024 +4.7%),并发越高收益越大。

## 验证与回归风险

精度:Qwen3.5-397B-A17B(4xGB200),GSM8K 1319 题 8-shot temp=0.6,NVFP4+no_buffer 0.977/0.977(conc 128/512)、NVFP4+extra_buffer 0.979/0.977、FP8+no_buffer 0.978/0.977、FP8+extra_buffer 0.979/0.979,与既有水平一致;MTP 场景被排除(FlashInfer GDN MTP verify 未支持 SM100+)。PR checklist 全部未勾选,未新增单测;性能对比为 flashinfer 后端开关的 A/B,而非改动前后(diff 本身只改默认值)。回归风险:(1) SM100+ 上未显式指定后端的 GDN 用户会静默切到 FlashInfer,依赖 FlashInfer >= v0.6.7 的修复(旧版本会复现 #20791 的 OOB 精度问题);(2) 默认路径排除投机解码,但其他 spec 组合未被显式验证;(3) 仅 GSM8K 验证,长上下文/多语言精度未覆盖。

## 关联

前序:PR #21422(FlashInfer 升级 v0.6.7)、issue/PR #20791(OOB 精度问题)、flashinfer-ai/flashinfer#2810(根因修复)。同区域 GDN 优化:PR #15631(Cute DSL GDN decode kernel)、PR #17094/19434/19775/21019(qwen3-next GDN 融合)。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
