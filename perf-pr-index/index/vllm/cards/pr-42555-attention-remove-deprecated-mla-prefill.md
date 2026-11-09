# PR #42555: [Attention] Remove deprecated MLA prefill arguments

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-05-14 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/42555 |

## 一句话结论

删除 MLA prefill 的三个 deprecated 布尔参数,统一收敛到 --attention-config.mla_prefill_backend 单一入口。

## 动机

#32623 重构后 MLA prefill 后端已由 --attention-config.mla_prefill_backend 指定,旧的 use_cudnn_prefill、use_trtllm_ragged_deepseek_prefill、disable_flashinfer_prefill 三个布尔参数按计划应在 0.22 移除(本 PR 计划在 0.21 cut 之后落地),留着会造成两套配置入口并存和迁移逻辑常驻。

## 优化点

AttentionConfig 删除三个 deprecated 字段及 __post_init__/_migrate_deprecated_mla_prefill_flags 迁移逻辑(cuDNN 后端此前已移除,报错分支一并删除);mla_prefill_backend 成为唯一入口(FLASH_ATTN/FLASHINFER/TRTLLM_RAGGED)。mla_attention.py 的 DCP 分支把 is_lse_base_on_e 固定为 True,去掉运行时 getattr(self, "_use_fi_prefill") 探测。bench 的 mla_runner._PREFILL_BACKEND_CONFIG 改为直接引用 MLAPrefillBackendEnum,删除按旧 flag 组合的配置方式。

## 关键改动

```text
vllm/config/attention.py: 删除三个 deprecated 字段与迁移逻辑
vllm/model_executor/layers/attention/mla_attention.py: DCP 路径 is_lse_base_on_e 固定 True
benchmarks/attention_benchmarks/mla_runner.py: prefill backend 配置改用 MLAPrefillBackendEnum
tests/engine/test_arg_utils.py、tests/v1/attention/test_mla_prefill_selector.py: 同步删改
```

```python
mla_prefill_backend: MLAPrefillBackendEnum | None = None
"""MLA prefill backend to use. If None, will be selected automatically.
Valid options: FLASH_ATTN (FA3/FA4), FLASHINFER, TRTLLM_RAGGED."""
```

## 性能声明

PR 未给出性能数字(Test Result 为 TBD);属配置面清理,不改变所选后端的计算路径。性能意义在于消除 deprecated flag 迁移开销与多入口误配的可能。

## 验证与回归风险

测试计划:CI 的 V1 Attention (H100) 与 V1 Attention (B200)。回归风险主要是用户可见的破坏性变更:继续传 use_trtllm_ragged_deepseek_prefill / disable_flashinfer_prefill / use_cudnn_prefill 的脚本与 JSON 配置会失效;mla_attention.py 中 is_lse_base_on_e 固定 True 改变了 FlashInfer prefill 下 DCP 的 LSE 语义(原逻辑为 _use_fi_prefill 时取 False),依赖旧行为的 DCP+FlashInfer prefill 组合需复测。

## 关联

前序:#32623(引入 --attention-config.mla_prefill_backend 的重构,本 PR 是其清理收尾)。同分类 Attention 卡:#20034(FlashInfer MLA ragged prefill)、#12528(MLA decode 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
