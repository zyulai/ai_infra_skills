# PR #37467: [HMA]Move hybrid blksize to update_block_size_for_backend to fix attn supported block size is not 16 issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37467](https://github.com/vllm-project/vllm/pull/37467) |
| 作者 | xuechendi |
| 合并日期 | 2026-03-30 |
| labels | intel-gpu, ready, v1 |
| 热度 | 👍 1 · 💬 36 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
hybrid 模型 block_size 对齐发生在 XPU 把 block_size 改成 64 之前,致 FA 与 SSM 页大小不整除报错;把对齐移入 update_block_size_for_backend。

## 问题现象
测试 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-bf16`(默认 block_size=64,TP4):FA block_size=16 时通过(SSM page 540672 / FA page 8192 = 66);FA block_size=64 时报错 `SSM_page_size will not be evenly divided by FA_page_size: 540672/32768 = 16.5`(SSM page 540672 不能被 FA page 32768 整除)。修复后 hybrid 模型在 XPU 上日志显示 "Update hybrid model block size to 1088",full-attn 模型显示 "Update FLASH_ATTN block size to 64",lm_eval gsm8k 两种 case 均通过。

## 根因
正文明确披露:默认 hybrid 模型的 block_size 对齐发生在 `platform.check_and_update_config` 之前,配置初始化顺序为 `platform init` => `cache_config init` => `model_config init`(hybrid 模型 block_size 对齐)=> `__post_init` => `platform.check_and_update_config`;由于该顺序,hybrid 对齐只按 block_size=16 计算,而 XPU 会在 `check_and_update_config` 中把 block_size 更新为 64,造成后续页大小不对齐。经与 reviewer 讨论,方案是把 block_size 更新移到 `platform.update_block_size_for_backend`。

## 修复方案
- `vllm/platforms/interface.py`:`update_block_size_for_backend` 重写为两阶段——先按非 SSM backend 的 `get_preferred_block_size` 选 block_size,再对 hybrid 模型调用新增的 `_align_hybrid_block_size`(把原先写在 `HybridAttentionMambaModelConfig.verify_and_update_config` 里约 130 行的对齐/padding 逻辑整体迁入,对齐尺寸改为从 backend 的 `get_supported_kernel_block_sizes()` 推导);新增 `_find_non_ssm_backend`;
- `vllm/v1/attention/backend.py`:`AttentionImplBase` 新增 `is_ssm()` 默认 False;GDN/Linear/Mamba1/Mamba2/ShortConv 各 backend 覆写返回 True;
- `vllm/v1/attention/backends/flash_attn.py`:XPU 上 `get_preferred_block_size` 返回 `max(default, 64)`;
- `vllm/platforms/xpu.py`:删除 `check_and_update_config` 中 block_size=64 的旧逻辑与 `update_block_size_for_backend` 的占位 pass;
- `vllm/config/cache.py` 新增 `user_specified_mamba_block_size`;测试在构造 GPUModelRunner 后补调 `update_block_size_for_backend`。

## 影响与回归风险
- 回归面:hybrid attention/mamba(HMA)模型的 KV cache 分组与 block size 计算(对齐时机从 config 阶段后移到层构建、backend 确定之后,可能覆盖用户显式 block_size);XPU 平台的 block_size 选择路径;各平台 `update_block_size_for_backend` 行为;模型 config verify 阶段不再做对齐。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
