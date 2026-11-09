# PR #53183: [Model Runner V2] Use MRV2 for all models by default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2026-08-27 |
| 分类 | Other |
| Label | ready, torch.compile, nvidia, mrv2 |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/53183 |

## 一句话结论

把 Model Runner V2 从架构白名单改为全模型默认(ROCm DeepSeek 除外),V1 专属特性改为显式报错。

## 动机

此前 `use_v2_model_runner` 依赖 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` 白名单(DeepseekV2/V32/V4、GlmMoeDsa、KimiK3、Qwen2Moe 等)加 hybrid/attention-free/MoE 推断来决定,大部分模型仍默认走 MRV1;白名单与 PCP、dspark、dflash2、diffusion 等强制 V2 的特例逻辑纠缠在 `use_v2_model_runner` property 里,分支多且不对称(V2 有校验、V1 无校验)。本 PR 将默认翻转为"全模型 MRV2",仅对 ROCm 上 MRV2 不支持或更慢的模型保留 MRV1 例外。

## 优化点

(1) 删除 `default_v2_model_runner_architectures` 白名单及 lru_cache,`use_v2_model_runner` property 改为:env 开关优先;ROCm 且架构命中 `ROCM_DEFAULT_MRV1_ARCHITECTURES`(DeepseekV32/DeepseekV4)时回退 V1 并 warning;无 HAS_TRITON 时 warning 回退。原来"PCP>1/dspark/dflash multi-kv-group/dflash2/diffusion 强制 V2"的分支删除——因为默认已是 V2,不再需要强制。(2) 新增 `_get_v1_model_runner_unsupported_features`/`_validate_v1_model_runner`:显式 `VLLM_USE_V2_MODEL_RUNNER=0` 时若使用 PCP、dspark、mixed sliding/full dflash drafts、dflash2、diffusion、batch-sharded sampling 则直接 ValueError,不再静默降级。(3) batch-sharded sampling 校验不再把"需要 V2"作为 blocker。(4) `vllm/compilation/decorators.py` 对经 forward context 进入计算图的 tensor(如 MoE padding mask)补 `torch._dynamo.mark_dynamic(is_padding, 0)`,避免 AOT 编译产物固化静态 batch guard。(5) `make_local_attention_virtual_batches` 中 `seq_lens_cpu.numpy()` 包进 `gpu_sync_allowed()`(关联 #31852)。

## 关键改动

```text
vllm/config/vllm.py: 删白名单,新增 ROCM_DEFAULT_MRV1_ARCHITECTURES;use_v2_model_runner 简化;V1 特性校验
vllm/compilation/decorators.py: forward context 的 is_padding tensor mark_dynamic
vllm/v1/attention/backends/utils.py: seq_lens_cpu.numpy() 纳入 gpu_sync_allowed()
tests/test_config.py: 测试改为验证全模型默认 V2 + ROCm carve-out
```

```python
if model_config is not None and current_platform.is_rocm():
    architectures = getattr(model_config, "architectures", ())
    if any(arch in ROCM_DEFAULT_MRV1_ARCHITECTURES for arch in architectures):
        logger.warning_once("Defaulting to V1 model runner on ROCm ...")
        return False
```

## 性能声明

PR 未注明测试条件,正文无 benchmark 数字;定位为默认行为翻转(全模型启用 MRV2 执行路径),性能收益依赖 MRV2 本身的能力(作者称 ROCm 上部分模型 MRV2 "slower or unsupported",故留 carve-out)。

## 验证与回归风险

`tests/test_config.py` 大幅改写:`test_models_default_to_v2_model_runner` 参数化验证 Qwen3/Qwen2/DeepSeek-V3/hybrid/attention-free 等默认 V2(此前期望 False 的行翻转为 True);`test_rocm_keeps_compiled_deepseek_defaults` 验证 ROCm 上 DeepseekV32/V4 走 compiled MRV1 且不进 breakable cudagraph;新增 `test_v1_model_runner_rejects_v2_only_features`(PCP 触发 ValueError);`test_fused_shared_expert` 补 MTP 配置字段。风险面:所有模型的默认 runner 改变,V1 专属未列特性可能遇到新的 ValueError 而非静默回退;依赖 `VLLM_USE_V2_MODEL_RUNNER` env 的用户行为变化;AOT 编译 guard 与 local-attention 虚拟批处理的同步行为变化影响 compile/attention 路径。

## 关联

同区域(MRV2/breakable cudagraph):索引内 PR #52228(MRV2 spec decode adaptive verification)、PR #62784(MRV2 跳过无请求可用步骤的 drafting)、PR #52162(PCP shard decode,依赖 MRV2)、PR #51012(Qwen3.5 breakable cudagraph 迁移,同改 `vllm/config/vllm.py` 的 breakable 架构集合)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
