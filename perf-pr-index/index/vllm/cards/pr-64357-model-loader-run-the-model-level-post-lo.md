# PR #64357: [Model Loader] Run the model-level post-load hook once at the end of every load and reload

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hao-aaron |
| 合并 | 2026-10-30 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | speculative-decoding, ready, ci/build, llama, qwen, deepseek, DSv4, dflash, kimi, k3, DSv4.1, beam |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「30 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/64357 |

## 一句话结论

让 model-level post-load hook 在每次 load/reload 末尾只跑一次，NCCL 权重更新 3440 tensors 0 mismatch。

## 动机

RL weight updates 不会用一次 `load_weights` 加载模型：NCCL、IPC、M2N、sharded RDT 等引擎与 `reload_weights` 会在 `initialize_layerwise_reload` / `finalize_layerwise_reload` 之间按 bucket 或按 tensor 多次调用。main 上由此出三类问题：`finalize_layerwise_reload` 只收尾 layers，模型 `process_weights_after_loading()` 只在冷加载路径跑，reload 后派生状态（fused buffers、broadcast copies）保留旧值；部分模型把该工作放进 `load_weights`，于是每 bucket 跑一次、在 meta 层上失败（`Cannot copy out of meta tensor`，影响 DeepSeek-V4/V4.1、DFlash/DSpark drafters、Muse Spark）；"what this call loaded" 检查每 bucket 都触发，误报 "MTP layer N weights missing" 或把 DSpark `confidence_head` 置 None。

## 优化点

新增 `finalize_model_weights(model)` 统一调用模型 `process_weights_after_loading()`：冷加载末尾（per-layer quant processing 之后）与每次 `finalize_layerwise_reload` 末尾各一次，覆盖 `reload_weights` 及 NCCL/IPC/M2N/sharded-RDT 引擎。sparse NCCL 引擎在 `finish_weight_update` 调用一次；torchao reload decorator 与 expert-substitution 错误路径改用 `finalize_layerwise_processing` 以免提前跑 hook；全 direct 的 M2N plan 仍跳过 hook（同 main）。把 post-load 工作从 `load_weights` 移入 hook（mHC broadcast、Engram page-cache drop/THP collapse、`finalize_mega_moe_weights`、`_build_fused_kv_buffers`、`materialize_residual_path_constants` 等）。"Complete load" 检查由 `is_mtp_completeness_check_enabled()` 门控；dots3 vision 的 `compile_block_modules` 改为 `Module.compile` 原地编译且只一次；Llama4 DP vision 改用 `QKVParallelLinear`/`RowParallelLinear`（`disable_tp=True`）并移除 `_consolidate_qkv_weights`；raw Tensorizer 加载改走 `BaseModelLoader.load_model`。

## 关键改动

```text
vllm/model_executor/model_loader/weight_utils.py: 新增 finalize_model_weights(model)，集中调用模型 process_weights_after_loading()。
vllm/model_executor/model_loader/utils.py: process_weights_after_loading 末尾由直接调用改为 finalize_model_weights。
vllm/model_executor/model_loader/reload/layerwise.py: finalize_layerwise_reload 在 finalize_layerwise_processing 后追加 finalize_model_weights；torchao_decorator 改用 finalize_layerwise_processing。
vllm/distributed/weight_transfer/sparse_nccl_engine.py: finish_weight_update 由 no-op 改为调用 finalize_model_weights 一次。
vllm/model_executor/model_loader/tensorizer_loader.py: 删除 _load_model_serialized_cpu，raw 加载改走 super().load_model；tests/model_executor/model_loader/test_reload.py 新增 TestModelHookOnSplitLoad。
```

```text
def finalize_model_weights(model: torch.nn.Module) -> None:
    """Run once after a full load or reload: the model-level post-load hook."""
    if hasattr(model, "process_weights_after_loading"):
        model.process_weights_after_loading()


def finalize_layerwise_reload(model, model_config):
    finalize_layerwise_processing(model, model_config)
    finalize_model_weights(model)
```

## 性能声明

PR 正文以 reload 正确性 / A/B 为主，未主张吞吐加速。DeepSeek-V4-Flash-DSpark 冷启动（GB200，TP=4，7 speculative tokens，GSM8K 1319 questions、5-shot、greedy）：GSM8K main 0.9477 / this PR 0.9484，draft acceptance 均 0.4938。NCCL weight update（server TP=2，trainer `fsdp=2`，layerwise reload，2 syncs 全 tensor 变化的 checkpoint）：main 报 `Cannot copy out of meta tensor` 失败，this PR 3440 tensors compared、0 mismatched、no memory growth。Llama-4-Scout BF16（GB200，TP=4，`mm_encoder_tp_mode="data"`，prefix caching off）：main 的 `_consolidate_qkv_weights` torch.cat 失败，this PR 1047/1047 tensors byte-identical 且与冷启动 greedy 图文样本一致。GB300/H200/MI355X 冷启动 A/B 的 GSM8K、image accuracy、DFlash acceptance length、throughput、KV cache size 在 run-to-run noise 内一致。正文明确列出 reload 后 draft acceptance 下降的已知回归：Beam DFlash2 TP4 split reload 7.03 → 4.89；MiniMax-M3.1 MXFP8 DSpark TP4 5.70 → 4.07。

## 验证与回归风险

验证：`TestModelHookOnSplitLoad`（两 bucket / partial / sparse NCCL）、`test_qwen3_dflash_mxfp8_context_kv.py`、`test_expert_substitution.py`、`test_deepseek_v4_vl.py`、`test_tensorizer.py`；GB200（slurm 41251）及 GB300/H200/MI355X 冷启动 A/B。风险（正文 "Not in this PR" 列出）：DeepSeek-V4 ROCm reload 现在 fail（`fuse_shared_expert` 下二次运行 `AttributeError`）；dots3 vision MoE、Muse Spark NVIDIA reload 仍 stale；Kimi-K3 AMD / Muse Spark 因 hook 重建新 tensor，CUDA graph 读旧 buffer 致 draft acceptance 下降；未覆盖 DeepSeek-V4.1 reload、drafter weights during update、Muse Spark、dots3 reload、XPU 与 ROCm reload。

## 关联

RFC #54477（direct weight reload stack，本 PR 是 PR 1）、#57733（本 PR 替代它并补回其收窄时丢的 Llama4 DP-vision fix）、#58382（direct reload，负责保持 derived tensors 地址）；正文另以序号提及 PR 5 `refill_derived` 与 PR 6（stack 后续，未给编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
