# PR #46994: [Spec][V2] Support MTP speculative decoding under pipeline parallelism

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @eastwood-c |
| 合并 | 2026-09-11 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, intel-gpu, speculative-decoding, ready, needs-rebase, v1, qwen, deepseek, nvidia, mrv2, dflash |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 "file(s) dropped from diff" 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/46994 |

## 一句话结论

补齐 V2 model runner 上 PP>1 的 MTP 投机解码，GLM-5.2-FP8 (PP2/TP4) K=1 acceptance 93.51%。

## 动机

MTP speculative decoding 在 V2 model runner 的 pipeline parallelism 下不工作。#50514 已落地通用 PP spec-decode transport，本 PR 覆盖 PP>1 路径上剩余的 MTP-specific gaps。**Depends on #55745**——`#50514` 的 `broadcast_drafts` 里一行 `record_stream` 守卫，缺了会在异步 launch 下触发 device-side `IndexKernel` assert；它是本分支底 commit，#55745 合并后可丢弃。

## 优化点

正文列出六处缺口：(1) MTP drafters 未实现 `SupportsPP`，引擎拒绝构建 draft（`NotImplementedError` ... `[DeepSeekMTPModel]`）；draft 只在最后 PP stage 运行，但接口仍要求 `make_empty_intermediate_tensors` 工厂，应用到 `DeepSeekMTP`、`Qwen3_5MTP`（镜像 V1 runner 的 #39704）。(2) Qwen3.5/3.6 MTP draft head 按 `is_first_rank` 分支，在 drafter 所在的最后 stage 为 False 而跳过 `fc` 投影，落到无内容可读的 intermediate-tensor 路径（约 1% acceptance），改为按 `intermediate_tensors is None` 分支。(3) DeepSeek MTP 不加载自身 `embed_tokens`，用未初始化权重 embed；PP 下 target 副本在 draft stage 是 `PPMissingLayer` 无法借用。(4) `DeepseekV32MTP` fork 后既无 `SupportsPP` 也无 `embed_tokens` 加载，涉及 GLM-5.2 与 DeepSeek-V3.2 路径；缺失加载是静默的，只有 acceptance 能看出：32.69% 对加载后的 93.51%，nvidia/amd 变体均补上。(5) sparse MLA 的 stale `topk_indices_buffer`：proposer 在 backend impl 快照 buffer 后才把 indexer 重指向 target，impl 读到无人写入的 buffer；新增 `SharedTopkIndicesBuffer` mixin 按读取时解析，覆盖所有 `SparseMLACommonImpl` 子类及三个非子类，mixin 放 `sparse_mla_attention.py`。(6) 无 encoder runner 的 rank 上 draft MM embedding gather 在 warmup 挂掉（`AttributeError: 'MambaHybridModelState' object has no attribute 'encoder_runner'`），`Qwen3_5MTP` 是首个同时 `SupportsPP` 与 `SupportsMultiModal` 的 in-tree draft。

## 关键改动

```text
vllm/model_executor/layers/attention/sparse_mla_attention.py: 新增 SharedTopkIndicesBuffer mixin，topk_indices_buffer 惰性解析
vllm/model_executor/models/deepseek_mtp.py: DeepSeekMTP 实现 SupportsPP 并加载自身 embed_tokens
vllm/model_executor/models/qwen3_5_mtp.py: 实现 SupportsPP、修正 draft head 的 fc 分支、MM embedding gather
vllm/models/deepseek_v32/nvidia/mtp.py 与 vllm/models/deepseek_v32/amd/mtp.py: DeepseekV32MTP 恢复 SupportsPP 与 embed_tokens 加载
tests/v1/attention/test_sparse_mla_backends.py: fix 5 的回归单测（repointed buffer、显式赋值优先、子类家族）
```

关键代码（来自 `tests/v1/attention/test_sparse_mla_backends.py` diff）：

```python
def test_sparse_impl_observes_repointed_indexer_buffer():
    impl = object.__new__(FlashInferMLASparseImpl)
    own = torch.zeros(4, 8, dtype=torch.int32)
    target = torch.ones(4, 8, dtype=torch.int32)
    indexer = SimpleNamespace(topk_indices_buffer=own)
    impl.init_topk_indices_buffer(indexer, None)
    assert impl.topk_indices_buffer is own
    indexer.topk_indices_buffer = target
    assert impl.topk_indices_buffer is target
```

## 性能声明

测试条件：1x H200 NVL (SM 9.0)、GSM8K 前 200 prompts、greedy、`max_tokens=512`、`VLLM_USE_V2_MODEL_RUNNER=1`、async launches，指标 `mean_acceptance_length`。GLM-5.2-FP8 (PP2/TP4)：K=1 `1.9359`/token accept `93.51%`，K=2 `2.7505`/`87.39%`，K=3 `3.4313`/`80.74%`。QuantTrio/Qwen3.6-35B-A3B-AWQ (PP2/TP1)：K=1 `1.9459`/`94.56%`，K=2 `2.7995`/`89.86%`，K=3 `3.5474`/`84.67%`。Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 (PP2/TP1)：K=1 `1.9535`/`95.32%`，K=2 `2.8156`/`90.66%`，K=3 `3.5854`/`85.97%`。`max_tokens=2048` 时 Qwen3.6-AWQ k=1：PP=1 `1.9375`，PP=2 四次 `1.9380`/`1.9373`/`1.9354`/`1.9357`。另给出 fix 4 的 acceptance `32.69%` → `93.51%`。早期 revision 在 GLM-5.2-FP8 (TP4/PP2, K=3) 跑 5+ 小时：`91,088 drafts`、`84.4% overall acceptance`（rebase 前，属先前佐证而非当前树验证）。PR 正文未给出吞吐/延迟类加速数字。

## 验证与回归风险

1x H200 NVL (SM 9.0) 上：8 个 attention 测试文件 → `186 passed, 616 skipped`；`test_pp_utils.py` → `4 passed`；`test_mtp_parallel_load.py` → `5 passed, 1 skipped`；`test_qwen3_5_mtp_config.py` 等 → `57 passed, 1 skipped`；`pre-commit run --files <12 changed>` 与 `mypy-3.12 --hook-stage manual` 通过。Skips 为能力/dtype 门控（SM 10.x、`fp8_ds_mla`/`nvfp4_ds_mla`、FA4、`block_size=32`），与本改动无关。非 v32 `DeepSeekMTP` 路径只有随机权重 fixtures（acceptance 无意义），在 `pipeline_parallel_size=2`、K=0/1/2 下能构建并生成。可能受影响路径：V2 model runner 的 PP>1 投机解码、Qwen3.5/3.6 MTP、DeepSeek-V3.2/GLM-5.2 的 sparse MLA、多模态 MTP draft、ROCm/XPU sparse MLA 后端。

## 关联

#50514（通用 PP spec-decode transport）、#55745（record_stream 守卫依赖）、#39704（V1 runner 的 SupportsPP 镜像）、#53408/#54635/#55081（其他 MTP 家族 SupportsPP，#55081 与 fix 4 的 SupportsPP 半部重叠）、#54709（qwen4_exp draft head 同一 bug）。均为正文明确出现的编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
