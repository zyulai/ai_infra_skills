# PR #65901: [Perf] Add opt-in FlashInfer multimodal encoder autotuning

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jzakrzew |
| 合并 | 2026-11-16 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, ready, ci/build, multi-modality, nvidia |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65901 |

## 一句话结论

为多模态 encoder 与 projector 增加 opt-in 的 FlashInfer autotuning(现有 language-model warmup 会跳过它们):`--kernel-config '{"enable_flashinfer_mm_encoder_autotune": true}'` 后,Nemotron 3.5 Embed NVFP4 在 448x448 图像上离线 embedding 吞吐提升 6.2-7.7%(RTX PRO 6000),ViDoRe finance_en 图像负载约 +2.6%;默认关闭,因为额外的 profiling 会拉长启动时间(16.1 s → 29.2 s)。

## 动机

FlashInfer autotune 现有流程只覆盖语言模型;vision encoder/projector 的 GEMM tactics 不被调优。本 PR 让 encoder profiling 用多模态 profiling inputs 与 FlashInfer 的 per-operation token buckets 完成 tuning。需要 FlashInfer autotuning 与 Model Runner V2;`--skip-mm-profiling` 时跳过;pass 在同步 warmup 之后运行,tuning process group 保持 weights TP mode。

## 优化点

- 新增 `KernelConfig.enable_flashinfer_mm_encoder_autotune`(默认 False,进 `compile_factors`),并加 `get_flashinfer_mm_encoder_autotune_config` 判断 eligibility(需支持多模态输入、开关打开、`skip_mm_profiling` 未设)。
- `_run_flashinfer_autotune_mm_encoder`:按 modality 用 `MultiModalBudget` 的 profiling 输入跑 encoder(`encoder_runner.execute_mm_encoder(dummy_inputs)`);data 模式下 `mm_encoder_tp_mode == "data"` 时把 item 数取整到 tp_size 倍数使 dummy items 均分到各 rank(预算小于 TP size 时 warn 并跳过该 modality)。保留 FlashInfer per-op buckets(raw vision token 计数可超 projected embeddings 的预算)。
- 缓存身份:encoder tuning 打开时,cache hash 增加 `_flashinfer_mm_encoder_autotune_hash`(encoder_config、enable_chunked_prefill、video_pruning_rate/method、mm_processor_kwargs、limit_per_prompt),防止 skip-profiling 或不同 processor 设置错误地命中 encoder 覆盖;IPC(`mm_tensor_ipc`)不影响该 hash。
- B200 CI 新增 optional nightly step(2 GPU),并接入 Nemotron 回归(检查真实 vision GEMM tactics 与 embedding 一致性)。

## 关键改动

```python
# vllm/model_executor/warmup/kernel_warmup.py(pack diff 摘录)
def _flashinfer_mm_encoder_autotune_num_items(
    num_items: int, mm_config: "MultiModalConfig", tp_size: int
) -> int:
    # Identical dummy items must split evenly for matching rank caches.
    if mm_config.mm_encoder_tp_mode == "data":
        return num_items // tp_size * tp_size
    return num_items

def _run_flashinfer_autotune_mm_encoder(runner: "GPUModelRunner") -> None:
    mm_config = get_flashinfer_mm_encoder_autotune_config(runner.vllm_config)
    if mm_config is None:
        return
    ...
    for modality in budget.mm_max_toks_per_item:
        num_profile_items = budget.mm_max_items_per_batch[modality]
        num_items = _flashinfer_mm_encoder_autotune_num_items(
            num_profile_items, mm_config, tp_size
        )
        ...
        dummy_inputs = budget.get_dummy_encoder_profile_inputs(
            modality, max_items_per_batch=num_items
        )
        outputs = encoder_runner.execute_mm_encoder(dummy_inputs)
        del outputs
```

```diff
 # vllm/config/kernel.py
+    enable_flashinfer_mm_encoder_autotune: bool = False
+    """Include multimodal encoders and projectors in FlashInfer autotuning.
+    Disabled by default because profiling can increase startup time. Requires
+    enable_flashinfer_autotune and Model Runner V2. Skipped when
+    skip_mm_profiling is set."""
```

## 性能声明

- 离线 embedding(`nvidia/Nemotron-3.5-Embed-1B-NVFP4`,`flashinfer_cutlass`、compiled vision encoding、两臂 LM tactics 相同;off/on/on/off 序列、每配置 2 次、唯一图片避免 mm cache hit):448x448 batch 1/4/8 为 43.59 → 46.97(+7.7%)、153.74 → 163.35(+6.3%)、262.12 → 278.48 images/s(+6.2%);1120x1120 batch 1/4/8 为 32.89 → 33.04(+0.5%)、34.26 → 34.50(+0.7%)、34.05 → 34.69(+1.9%)(重复运行间吞吐波动至 2.1%,1120x1120 的小增益应据此解读)。
- ViDoRe finance_en image-only:`enable_flashinfer_mm_encoder_autotune` 54.03 req/s vs main 52.67 req/s(约 +2.6%;main=upstream/main 831dd92674,两次启动 52.73/52.61 vs 54.07/53.99)。
- 启动代价:fresh encoder tuning 使启动从 16.1 s 增至 29.2 s(compilation 与 LM caches 已热)——这也是默认关闭的原因。
- 精度:embedding agreement checks 通过;two-GPU 启动验证覆盖 skipped profiling、fresh tuning、cached tuning(vision tactics 跨 rank 一致、embeddings 跨运行一致、cached 启动不改 cache 文件)。

## 验证与回归风险

- `tests/model_executor/test_flashinfer_autotune_warmup.py`:78 passed(新增 item 均分、hash 身份覆盖 profiling 输入/剪枝方法/chunked prefill、eligibility 等测试);`tests/models/multimodal/test_nemotron_embed.py` 3 cases 均可收集;新回归文件 `tests/model_executor/test_flashinfer_encoder_autotune.py` 要求 SM100+ 与 FlashInfer,含单/双卡 vision GEMM tactics 与缓存身份测试。
- 风险(正文明示):需要 Model Runner V2 与 FlashInfer autotuning;`skip_mm_profiling` 时整个 encoder tuning 被跳过;该 checkpoint 不练习真正的 encoder data parallelism;多模态 profiling 增加启动时间(默认关闭即为此权衡)。

## 关联

- 相关 #63159 改现有 LM sweep 的 bucket 处理,不加 encoder profiling;正文明示 open-PR 搜索未发现其他 encoder-warmup 实现。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
