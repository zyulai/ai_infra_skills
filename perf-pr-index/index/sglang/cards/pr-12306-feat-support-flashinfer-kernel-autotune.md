# PR #12306: feat: support flashinfer kernel autotune

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @elvischenv |
| 合并 | 2025-11-29 |
| 分类 | Attention / Kernel |
| Label | high priority, run-ci |
| Diff | diffstat 缺失(素材未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/12306 |

## 一句话结论

在 CUDA graph 捕捉前插入 FlashInfer autotune 预热(dummy forward),gpt-oss-120b 随机负载吞吐 +16%。

## 动机

FlashInfer MoE 需要 autotune 才能选出最优 kernel,而这一过程必须在 cudagraph capture 之前完成,否则捕捉进 graph 的 kernel 配置不是最优,运行期无法再调整。

## 优化点

新增 `--enable-flashinfer-autotune`(默认关闭);ModelRunner.initialize 在 init_attention_backend 之后、init_device_graphs(即 CUDA graph 捕捉)之前调用 kernel_warmup()。`_should_run_flashinfer_autotune` 限定 CUDA、SM major ≥ 9、attention_backend ∈ {flashinfer, trtllm_mla, trtllm_mha},speculative(EAGLE/standalone/ngram)时跳过 draft worker;`_flashinfer_autotune` 在 `torch.inference_mode()` + flashinfer `autotune()` 上下文中,以 req_to_token_pool.size 为 batch_size 跑一次完整 `_dummy_run`(构造 ForwardBatch,覆盖 DP/PP/LoRA/EAGLE/Ngram 分支),让 FlashInfer MoE kernel 完成 tuning。另:mxfp4 trtllm moe 调用传入 `tune_max_num_tokens=next_power_of_2(tokens)`,并把 x_scale reshape 修正为保留 token 维;hybrid_linear_attn_backend 在 dummy run(retrieve_next_token 为 None)时跳过 tensor 创建。

## 关键改动

```text
python/sglang/srt/model_executor/model_runner.py: 新增 kernel_warmup/_flashinfer_autotune/_dummy_run,warmup 插在 cuda graph capture 之前
python/sglang/srt/server_args.py: 新增 --enable-flashinfer-autotune(默认 False)
python/sglang/srt/layers/quantization/mxfp4.py: trtllm moe 调用传 tune_max_num_tokens;x_scale reshape 保留 token 维
```

```python
def _flashinfer_autotune(self):
    from flashinfer.autotuner import autotune
    logger.info("Running FlashInfer autotune...")
    with torch.inference_mode(), autotune():
        self._dummy_run(batch_size=self.req_to_token_pool.size)
    logger.info("FlashInfer autotune completed.")
```

## 性能声明

openai/gpt-oss-120b,bench_serving(backend sglang-oai,random dataset,random-input-len 1024 / random-output-len 1024,random-range-ratio 1,max-concurrency 512,2560 prompts):PR:request throughput 18.51 req/s,total token throughput 37,900.23 tok/s,median ITL 21.66 ms,mean E2E 27,573.10 ms;main:15.84 req/s,32,450.49 tok/s,median ITL 26.49 ms,mean E2E 32,188.29 ms;TTFT 基本持平(2658.05 vs 2632.67 ms)。PR 标注 16% perf improvement。PR 未注明 GPU 型号。

## 验证与回归风险

lm_eval gsm8k(limit 0.5)flexible-extract 0.8394 / strict-match 0.6318,与 main 完全一致。影响面:默认关闭(opt-in);开启后启动阶段多一次 max-batch dummy forward 的预热开销;仅覆盖 flashinfer/trtllm_mla/trtllm_mha attention backend 且需 SM≥9;speculative 场景只对 target worker 生效;tune_max_num_tokens 会影响 trtllm moe 的 autotune 搜索上界。

## 关联

无(素材未列出同区域其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
