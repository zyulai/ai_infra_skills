# PR #54053: [feature] Watermarked generation and detection (Gumbel-max algorithm)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @TQCB |
| 合并 | 2026-09-10 |
| 分类 | Other（tier: major） |
| Label | documentation, performance, frontend, ready, mrv2 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 12 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/54053 |

## 一句话结论

为 vLLM 引入基于 Gumbel-max 的文本水印生成与检测，含 engine 级 key/PRF 与按请求 opt-out。

## 动机

PR 之前 vLLM 没有文本水印能力。本 PR 落地 RFC #53916 的提案，在 vLLM 内实现文本水印，用于生成文本的溯源与检测；设计决策、论证与实证结果均放在该 RFC 中，PR 正文只列实现范围与测试入口。

## 优化点

新增使用 Gumbel-max 算法的水印 sampler、配套的检测原语、用于 keyed generation 的稳定 PRF（默认 philox）、按请求 `watermarking=false` 的 opt-out（算法与 key 保持在 engine 级），以及一个检测示例 endpoint。配置层新增 `WatermarkConfig`（key、algorithm="gumbel"、context_width=4、prf="philox"）；`VllmConfig` 在启用水印时强制走 Model Runner V2，并对 speculative decoding、beam search、模型自定义 sampler 抛错；`model_runner` 在 `watermark_config` 非 None 时把 `Sampler` 换成 `GPUWatermarkSampler`。

## 关键改动

```text
vllm/config/watermarking.py: 新增 WatermarkConfig（key/algorithm/context_width/prf）
vllm/v1/watermarking/gumbel.py: GumbelWatermarker，Gumbel-max 采样与检测打分
vllm/v1/watermarking/gpu_sampler.py: GPUWatermarkSampler，按请求读 watermarking 开关
vllm/v1/watermarking/prfs/philox.py + detector.py: 稳定 PRF 与检测原语
vllm/v1/worker/gpu/model_runner.py: 按 watermark_config 切换到水印 sampler
```

```python
    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        _random_sample: RandomSampler,
    ) -> WatermarkSample:
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample
```

## 性能声明

PR 正文未给出定量性能数据。正文仅说明按 RFC #53916 的 "Implementation plan"/"Results" 章节，用标准工具与评测测试推理性能与生成质量，具体数字与测试条件未在本 PR 正文列出。

## 验证与回归风险

`tests/engine/test_arg_utils.py` 新增 `test_watermark_config_cli`，校验 `--watermark-config` 解析出的 algorithm=gumbel、key=42、context_width=4、prf=philox 与 `supports_speculative_decoding=False`。风险面：启用水印会强制 MRv2 并禁用 spec decode / beam search / 自定义 sampler；按请求 opt-out 牵动 chat/completions/responses 等 protocol 与 `sampling_params.py`；`GPUWatermarkSampler` 在 temperature=0 的贪心请求上退化为普通采样。pack 末尾注明 12 个文件因体积被从 diff 剔除，完整 diff 需看 PR 网页。

## 关联

PR 正文依据 RFC #53916（issue，非 PR）。素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
