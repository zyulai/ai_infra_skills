# PR #41501: [Feat] diffusion/sana-video2: Support breakable CUDA graphs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @cr-gao |
| 合并 | 2026-11-12 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | documentation, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/41501 |

## 一句话结论

SANA-Video 2.0 接入共享 breakable CUDA graph (BCG) runner:warmup 捕获 T2V/TI2V 两个 signature、复用 RoPE 表、质量切换时 drop;A800 上 denoise 日志每步降 16.9%/20.4%,但 e2e 仅 +0.6%/+0.4%,峰值显存 20.65→21.32 GiB、启动(含 warmup)23.8→62.7 s。

## 动机

为 native SANA-Video 2.0 启用 breakable CUDA graphs,沿用 #35729 的做法。

## 优化点

- 把 custom samplers 接到共享 BCG runner:warmup 期间 capture T2V 与 TI2V 两个 signature,匹配的请求 replay,未见过的 shape eager 运行。
- Transformer 保持常驻;RoPE frequencies 缓存到 device(`_RopeFreqs.forward` 把 `_freqs` 搬到输入 device;captured graphs 按地址读这些表,stage 以 `_bcg_ropes[transformer._ropes_key]` 保住引用直到 graphs 被 drop)。
- `validate_server_args`:`enable_breakable_cuda_graph` 时要求 transformer component resident(`require_component_resident("transformer", ...)`),仅在开关打开时生效(有对应单测)。
- `drop_breakable_cuda_graphs` 泛化:base stage 增加默认 no-op,`gpu_worker._drop_breakable_cuda_graphs` 对所有 stages 调用(原来只对 `DenoisingStage`);quality 变化时 drop 已捕获 graphs 并 warning("requests run eager until restart")。
- Server args 注册 `SanaVideo2PipelineConfig` 与 `sana-video_2.0_5b_720p` 模型别名;cookbook 增加 serving 示例(`--enable-breakable-cuda-graph --component-residency=all=resident --warmup-resolutions 832x480 --warmup-num-frames 81`)。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sana_video2.py: _prepare_bcg_runner(warmup 捕获 T2V+TI2V)/ drop_breakable_cuda_graphs / quality 变化 drop
python/sglang/multimodal_gen/runtime/models/dits/sana_video2.py: RoPE freqs 缓存 device
python/sglang/multimodal_gen/configs/pipeline_configs/sana_video2.py: BCG 开启时要求 transformer resident
python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py: drop_breakable_cuda_graphs 默认 no-op
python/sglang/multimodal_gen/runtime/managers/gpu_worker.py: 对所有 stage drop(不再 isinstance DenoisingStage)
python/sglang/multimodal_gen/runtime/server_args/server_args.py: 注册 SanaVideo2PipelineConfig / sana-video_2.0_5b_720p
docs/cookbook/diffusion/SANA-Video/SANA-Video.mdx: BCG serving 示例
python/sglang/multimodal_gen/test/unit/test_sana_video2_pipeline.py / test_sana_video2_sequence_parallel.py: 回归测试
```

```python
def _prepare_bcg_runner(self, batch, embeds, mask):
    if self._bcg_runner is None:
        from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
            DiffusionBreakableCudaGraphRunner,
        )
        self._bcg_runner = DiffusionBreakableCudaGraphRunner(
            self.transformer, batch.latents.device
        )
    if batch.is_warmup:
        inputs = batch.latents
        if batch.do_classifier_free_guidance:
            inputs = torch.cat([inputs, inputs])
        timestep = torch.full((inputs.shape[0],), 1000.0, device=inputs.device,
                              dtype=torch.float32)
        frame_timestep = (... .expand(-1, 1, inputs.shape[2], 1, 1).clone())
        frame_timestep[:, :, 0] = 0
        # Capture both T2V and TI2V signatures during startup.
        for warmup_timestep in (timestep, frame_timestep):
            self._bcg_runner.capture(
                hidden_states=inputs, timestep=warmup_timestep,
                encoder_hidden_states=embeds, encoder_attention_mask=mask,
            )
        # Captured graphs read these tables by address, and the transformer
        # replaces them when a request has another shape.
        self._bcg_ropes[self.transformer._ropes_key] = self.transformer._ropes
    return self._bcg_runner
```

## 性能声明

- 环境:A800,832×480、81 frames、50 steps、guidance 8、seed 42、all components resident、torch.compile disabled;warmup 后 3 个请求的均值;e2e 含视频保存。
- T2V:eager denoise 692.8 ms/step → BCG 576.1(−16.9%);logged decode 0.12 s → 3.71 s;saved-request e2e 35.96 → 35.74 s。TI2V:eager 694.2 → BCG 552.3 ms/step(−20.4%);logged decode 0.12 s → 4.75 s;saved-request e2e 36.06 → 35.91 s。
- 正文如实注明:logged denoise 降 16.9%/20.4%,而 e2e 仅提升 0.6%/0.4%;stage timing 未同步,排队中的 GPU work 可能被记到 decode 或后续同步上,denoise 数字不是孤立的 GPU 加速。
- 资源:peak reserved memory 20.65 → 21.32 GiB;startup including warmup 23.8 → 62.7 s。

## 验证与回归风险

- A800 上 T2V/TI2V 视频与 eager 完全一致(含 decoded frames);changed prompts/seeds、restored inputs、未捕获的 frame count 也匹配(lifecycle probes 用四步)。Focused tests 与 pre-commit passed。
- 单测覆盖:warmup-only 捕获两 signature(T2V/TI2V × CFG on/off)、quality 变化 drop captured graphs("quality changed")、RoPE 表保活到 graphs drop(weakref 断言)、worker resume drop denoising graphs、BCG 关闭时直接调用 transformer、BCG 仅在开启时要求 transformer resident。
- 风险:BCG 要求 transformer resident;quality 变化后 graphs 被 drop、eager 直到重启;其他分辨率/帧数 eager;启动时间显著增加(23.8→62.7 s)与显存增加(约 0.67 GiB)。

## 关联

#35729(正文注明 following)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
