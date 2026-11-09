# PR #24660: [diffusion] fix: further align ltx2.3 accuracy with tp

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24660](https://github.com/sgl-project/sglang/pull/24660) |
| 作者 | mickqian |
| 合并日期 | 2026-05-11 |
| labels | documentation, run-ci, diffusion, jit-kernel |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
LTX-2/2.3 多卡与 HQ 路径对官方输出偏差:改用 TP/CFG 并行、RoPE 保持设备端生成并遵循 frequencies_precision、修 CFG 广播 rank 等,并收紧 CI 阈值。

## 问题现象
LTX-2 / LTX-2.3 的扩散输出与官方生成结果存在偏差,多 GPU CI 案例与 LTX-2.3 HQ two-stage 路径尤为明显:此前多卡 CI 部分案例阈值过松,且可能选用比官方输出更不对齐或更慢的并行模式(SP/Ulysses);HQ 路径在 PR #23366 的 SpongeBob 复现上出现回归。正文给出量化对比:该复现(1 GPU, `LTX2TwoStageHQPipeline`, 768x512, 241 frames, 30 steps, seed 10)old PR attachment vs official `PSNR=20.707109`,修复后 `PSNR=20.722057`;`ltx_2.3_one_stage_ti2v`(2 GPUs, CFG parallel)测得 `clip=0.9396, SSIM=0.8254, PSNR=22.2667, mean_abs_diff=8.3209`;`ltx_2.3_two_stage_t2v_2gpus` 测得 `clip=0.9545, PSNR=16.4739, mean_abs_diff=26.4187`。

## 根因
正文按修改项披露多处不对齐:SP/Ulysses 并行模式在部分案例损害精度或性能;LTX-2.3 HQ 对 RoPE 舍入敏感,而频率生成被缓存在 CPU/NumPy 张量上、且未遵循 LTX/Diffusers 配置的 `frequencies_precision="float64"`(diff 注释:"LTX-2.3 HQ is sensitive to RoPE rounding; keep frequency generation on the target device instead of caching a CPU/NumPy tensor");HQ 的 res2s noise dtype 边界与 stage-2 sigma-space res2s 数学出现回归(正文称「restore」);CFG 并行广播 pipeline batch 时 `src=0` 不是该 process group 的全局源 rank(diff 注释:"`dist.broadcast(src=...)` expects a global rank for process groups");显式 `--quantization` 被 checkpoint 元数据覆盖。此外 transformer fp8-cast 经验证对本 PR 目标 CI 路径无实质帮助,被有意排除。

## 修复方案
(diff 在 40000 字符处截断,以下基于已获取部分;截断处之后可能还有阈值 JSON、注意力后端继承、FA3 varlen、quantization 优先级等未展示改动。)
- `attention/backends/sdpa.py` 与 `attention/layer.py`:新增 `allow_cudnn_sdp` 参数,为 True 且在 CUDA 上时用 `sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)`(CUDNN_ATTENTION/FLASH/EFFICIENT/MATH)包裹 SDPA 调用,默认关闭;`models/dits/ltx_2.py` 的 LTX2 注意力显式传 `allow_cudnn_sdp=True`(注释:"official LTX2 torch_sdpa uses cuDNN; cuda setup disables it")。
- `models/dits/ltx_2.py`:删除 `functools.lru_cache` 的 NumPy 频率网格,`torch.linspace` 直接在目标设备上以 float64/float32 生成;`rope_double_precision` 改为先读 `frequencies_precision`(hf_config 或 arch),为 `"float64"` 才开双精度。
- `encoders/gemma_3.py`(由 diff 推断:LTX2 文本编码器对齐官方 Gemma3):改用 `get_rope` 的 `cos_sin_cache` 做 RoPE(按 HF Gemma3 半维扩展);无 `layer_types` 时回退 `sliding_window_pattern`;滑动窗口掩码方向由 `idx[None,:] - idx[:,None]` 修正为 `idx[:,None] - idx[None,:]`。
- `pipelines/ltx_2_pipeline.py`:删除 HQ 分支手动 `batch.sigmas.append(0.0011)`;`stages/denoising_av.py`:`num_steps` 与 `timesteps` 改由 `distilled_sigmas` 计算,0.0011 注入只改最后一对 sigma、不新增去噪步。
- `executors/parallel_executor.py`:CFG 并行广播改用 `rank=get_world_rank()` 与 `src=cfg_group.ranks[0]`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:LTX2/LTX2.3 各 pipeline 的并行与数值路径(RoPE、sigma 调度、CFG 广播)、Gemma3 文本编码器 RoPE 与滑动窗口掩码、注意力层新增 `allow_cudnn_sdp`(仅显式传 True 的模型走 cuDNN 分支);CI 阈值收紧可能暴露机器间方差(正文称 final CI 仍是 per-machine variance 的最终依据);diff 截断,未展示部分的回归面无法完整评估。

## 关联
- [PR #23366](https://github.com/sgl-project/sglang/pull/23366)(LTX-2.3 HQ SpongeBob 复现)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
