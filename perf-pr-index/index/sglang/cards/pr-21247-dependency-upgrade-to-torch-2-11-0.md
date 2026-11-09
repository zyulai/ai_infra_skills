# PR #21247: [Dependency] Upgrade to Torch 2.11.0

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-05-02 |
| 分类 | Other(tier: major) |
| Label | documentation, high priority, amd, dependencies, Multi-modal, deepseek, sgl-kernel, npu, run-ci, diffusion |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留部分文件并注明 "8 file(s) dropped"(pyproject 等版本 pin 文件不在保留范围内) |
| 链接 | https://github.com/sgl-project/sglang/pull/21247 |

## 一句话结论

全仓升级到 PyTorch 2.11.0(沿用 #18862 的升级路径),配套 sglang-kernel 0.4.2、多模态 tokenizer 兼容 shim、DeepSeek 双流路径与 torch.compile 的互斥保护。

## 动机

正文极简:动机即 PyTorch v2.11.0 官方 release,修改方式沿用此前 torch 升级 PR #18862 的做法。属于框架依赖换代 PR,性能收益(若有)来自新版本 torch 本身,PR 未单独声明。

## 优化点

保留 diff 内可见的适配点:(1) multimodal_gen 组件加载器与 CLIP processor 增加 `RobertaProcessing` 兼容 shim(tokenizers 新版把 `sep`/`cls` 从 kwargs 收紧为位置参数),加载失败时回退 `use_fast=False`;(2) `deepseek_v2.py` 的 dual-stream fused-shared-experts 快路径在 `enable_torch_compile` 且 token 数不超过 `torch_compile_max_bs * speculative_num_draft_tokens` 时让路给 `forward_normal_dual_stream`,避免与 torch.compile 捕获冲突;(3) `engine.py` 把 sglang-kernel 最低版本从 0.4.1.post1 提到 0.4.2;(4) 删除 `scripts/ci/cuda/cache_nvidia_wheels.sh`(pypi.nvidia.com 的本地 wheel 缓存脚本);(5) 多模态一致性测试阈值(ssim/psnr/mean_abs_diff)与 perf 基线数值按新 torch 行为重校,并新增一致性失败时的 artifact 保存(`save_consistency_failure_artifact`)。

## 关键改动

```text
python/sglang/srt/models/deepseek_v2.py: torch.compile 范围内禁用 dual-stream 快路径
python/sglang/multimodal_gen/.../component_loader.py, test_utils.py: RobertaProcessing 兼容 shim
python/sglang/srt/entrypoints/engine.py: sglang-kernel >= 0.4.2
.github/workflows/*.yml: AMD rocm720 / pr-test-amd / jit-kernel / multimodal-gen CI 更新
scripts/ci/cuda/cache_nvidia_wheels.sh: 删除
```

## 性能声明

无。正文未含任何 benchmark 数字或 perf claim(素材正则未检出),PR 未注明测试条件。多模态 `perf_baselines.json` 仅是测试基线数值更新(如个别 slot 190→221/307),不是性能声明。

## 验证与回归风险

依赖 CI 矩阵(pr-test-amd、rocm720、jit-kernel、multimodal-gen 等 workflow 同步改)与新 torch 下的多模态一致性阈值重校。风险面大:torch 大版本换代影响全部后端(CUDA/AMD/NPU)、全部模型路径(DeepSeek、diffusion、多模态);多模态一致性阈值普遍下调(如 qwen_image_edit_2511_ti2i ssim 0.95→0.83)说明新 torch 下生成数值有漂移;`cache_nvidia_wheels.sh` 删除会改变 CI 的 NVIDIA wheel 安装方式;cudnn 9.16.0.29 / nvshmem 3.3.20 等 pin 见脚本注释(已删)。

## 关联

前序:#18862(上一次 torch 升级)。上游:pytorch v2.11.0 release。同区域:其他依赖升级 PR(#31927、#48468 FlashInfer bump)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
