# PR #48399: [Refactor] env: Dispatch on the GPU architecture instead of env vars that restate it

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-11-06 |
| 分类 | Model-specific Fast Path |
| Label | documentation, quant, amd, deepseek, speculative-decoding, hicache, npu, run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用,另有 85 个文件因超长被剔除) |
| 链接 | https://github.com/sgl-project/sglang/pull/48399 |

## 一句话结论

删除 41 个复述硬件的环境变量、改由 `current_arch()` 在读取点判定,并修复一批错误架构门控;同时把 5 个有据可查的架构 fast path 转为默认开启。

## 动机

SGLang 声明约 910 个环境变量(819 在 `srt/environ.py`,91 在 `multimodal_gen/envs.py`),另裸读约 115 个;其中大量只是复述硬件——某 GPU 上必须手工打开的路径开关,或本就按架构门控的路径的 kill switch。这类变量造成配置漂移与错误门控,改为在读取点用 `sglang.kernels.arch.current_arch()`/platform helper 直接判定。

## 优化点

按三条规则处理每个变量:复述硬件的选择器直接删除、读取点查架构;默认开启且按架构门控路径的 kill switch 删除(文档要求或 CI 用作 A/B 对照的保留);默认关闭的单架构 fast path 仅把证据最强的 5 个转默认开启(cookbook 本就只在该硬件上设置,GSM8K 可查的都验过)。共删 41 个(33 声明、8 裸读),4 个改为按文档名读取。行为变化包括:DSV4 on gfx950 的 aiter batched GEMM/fused invrope/C4 indexer rowgroup/FlyDSL MXFP8 默认开启;DSA `flashmla_sparse_q8` 每行 top-k 长度恒开(按构造 bit-exact);MiniMax-H3 VAE fused residual-add+RMSNorm 只在 lossless/high 跑,`exact` 恢复与 unfused 逐字节一致;`SGLANG_FLASHINFER_USE_TENSOR_CORE` 修为布尔解析(旧实现 `=1` 反而关掉 tensor core);`SGLANG_USE_DEEPGEMM_BMM` 限 SM90;V4.1 indexer/DSA workspace 限 SM100/SM10x(此前 SM120 会误入并在 build 时报错);DeepSeek-V4 ROCm hook 不再无条件覆盖用户值;MegaMoE on ROCm 直接走 aiter MegaMoEv2;gfx1250 启动时自动 `os.environ.setdefault("AITER_FORCE_A8W4", "1")`。

## 关键改动

```text
python/sglang/srt/environ.py 与 multimodal_gen/envs.py: 删除硬件选择器/kill switch 变量
python/sglang/srt/arg_groups/model_hook.py: ROCm/CUDA hook 不再覆盖用户显式设置
python/sglang/srt/arg_groups/platform_hook.py: gfx1250 自动设置 AITER_FORCE_A8W4
python/sglang/kernels/ops/attention/flash_attn/cute/interface.py: FA4 tmem_load_red_max 环境开关移除
python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/vae_vit.py: fused norm 改由 VaeFastPathGate 门控
```

## 性能声明

本 PR 无直接加速声明,给出的相关数字:`SGLANG_AITER_CAPTURE_COPY_IN` 保持关闭——Qwen3-30B-A3B TP4/DP4 DP-attention(MI350X)上打开会损失约 2.7% 输出吞吐(GSM8K 不变);FA4 `ld.red` 行 max 在 GB300 两轮 benchmark 中 d64 ViT 形状慢 5-7%、d72/d80 快 3-10%(删除该 kill switch 后无运行时回退开关,此数据作为决策记录)。

## 验证与回归风险

全量 sweep:test/registered/unit 与 multimodal_gen/test/unit 全部 1,561 个文件加触及的 kernel 测试,双树逐文件独立进程运行,H200(sm_90)/GB300(sm_103)/RTX 5090(sm_120)/MI350X(gfx950)均 0 个仅分支失败。GSM8K(sgl-eval 0.1.3,DeepSeek-V4-Flash TP4 200 题):GB300 98.0→97.5%;MI350X 多组 97.0-98.5%(双侧均在 97-98.5% 区间);Qwen3-30B-A3B 全 1,319 题 0.9545-0.9598。定向检查:MiniMax-H3 VAE `exact` 与 unfused 逐字节一致(base 差至 1.36e-2);FA4 sm_103 FMNMX vs ld.red bit-exact 62 passed;FlyDSL MXFP8 与 fp32 参考差在 bf16 舍入内。未上硬件验证:H20 78-SM 规则、gfx1250/gfx1151/gfx115x(仅 arch-mock 单测)、MegaMoE EP8、DGX Spark、MuseSpark、MiniMax-M3 NVFP4 e2e。风险面:41 个变量被删,依赖旧变量的用户脚本/文档需迁移(4 个文档名生效、旧名部分保留为 deprecated alias);多处默认行为翻转(gfx950 DSV4、MiniMax-H3 exact、FlashInfer tensor core 解析)。CI:Extra 失败(x),Base/AMD pending。

## 关联

同文件重叠:PR #48366(19 个相同文件,删不同变量,后落地方需 rebase);关联:#43077(SGLANG_AITER_CAPTURE_COPY_IN 置 off)、#41308(gfx950 V4.1 multi-stream 分支)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
