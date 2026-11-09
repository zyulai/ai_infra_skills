# PR #33838: [AMD] Perf Kimi-K3 MoE optimization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kkHuang-amd |
| 合并 | 2026-09-03 |
| 分类 | MoE(tier: major) |
| Label | amd, run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/33838 |

## 一句话结论
对齐 AITER 的 Kimi-K3 SiTUv2 MXFP4 权重布局优先级并修 A4W4 正确性,MI35x TP8 吞吐最高 +4.75%。

## 动机
Kimi-K3 SiTUv2 MXFP4 MoE 对不同激活量化模式需要不同的 preshuffled 权重布局:A16W4/A8W4 用 AITER 的 GU-interleaved `shuffle_weight_a16w4` 布局,A4W4 用通用 separated 布局,且 A8W4/A4W4 标志同时设置时 AITER 以 A8W4 优先。此前 SGLang 侧的布局选择未与该优先级对齐——把 A16/A8 布局喂给 A4W4 会导致 real-checkpoint 的 MoE 输出接近正交(A4W4 GSM8K 只有 0.005)。另有两处小损耗:correction_bias 的 dtype 转换每步重复做,以及 SiTU runner 返回的 FFI wrapper 张量未被零拷贝契约使用。

## 优化点
(1) `mxfp4.py` 新增 `_aiter_situ_uses_gu_interleaved_weights`,按 AITER 的激活模式优先级(`a8w4 or not a4w4`)选权重布局:SiTU 激活下 A16W4/A8W4 走 GU-interleaved preshuffle,A4W4 走通用 separated shuffle;(2) `topk.py` 的 `TopKConfig` 增加 `correction_bias_for_dtype` 惰性缓存,以 (data_ptr, _version, device, dtype, 目标 dtype) 为 key,权重重载后自动失效,AITER grouped topk 路径复用转换结果;(3) SiTU routed MoE 返回显式 published 输出缓冲 `symm_output` 而非 FFI 返回的 wrapper/新分配,明确 K3 零拷贝 latent buffer 的指针契约。依赖 ROCm/aiter#4534(revision `6dc26b7a8`,含 Kimi-K3 Opus/FlyDSL A8W4 SiTUv2 MoE dispatch,依赖 FlyDSL 0.3.0)。

## 关键改动

```text
python/sglang/srt/layers/quantization/mxfp4.py: 布局优先级 _aiter_situ_uses_gu_interleaved_weights;SiTU 输出改回 symm_output
python/sglang/srt/layers/moe/topk.py: TopKConfig.correction_bias_for_dtype 惰性缓存(版本号失效)
test/registered/unit/layers/moe/test_topk_correction_bias_cache.py 等 3 个新测试: 布局优先级/缓存失效/零拷贝输出契约
```

## 性能声明
8x MI35x/gfx950、TP8、公开 Kimi-K3 checkpoint;client 为 random 数据集、固定 input/output 8192/1024。相对原 MI35x reference:concurrency 2/4/8/16/32 的 total throughput delta 分别 +2.95%/+4.75%/+1.18%/+2.06%/+2.68%(concurrency 32 为 5381.41 tok/s,E2EL -2.55%、TTFT -3.57%、TPOT -2.54%、ITL -2.29%)。MoE copy 改动相对前一轮同配置 sweep:吞吐 +0.42%–1.57%,TPOT/ITL 在所有并发下均有改善。另注明 A4W4 数值正确但非性能默认:median output throughput 530.84 tok/s vs 修复后 A8W4 537.33 tok/s。

## 验证与回归风险
注册了 AMD/CPU/CUDA 单测覆盖布局优先级、correction-bias 缓存与重载失效、零拷贝输出契约(11 passed);AITER strict K3 A4W4 oracle(SiTU beta 4.0/linear beta 25.0)17 例全过;GSM8K:A4W4 200 题修复后 0.980(修复前 0.005)、A8W4 200 题 0.980。风险:行为受多个环境变量组合(`SGLANG_USE_AITER`、`AITER_SITUV2_A8W4/A4W4` 等)控制,且硬依赖特定 AITER revision;CI 状态显示 PR Test (Base)/(Extra)/(AMD ROCm 7.2) 为失败标记。

## 关联
依赖 ROCm/aiter#4534 与 ROCm/aiter#4463(A4W4 SiTUv2 模式)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
