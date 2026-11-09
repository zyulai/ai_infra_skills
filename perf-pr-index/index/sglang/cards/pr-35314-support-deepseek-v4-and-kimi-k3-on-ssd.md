# PR #35314: Support deepseek v4 and kimi k3 on ssd

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @beyondHJM |
| 合并 | 2026-08-26 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | quant, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用但超长截断(39 个文件未展示) |
| 链接 | https://github.com/sgl-project/sglang/pull/35314 |

## 一句话结论

新增 `expert_pack` 加载格式:路由专家权重驻留 SSD、按 router 选择按需加载,使权重超显存+内存的 DeepSeek-V4-Flash/Kimi-K3 可在 RTX 5090 上推理。

## 动机

DeepSeek-V4-Flash 与 Kimi-K3 这类超大 MoE 的完整权重可能大于 GPU VRAM 与主机内存之和,常规加载路径无法运行。本 PR 提供 opt-in 的 SSD 支撑推理路径:被路由命中的专家才从 SSD 读取,其余仍驻盘,使单卡 32 GB(5090)+ 32 GB 主机内存的"边缘设备"也能跑这些模型。

## 优化点

机制上由四件事组成:(1) 专家优先(expert-major)的 Expert Pack 排布,优化随机访问;(2) 直接 I/O 到对齐的 pinned 主机缓冲;(3) 异步 host-to-device 传输;(4) 按字节预算的 LFU/LRU GPU cache 缓存高频专家。运行时不剪枝、合并、跳过、替换或重量化专家,也不降低 Top-K,因此数值路径与标准加载一致。进入路径仅在 `--load-format expert_pack` 时启用,默认 `auto`/`safetensors`/`gguf` 加载不变。已验证输入:DeepSeek-V4-Flash 的单文件 MXFP4 GGUF(路由专家 MXFP4、大矩阵 Q8_0、router 输入 BF16),Kimi-K3 的 38-shard Q2_K/Q3_K text-only GGUF;expert pack 首次启动时构建(DeepSeek 估 5–10 分钟,Kimi-K3 实测 29 分 42 秒),之后复用。diff 还包含 MXFP4 matvec/marlin repack JIT CUDA kernel(`mxfp4_matvec`、`mxfp4_matvec_dual` gate/up 融合、`mxfp4_marlin_repack`)与 `expert_pack_hook` 参数校验,并附带的 dsv4 compress 与 kimi_k3 attn_res dtype 修正。

## 关键改动

```text
python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cu: 新增 MXFP4 matvec 与 Marlin repack CUDA kernel(565 行)
python/sglang/srt/arg_groups/expert_pack_hook.py: expert_pack 参数归一化与启动期错误汇总
python/sglang/srt/configs/load_config.py: LoadFormat 增加 EXPERT_PACK
python/sglang/srt/configs/model_config.py: is_kimi_k3 识别 KimiK3LinearForCausalLM
```

```python
+    if server_args.load_format != "expert_pack":
+        return
...
+        disable_cuda_graph=True,
+        disable_shared_experts_fusion=True,
```

## 性能声明

目标平台:单 RTX 5090(32 GB VRAM)、32 GB 主机内存、NVMe SSD,每 prompt 生成 200 token:

- DeepSeek-V4-Flash vs Ollama(5 prompts,200 token/prompt):prefill 3.265 vs 1.457 token/s(2.24x),decode 2.020 vs 1.354 token/s(1.49x)。
- Kimi-K3 vs llama.cpp(91-token prompt 生成 200 token):prefill 0.66412 vs 0.11289 token/s(5.88x),decode 0.44830 vs 0.1810 token/s(2.48x;llama.cpp decode 基线来自较早的 50-token 运行,该数字为指示性而非严格等长对比)。
- 首次可用时间:DeepSeek-V4-Flash 约 8–15 分钟(含 pack 构建),Kimi-K3 约 35–45 分钟;38 个 Kimi-K3 shard 加 Expert Pack 合计约 1.814 TiB,建议 4 TB SSD。

## 验证与回归风险

- 正确性:DeepSeek-V4-Flash 与 Ollama 在多类 prompt 上语义等价;Kimi-K3 与 200-token SGLang 参考输出完全一致,92 个路由层 Top-16、`fallback_count=0`、`io_errors=0`。
- 回归风险:默认与 safetensors/gguf 路径不改;expert_pack 强制关闭 CUDA graph 与 shared-experts fusion(为正确性),单文件 GGUF 校验、TP/DP/EP 必须为 1;Kimi 完整多模态 checkpoint 不支持,其他 GGUF 量化变体未验证。PR CI 三条(Base/Extra/AMD ROCm 7.2)显示失败。
- 依赖本地 SSD 带宽,解码吞吐绝对值极低(token/s 个位数),适用面是资源受限单卡场景。

## 关联

- 同区域:DeepSeek/Kimi 模型特化路径系列;DeepSelect JIT kernel #40556 同样服务 DeepSeek V4.x。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
