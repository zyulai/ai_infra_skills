# PR #43839: [Perf] [AMD] qwen3_5: Quantize the BF16 GDN projections to MXFP4 online

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @omirosh |
| 合并 | 2026-10-27 |
| 分类 | Speculative Decoding |
| Label | documentation, amd, run-ci |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 environ.py、quark_w4a4_mxfp4.py、qwen3_5.py、新增 AMD 测试与文档行 |
| 链接 | https://github.com/sgl-project/sglang/pull/43839 |

## 一句话结论

Qwen3.5/3.8 的 BF16 GDN 投影在加载后在线量化为 MXFP4 走 AITER MXFP4 GEMM,8x MI355X TP8 上 NEXTN MTP 中位 TPOT 降 5.1%、无 MTP 降 4.0%。

## 动机

Quark MXFP4 checkpoints 把 Qwen3.5/3.8 的 GDN `in_proj_qkvz` 与 `out_proj` 留在 BF16。TP8 下 `in_proj_qkvz` 是每 GPU 4608x8192 的权重,是每个 decode/verify 步都读的最大稠密权重。作者尝试过针对这些形状重调 BF16 GEMM 表(无端到端收益,仅为现表的 99.8%),也没有 BF16 kernel 能在 4608x8192 上超过 hipBLASLt;而 MXFP4 权重连同 scale 只有约四分之一字节数。16 行时 MXFP4 `in_proj_qkvz` 加 BF16 `in_proj_ba` 合计 8.1+7.9µs,而 BF16 打包 GEMM 要 23.1µs。

## 优化点

新增 `AiterMxfp4OnlineLinearMethod`(LinearMethodBase):模型在 `create_weights` 后把该 method 设到 BF16 linear 上,loader 的 `process_weights_after_loading()` 首次调用即用现有 `dynamic_mxfp4_quant` 量化 BF16 权重,把权重(16x16)与 E8M0 scale 做 preshuffle 并记录 AITER 是否有 (N,K) 的 tuned preshuffle 配置,第二次调用为 no-op。`apply()` 用 `per_1x32_f4_quant_hip` 量化激活:≤64 行且有 tuned config 走 Triton `gemm_afp4wfp4_preshuffle`(32 行以下用未 shuffle 的激活 scale),其余 batch 走 ASM `gemm_a4w4(bpreshuffle=True)`。`qwen3_5.py` 在 import 时读取新开关 `SGLANG_ROCM_GDN_ONLINE_MXFP4`(默认 true),仅 gfx95x + `SGLANG_USE_AITER=1` + `qwen3_5_text`/`qwen3_5_moe_text` 模型类型生效;每个 GDN 层在 `__init__` 把该 method 挂到两个投影上,量化发生在 `load_model` 期间、KV cache 定容与 CUDA graph capture 之前——省出的 BF16 显存(Qwen3.8 TP8 约 5.5 GB/GPU)使 KV pool 增长 6.4%。跳过已被 checkpoint 量化或 N/K 非 256 倍数的投影;`in_proj_qkvz` 不再是 UnquantizedLinearMethod,`finalize_fused_in_proj()` 不再构建 packed BF16 qkvz+ba GEMM,`in_proj_ba`(N=32)单独跑 BF16。复用 `Qwen3_5GatedDeltaNet` 的 `qwen4_exp`、`interns2_mobius` 保持 BF16。

## 关键改动

```text
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py: 新增 AiterMxfp4OnlineLinearMethod(quantize_/apply)
python/sglang/srt/models/qwen3_5.py: GDN 层挂载 online MXFP4,prepare_before_cuda_graph_capture 记录计数
python/sglang/srt/environ.py: 新增 SGLANG_ROCM_GDN_ONLINE_MXFP4(默认 true)
test/registered/amd/test_qwen35_gdn_online_mxfp4.py: 两个 TP8 形状 x 8 种行数对 dequant FP32 参考
docs/docs/references/environment_variables.mdx: 新变量一行
```

apply 的 kernel 分派:

```python
if m <= self.PRESHUFFLE_MAX_M and layer.mxfp4_preshuffle_tuned:
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=m >= 32)
    ... gemm_afp4wfp4_preshuffle(...)
else:
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
    y = gemm_a4w4(..., bpreshuffle=True)[:m]
```

## 性能声明

条件:8x MI355X 单节点 TP8,Qwen3.8-2.4T-A95B MXFP4,`rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261016`(main 34c4ba14c8 + AITER 2bb549102b + 本 PR 首个 commit),page size 16、fp8 KV cache、chunked prefill 16384、radix cache off、≤256 running;NEXTN 用 3 steps/top-k 1/4 draft tokens;bench_serve random range ratio 0.8,每格一次。12 个 serving shape 的 geomean:NEXTN MTP 中位 TPOT **−5.1%**、输出 tok/s **+5.9%**;无 MTP TPOT **−4.0%**、tok/s **+4.2%**。单点如 8k/1k c256 TPOT −7.3%/tok/s +8.0%,128k/1k c16 TPOT −6.3%。中位 TTFT geomean 降 8.6%(NEXTN)/6.5%(无 MTP),NEXTN accept length 不降(3.14 → 3.18)。逐内核(单 MI355X,CUDA-graph replay,µs):256 行时 BF16 out_proj 16.0 → MXFP4 8.1;packed BF16 in_proj 48.7 → MXFP4 qkvz+BF16 ba 30.6。Load path check 显示量化移入 load path 后 TPOT 变化 ≤0.5%。注意速度测试跑在首个 commit(量化在 pool 定容之后),配对服务器 KV pool 相同、峰值占用 63%/56%,作者声明提速不来自显存。

## 验证与回归风险

新增注册测试 `test/registered/amd/test_qwen35_gdn_online_mxfp4.py`(MI35x suite):形状 (4608,8192) 与 (8192,2048) x 行数 1/4/16/31/32/64/65/256,对 dequant FP32 参考相对误差 <1e-2,并验证第二次 `process_weights_after_loading()` 不重复量化;行数覆盖 32 行 scale-layout 分界与 64 行 kernel 切分两侧。端到端评测:GSM8K 5-shot NEXTN 0.9765→0.9742 / 无 MTP 0.9788→0.9818;GPQA Diamond 0.9343→0.9091(NEXTN)与 0.9343→0.9242(无 MTP);RULER niah 64k/128k 基本持平。风险点:纯 MXFP4 投影使 thinking 长度增加 10.7%(95% CI 6.1%-15.7%),GSM8K accuracy 0.9641→0.9597(McNemar p=0.0017),重视长推理时建议关闭;不支持在线权重更新(量化改变权重 shape/dtype),收权重的服务器需 `SGLANG_ROCM_GDN_ONLINE_MXFP4=0`;Triton 路径依赖 AITER pin 2bb549102b(含 Rocm/aiter#5998),旧 AITER 会全走 ASM(正确但未测速度);仅测了 Qwen3.8-2.4T TP8,Qwen3.5 checkpoint 与其他 TP 未测;CI Extra 与 AMD ROCm 10 显示失败标记(x)。

## 关联

同区域优化:`SGLANG_ROCM_K3_FUSE_KDA_INPROJ`(environ.py 相邻条目,K3 GDN in_proj 融合)、`SGLANG_USE_AITER` 生态、未决的 Hadamard-32 rotation 后续 PR(Rocm/aiter#6734 open,待 AITER pin 移动后跟进);引用 vllm#62750 的 thinking-length 检查方法。同批 PR #33599/#48437/#35770 同为 AMD 优化系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
