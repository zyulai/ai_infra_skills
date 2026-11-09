# PR #3888: [Feature] DeepSeek V3/R1 INT8 Quantization (channel-wise)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HandH1998 |
| 合并 | 2025-03-07 |
| 分类 | Quantization(major) |
| Label | (none) |
| Diff | 素材未注明 diffstat(diff 片段可见 5 个文件,含 1 个新增测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/3888 |

## 一句话结论

为 DeepSeek V3/R1 引入 channel-wise INT8(W8A8)MoE 量化,A100 集群吞吐最高 +50.6% 且无精度损失。

## 动机

为 DeepSeek V3/R1 提供 channel-wise INT8 量化支持。PR 正文给出的理由很直接:INT8 是对大多数硬件平台都友好的类型(相比 FP8 有更广的硬件覆盖)。配套发布了对齐权重的 meituan/DeepSeek-R1-Channel-INT8(与 #3730 同一系列工作)。

## 优化点

在 fused MoE Triton 内核中新增 `use_int8_w8a8` 的 channel-wise 分支:权重按列(per-column)加载 scale,激活按 token 动态量化(新调用 `per_token_quant_int8`),在内核内完成 `a_scale * b_scale` 反量化;同时把该分支的累加从 `tl.dot(a, b, acc=accumulator)` 改为 `accumulator += tl.dot(a, b)`,diff 注明是为了修复 shared memory 溢出问题。量化配置侧在 `w8a8_int8.py` 新增 `W8A8Int8MoEMethod` 并注册到 FusedMoE 层,负责创建 int8 专家权重与 per-channel scale、在 `apply` 中接 `fused_experts`;Linear 层则沿用已有的 CUTLASS `int8_scaled_mm`(W8A8Int8LinearMethod)。`deepseek_v2.py` 权重加载对 channel-wise int8 的 `kv_b_proj` 增加 `w * weight_scale` 的反量化路径(区别于 block-wise 的 `int8_block_dequant`)。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py: 内核支持 int8 w8a8 channel-wise scale 加载;激活改 per-token 量化;int8 分支累加方式调整
python/sglang/srt/layers/quantization/w8a8_int8.py: 新增 W8A8Int8MoEMethod(权重/scale 创建、apply 接 fused_experts)
python/sglang/srt/models/deepseek_v2.py: channel-wise int8 的 kv_b_proj 反量化加载路径
test/srt/test_int8_kernel.py: 新增;fused moe int8 内核 vs 原生 torch 参考实现对比测试
test/srt/run_suite.py: 将 test_int8_kernel.py 加入 per-commit 套件
```

```python
# channel-wise
else:
    # Load per-column scale for weights
    b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse
                    + offs_bn[None, :] * stride_bsn)
    b_scale = tl.load(b_scale_ptrs)
    # Load per-token scale for activations
    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
    a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
```

## 性能声明

基准与 #3730 相同:DeepSeek-R1,A100 集群独占 TP,QPS=128、ShareGPT random 128 输入/128 输出、1000 prompts。

| 配置 | GSM8K | MMLU | 输出吞吐 |
| --- | --- | --- | --- |
| BF16 R1(A100 TP32) | 95.5 | 87.1 | 3342.29 |
| Channel-INT8(FP32 累加器,(A100 TP16)×2) | 95.8(+0.3) | 87.2(+0.1) | 5035.82(+50.6%) |
| Channel-INT8(INT32 累加器,(A100 TP16)×2) | 95.1(-0.4) | 86.9(-0.2) | 5957.48(+78.2%) |

PR 注明:INT8 部署 16 卡 A100 即可,故公平对比按两台 TP16 服务器估算总吞吐(单机性能 ×2,未做负载均衡)。INT32 累加器版本未合入,在分支 `dsv3-w8a8-channel-int8-opt` 提供。

## 验证与回归风险

新增 `test_int8_kernel.py`(需 CUDA)对 fused MoE int8 路径做 M/N/K/topk/dtype 参数化验证,与原生 torch per-column 参考实现对比相对误差 < 0.05,并加入 per-commit CI 套件。风险面:`fused_moe_kernel` 的 int8 分支累加行为改变(为避开 shared memory 溢出),影响所有走 `use_int8_w8a8` 的 MoE 调用;`deepseek_v2.py` 对 int8 权重加载逻辑分支重构,影响 DeepSeek V2/V3 家族在 block-wise/channel-wise 两种 int8 下的权重加载正确性;INT32 累加器的高吞吐变体未合入主线。

## 关联

前序:PR #3730(同系列 DeepSeek 量化工作,基准方法沿用)。同分类:DeepSeek 量化相关后续 PR(素材未列出编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
