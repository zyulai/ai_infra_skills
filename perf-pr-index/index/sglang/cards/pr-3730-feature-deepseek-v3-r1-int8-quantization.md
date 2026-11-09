# PR #3730: Feature DeepSeek V3/R1 INT8 Quantization (block-wise)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @laixinn |
| 合并 | 2025-02-24 |
| 分类 | Quantization |
| Label | 无 |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 中可见新增 blockwise_int8.py、int8_utils.py 等,另有 3 个文件因超长未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/3730 |

## 一句话结论

为 DeepSeek V3/R1 增加 block-wise INT8 量化支持,A100*32 TP 下输出吞吐提升最高 33%,且精度无损失。

## 动机

PR 之前 DeepSeek V3/R1 只有 FP8 block-wise 量化路径,而 FP8 在许多硬件平台上并不友好。INT8 是大多数硬件普遍原生支持的数据类型,以 block-wise 方式做 INT8 量化可以在不支持 FP8 的环境(如 A100)上以更低的部署成本获得吞吐收益。为此 PR fork 了 FP8 的实现,并配套发布了 INT8 权重 checkpoints(meituan/DeepSeek-R1-Block-INT8)。

## 优化点

核心是把 FP8 block-wise 的 quant 方法复制为 INT8 版本,并在 Triton kernel 层打通 w8a8 INT8 路径。新增 `BlockInt8Config` / `BlockInt8LinearMethod` / `BlockInt8MoEMethod`,对 Linear 与 FusedMoE 分别注册量化方法;新增 `per_token_group_quant_int8` Triton kernel,按 group 对激活做 per-token-group 量化(取 absmax/127 为 scale);新增 `_w8a8_block_int8_matmul`,在 K 维循环中加载 per-block 的 A/B scale 并做反量化累加。FusedMoE 的 Triton kernel 增加 `use_int8_w8a8` 分支,与 fp8_w8a8 共享 block-wise scale 索引逻辑;kernel 配置支持从 configs 目录按 `dtype=int8_w8a8,block_shape=[...]` 读取调优文件。权重加载路径(deepseek_v2.py)对 INT8 checkpoint 的 block 量化 scale 做反量化处理。

## 关键改动

```text
python/sglang/srt/layers/quantization/blockwise_int8.py: 新增 BlockInt8Config/BlockInt8LinearMethod/BlockInt8MoEMethod
python/sglang/srt/layers/quantization/int8_kernel.py: 新增 per_token_group_quant_int8 与 w8a8_block_int8_matmul Triton kernel
python/sglang/srt/layers/quantization/int8_utils.py: 新增 apply_w8a8_block_int8_linear、input_to_int8、block_dequant
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py: fused_moe kernel 增加 use_int8_w8a8 分支
python/sglang/srt/models/deepseek_v2.py: 加载 INT8 checkpoint 时做 block 反量化
```

关键 kernel 量化逻辑:

```python
y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
_absmax = tl.maximum(tl.max(tl.abs(y)), eps)
y_s = _absmax / int8_max
y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)
```

## 性能声明

PR 正文给出 A100*32、TP、exclusive 部署下的结果(GSM8K 1400 题、MMLU、bench_serving ShareGPT random 128/128、QPS=128、1000 prompts):

| 模型 | 配置 | GSM8K | MMLU | 输出吞吐 (QPS=128) |
|---|---|---|---|---|
| BF16 R1 | A100*32 | 95.5 | 87.1 | 3342.29 |
| INT8 R1 | A100*16x2 | 95.8 | 87.1 | 4450.02 (+33%) |

正文注明:INT8 只需 16 张 A100 即可部署,故按两台 16 卡服务器估算总吞吐(单机性能乘 2);开启 `--enable-torch-compile --torch-compile-max-bs 8`,且当时的复现命令关闭了 flashinfer(`SGLANG_IS_FLASHINFER_AVAILABLE=false`,flashinfer >= 0.2.2.post1 可去掉)。

## 验证与回归风险

PR 未见新增单元测试(diff 中无 test 文件);正文引用了 accuracy 复现脚本(gsm8k/mmlu bench_sglang)。风险点:改动触碰了共用的 fused_moe Triton kernel 与 invoke 路径,`use_int8_w8a8` 参数贯穿所有 fp8/int8 分支,其他使用 fused_moe 的模型(如 FP8 DeepSeek)依赖该路径行为不变;`fused_experts_impl` 中 padded_size 判断从 `not use_fp8_w8a8 or block_shape is not None` 改为同时考虑 int8,逻辑变化需关注 FP8 padding 行为是否回归。仅支持 int8-serialized checkpoint + dynamic activation scheme,不支持静态 scale。

## 关联

同属 DeepSeek 量化的 FP8 block-wise 前序实现(fp8.py / per_token_group_quant_fp8),本 PR 是其 INT8 fork;后续 DeepSeek 系列量化优化 PR 可参考索引中 Quantization 分类其他卡片。pack 素材未列出具体关联编号,此处不展开。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
