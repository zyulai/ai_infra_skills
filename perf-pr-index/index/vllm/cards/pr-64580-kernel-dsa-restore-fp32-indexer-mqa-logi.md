# PR #64580: [Kernel][DSA] Restore FP32 indexer MQA logits for non-DeepSeek models and add `indexer_logits_dtype`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-11-03 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, ready, ci/build, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat 数字;pack diff 收录大部分文件,1 个文件因超长未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/64580 |

## 一句话结论

升级 DeepGEMM 并新增 `--attention-config.indexer_logits_dtype` 配置,恢复 GLM-5/5.3、HY-V4 等 DSA 模型的 FP32 indexer MQA logits,消除 BF16 造成的 top-k 选择误差。

## 动机

#61149 曾在 SM100 上强制 BF16 indexer logits,原因是当时 DeepGEMM 版本移除了 `logits_dtype` 参数。但 BF16 logits 有实际精度代价:在 GB300 上对照 FP64 参考实现,BF16 logits 会让 top-2048 选中 token 有 0.4–0.75% 偏差,而 FP32 logits 能精确选中。对依赖 top-k 稀疏选择准确性的模型(GLM-5、GLM-5.3、HY-V4)这是精度风险,需要把 FP32 输出找回来,同时保留 DeepSeek 系列在 main 上的 BF16 行为与性能。

## 优化点

两步:一是把 DeepGEMM 从 `5cdb145` bump 到 `ae4fd25`,恢复 `fp8_fp4_(paged_)mqa_logits` 的 FP32 输出(以 `logits_dtype` 参数形式),顺带修复 mega_mhc 的 CUDA-graph replay barrier;二是在 vLLM 侧新增 `--attention-config.indexer_logits_dtype={auto,float32,bfloat16}`,`auto` 按架构解析:GLM-5/5.3 与 HY-V4 恢复 FP32,DeepSeek-V3.2/V4/V4.1 经 config hook 保持 SM100 上的 BF16(默认以架构名判定,GLM-5 复用 DeepSeek-V3.2 模型类所以按架构名而非层判定);无 hook 的 indexer 模型 `auto` 解析为 FP32。SM90/SM120 默认行为不变(始终 FP32)。`mqa_logits_mx_inputs` 按 `logits_dtype` 转换 head weights:FP32 权重走 FP32 归约,BF16 权重走 BF16 归约并直接输出 BF16;FP32 decode logits 免去原 SM100 路径的 `as_fp32_logits` upcast。sparse MQA-logits 路径(`indexer_sparse_logits`)不变,始终 BF16。

## 关键改动

```text
cmake/external_projects/deepgemm.cmake: DeepGEMM pin 5cdb145 → ae4fd25
vllm/config/attention.py: 新增 indexer_logits_dtype 选项与 resolve_indexer_logits_dtype
vllm/model_executor/models/config.py: DeepSeek 系 config hook 设 SM100 BF16 默认
vllm/utils/deep_gemm.py: mqa_logits_mx_inputs 与两个 MQA-logits 封装透传 logits_dtype
vllm/model_executor/layers/sparse_attn_indexer.py: 全路径透传 logits_dtype;decode chunk 预算按 itemsize 计算
```

关键片段(deep_gemm.py):

```python
def mqa_logits_mx_inputs(q, q_scale, weights, logits_dtype):
    ...
    return q_scale, weights.to(logits_dtype)
```

## 性能声明

对照 FP64 参考的随机输入测试(64 heads、head dim 128,GB300,早前 pin 9918052):
- 精度:FP32 相对误差约 6.4e-8、top-2048 overlap 1.0000;BF16 约 4.8e-3、overlap 0.9925–0.9959。
- 耗时(prefill):BF16 快 3–7%(2048×16K:217→210 µs;4096×64K:1770→1645 µs)。
- 耗时(decode,eager 非 CUDA-graph):FP32 54/49 µs 对 BF16+upcast 93/92 µs(b=64 ctx16K、b=8 ctx64K);差距主要来自 BF16 路径所需的 upcast launch,DeepSelect+BF16 未计时。
- E2E 精度:GLM-5.3-Flash FP8、TP4、4×GB200、无 MTP,gsm8k 5-shot 全 1319 题:`auto`(FP32)93.10/93.03 vs `bfloat16` 93.18/93.03,差异在噪声内(GSM8K prompt 短,top-k 不会丢 token);长上下文 MRCR 验证仍 pending。

## 验证与回归风险

`test_deepgemm_fp8_mqa_logits`、`test_deepgemm_fp8_fp4_paged_mqa_logits` 参数化到两种 dtype 并断言输出 dtype;新增 CPU 单测 `tests/models/test_indexer_logits_dtype_config.py`(30 passed,覆盖各架构 SM100 上/下的 auto 解析与显式覆盖)。风险面:影响所有 DeepGEMM dense MQA-logits 用户(DSA sparse indexer、GLM-5.3 kpool indexer、DSv4.1 specialized decode indexer);与开放 PR #62170/#62172/#64138 的 BF16 优化路径交互 —— FP32 logits 下这些优化不生效但也不破坏;decode logits 内存预算按 dtype itemsize 重算。

## 关联

前序:#61149(SM100 强制 BF16 的起因)。关联 BF16 路径优化:#62170、#62172、#64138(后两者已在索引中)。同区域:#56254(DeepGEMM sparse MQA logits 接入)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
