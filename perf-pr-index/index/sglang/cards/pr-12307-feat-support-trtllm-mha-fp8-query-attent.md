# PR #12307: feat: support trtllm_mha FP8 query attention kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @elvischenv |
| 合并 | 2025-10-31 |
| 分类 | Quantization(major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供统计);diff 实际仅 1 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/12307 |

## 一句话结论

trtllm_mha 在 KV cache 为 FP8 时把 query 也转 FP8,启用 FP8-qkv/BF16-out kernel,gpt-oss-120b 吞吐 +5%。

## 动机

TRTLLM_MHA 后端此前已支持 FP8-qkv、BF16-out 的 attention kernel,相比原有的 BF16-q + FP8-kv 组合 kernel 性能更好(关联讨论 #9782)。此前 SGLang 的 q 一直是 BF16 进 kernel,无法用上这条更快路径;本 PR 在 KV cache 类型同为 FP8 时把 query 转成 FP8,以匹配 kernel 的 FP8-qkv 输入要求。

## 优化点

改动极小而机制直接:`trtllm_mha_backend.py` 的 `forward_decode` 与 `forward_extend` 中,当 `self.data_type == torch.float8_e4m3fn`(即 KV cache 为 FP8)时,对 q 做一次 `q.to(torch.float8_e4m3fn)` 类型转换,随后调用 TRT-LLM 的 attention 接口并新增传参 `out_dtype=self.q_data_type`(即 model_runner 的原 dtype,BF16),保证输出仍按模型精度返回。这样 decode 与 prefill 两条路径都走 FP8-qkv + BF16-out 的 kernel 变体,省去 kernel 内部对 BF16 q 的处理路径。

## 关键改动

```text
python/sglang/srt/layers/attention/trtllm_mha_backend.py: q 转 FP8 + 指定 out_dtype(decode/extend 两处)
```

```python
if self.data_type == torch.float8_e4m3fn:
    q = q.to(torch.float8_e4m3fn)
q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
...
sinks=attention_sink,
out_dtype=self.q_data_type,  # model_runner.dtype
```

## 性能声明

bench_serving,openai/gpt-oss-120b,random 输入/输出各 1024,range-ratio 1,max-concurrency 512,2560 请求(PR 未注明 GPU 型号):

| 指标 | main | 本 PR |
| --- | ---: | ---: |
| Benchmark duration (s) | 161.57 | 153.18 |
| Total token throughput (tok/s) | 32450.49 | 34227.89 |
| Mean ITL (ms) | 28.98 | 27.24 |
| Mean TTFT (ms) | 2632.67 | 2720.10 |

作者标注 "5% perf improvement"(总吞吐 +5.5%)。

## 验证与回归风险

gsm8k(lm_eval,openai/gpt-oss-120b,limit 0.5):本 PR flexible-extract 0.8682 / strict-match 0.6091,main 0.8394 / 0.6318——flexible-extract 高于 main,strict-match 略低,PR 未对差异作说明。回归风险:仅当 KV cache 为 FP8(e4m3)时 q 才被量化,引入一次额外的 q 转换 kernel 与显存拷贝;FP8 q 的舍入可能影响所有走 trtllm_mha + FP8 KV cache 模型的 attention 输出精度(注意力分母/softmax 在 kernel 内以 FP8 q 计算)。TTFT 在该基准中略升,作者未解释。

## 关联

前序:#9782(trtllm_mha FP8-qkv kernel 能力的关联讨论)。同区域无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
