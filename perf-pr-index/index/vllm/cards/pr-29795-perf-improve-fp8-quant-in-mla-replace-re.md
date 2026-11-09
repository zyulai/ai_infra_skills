# PR #29795: [Perf] Improve fp8 quant in mla; replace ReduceSum with ReduceScatterSum

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @IwakuraRein |
| 合并 | 2025-12-08 |
| 分类 | Quantization(major) |
| Label | ready, v1, nvidia |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 MLA 公共路径与 CUDA communicator 两个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/29795 |

## 一句话结论

MLA fp8 注意力把 decode q 的两次 fp8 量化合并为一次并消除 torch.cat,reduce_scatterv 等尺寸时改走 ReduceScatterSum;预计 decode +5%。

## 动机

fp8 attention 下 decode_ql_nope 与 decode_q_pe 各做一次 scaled_fp8_quant(两次量化 kernel),flashinfer mla 后端还要 torch.cat 拼接,开销明显(profile 图见 PR);reduce_scatterv 在各 rank 尺寸完全相同时仍走 reduce_scatterv,而非更高效的 reduce_scatter。

## 优化点

v1 MLA common.py decode 路径:fp8_attention 时先 torch.empty 出 decode_q0 buffer,用 copy_ 分别写入 ql_nope 与 q_pe(正文说明 torch.cat 引入显著开销故用 empty+copy),把 view 成 2D 的整块 q 交给一次 scaled_fp8_quant,再 view 回 (B, N, L+P) 3D;非 fp8 路径保持 (decode_ql_nope, decode_q_pe) 元组。cuda_communicator.reduce_scatterv:sizes 非空且所有元素相等时改走 pynccl reduce_scatter(ReduceScatterSum),不再调用 varint 版 reduce_scatterv;尺寸不均时行为不变。

## 关键改动

```text
vllm/v1/attention/backends/mla/common.py: decode fp8 q 合并为单次量化,
  用 empty+copy_ 取代 cat/两次 quant
vllm/distributed/device_communicators/cuda_communicator.py:
  reduce_scatterv 在 sizes 全相等时改走 reduce_scatter
```

## 性能声明

5% projected improvement in decode when using flashinfer mla backend,DeepSeek R1 FP4 serving on 4 GB200 with DEP4(投影值,PR 未给出实测 A/B 数字)。精度验证:DeepSeek R1 0528 + 8 H200、VLLM_ATTENTION_BACKEND=FLASHMLA,gsm8k 20-shot flexible 0.9530 / strict 0.9522;DeepSeek R1 0528 FP4 v2 + 4 GB200、FLASHINFER_MLA,0.9538 / 0.9522。服务参数含 --kv-cache-dtype fp8、async-scheduling、FULL_DECODE_ONLY。

## 验证与回归风险

lm-eval gsm8k 精度对照如上(两个后端、两种 checkpoint);pack diff 未见新增单元测试。风险:reduce_scatterv 的等尺寸分支改变所有 CUDA TP 通信路径的选择;MLA decode fp8 路径新增 shape 断言(ql_nope 与 q_pe 的前两维必须一致),DCP(fp8 不支持)与后端形状约定需保持。

## 关联

无(pack 未列出关联 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
