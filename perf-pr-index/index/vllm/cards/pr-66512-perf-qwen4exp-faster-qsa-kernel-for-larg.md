# PR #66512: [Perf][Qwen4Exp] Faster QSA kernel for large fp8 batches on Blackwell

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-11-09 |
| 分类 | Speculative Decoding(tier: major) |
| Label | ready, ci/build, qwen |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66512 |

## 一句话结论

Qwen4Exp QSA fp8 稀疏注意力在 Blackwell 大 batch 换单 pass kernel:decode/verify 2.63x,TPOT −4.4%。

## 动机

原 QSA sparse attention 在 fp8 KV cache 下走 split-K kernel;400-row MTP decode/verify shape 上最优 split-K 配置只达 1.18x,把两项改进折进 split-K 也只 2.46x,且 split-K 有 94 个 register spill、每 tile 360 次字节级 shared-memory load。

## 优化点

新增 leaf module `vllm/models/qwen4_exp/nvidia/ops/qsa_fp8.py`:(1) slot pass 把每 row 的 selected tokens 经 page table 翻译一次、由所有 KV heads 共享,attention loop 的 K/V 地址提前可知、可 prefetch;(2) 单 pass attention kernel 将 K/V 作为 int16 pairs of e4m3 codes gather、精确转成 fp16 halves 喂 MMA,byte-wise shared-memory loads 从 360 降到 40 per tile,允许 4 warps + 5 级 `cp.async` pipeline。QK 保持 bf16、PV 用 fp16(e4m3→bf16/fp16 精确),每个 QK/PV 乘积与 split-K 相同,仅 fp32 求和分组不同。门控 `use_fp8_decode_kernel`:fp8-e4m3 cache、sm_100/sm_103(B200/B300)、rows × KV heads per rank > 64——低于它 split-K 快 0.7–0.9x;高于它除 TP4 恰低于一个 full wave(~140 programs)约 0.96x 外全胜。bf16 cache、更小 batch、其他 GPU 保持 split-K 路径 bitwise 不变。

## 关键改动

```text
vllm/models/qwen4_exp/nvidia/ops/qsa_fp8.py(新增): slot pass + 单 pass decode kernel + use_fp8_decode_kernel 门控 + warmup
vllm/models/qwen4_exp/nvidia/ops/qsa.py: qsa_sparse_paged_attention 分派到 fp8 kernel 或 split-K(_qsa_sparse_paged_splitk)
tests/models/qwen4_exp/test_qsa_reference.py: 新增 test_qsa_fp8_kernel_correctness
.buildkite/test_areas/models_basic.yaml: B200 CI step 改名 qwen4-exp-kernels-b200 并依赖 qsa.py/qsa_fp8.py
```

```python
return (
    cache_dtype == torch.float8_e4m3fn
    and base_programs > _DECODE_MIN_PROGRAMS
    and _is_sm100_family()
)
```

## 性能声明

- Kernel(GB200,CUDA graphs,2048 selected tokens/row,24 query / 2 KV heads):decode/verify 400 rows 396→151 µs(2.63x);1600 rows 1153→525 µs(2.20x);prefill 4096–8192 rows 1.37–1.42x。
- 端到端(one 4×GB200 node,TP4,MTP3,fp8 KV + indexer cache;`vllm bench serve`,random dataset,8192-token cached prefix,100 output tokens,`--ignore-eos`,concurrency 512,3000 prompts;每臂 2 runs):median TPOT 54.64→52.22 ms(−4.4%);output throughput 7695→7975 tok/s(+3.6%);MTP acceptance length 2.186→2.217(+1.4%)。
- 准确度(同 serving 设置):GSM8K(5-shot,temp 0,n=2638)96.78%→96.70%;GPQA-Diamond(thinking,temp 0,n=198)72.2%→75.3%;AIME25(thinking,temp 0.7,4 epochs,n=120)90.8%→85.8%(作者注明 30 题在该温度下属采样噪声)。

## 验证与回归风险

`pytest tests/models/qwen4_exp/test_qsa_reference.py`:82 passed(GB200,约 40 s)。新增 `test_qsa_fp8_kernel_correctness`:MTP-verify 形状 fp8 batch 在全部 TP head layout(24/2、12/1、6/1、3/1)对照 dense reference(rtol=atol=2e-2),覆盖 masked tails 与 CUDA-graph padding row(必须保持零),并断言 gate 触发。风险:新路径非 bitwise(fp32 求和顺序);仅 GB200 实测,sm_103 未验证;slot pass 每次调用增加一次小 launch 与 `rows × selection width` 的 int32 workspace。

## 关联

无(正文注明原始 kernel 设计与原型出自 @abmfy,第一 commit 的 author of record)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
