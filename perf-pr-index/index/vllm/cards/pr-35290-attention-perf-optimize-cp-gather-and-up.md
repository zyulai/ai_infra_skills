# PR #35290: [Attention][Perf] Optimize cp_gather_and_upconvert_fp8_kv_cache - DeepSeek-v3.2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-03-09 |
| 分类 | Quantization |
| Label | performance, ready, deepseek, nvidia |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/35290 |

## 一句话结论

重写 DeepSeek MLA 的 FP8 KV cache gather+升转 CUDA 内核:flat 网格每 warp 处理一个 token,内核延迟最高降约 100 倍,e2e TTFT 提升约 45%。

## 动机

`cp_gather_and_upconvert_fp8_kv_cache` 负责把 paged FP8 KV cache(每 token 656 字节:512 FP8 + 16 scale + 128 BF16 RoPE)gather 并升转为 576 列 BF16 workspace,是 DeepSeek-V3.2 长上下文 prefill 的热路径。旧实现 grid 为 `(batch, num_splits)`、每 block 576 线程,block 内串行 for 循环逐 token 处理,还带按 batch size 选 split 数的启发式;长 prefill 时单请求内核延迟高达毫秒级(98K tokens:3117 us)。

## 优化点

新内核改为 flat 网格(8 warps/block,共 ceil(total_tokens/8) 个 block),每个 warp 恰好处理一个输出 token,消除串行 token 循环与 split 启发式;用二分查找在 `workspace_starts` 前缀和里定位 token 所属请求,再经 block table 算物理地址。访存全部向量化:`int4` 读 16 字节 FP8、`scaled_vec_conversion<bf16_8_t, uint2>` 一次反量化 8 个元素(按 128 元素组的 scale)、`int4` 写 BF16、RoPE 段按 `int` 128-bit 拷贝。host 侧 kernel 签名简化:去掉 `seq_lens` 与 `head_dim` 参数,`total_tokens` 由 `dst.size(0)` 得出。

## 关键改动

```text
csrc/cache_kernels.cu: cp_gather_and_upconvert_fp8_kv_cache 重写(warp-per-token + 二分查找 + 向量化访存)
benchmarks/kernels/bench_cp_gather_fp8.py: 新增内核 microbenchmark(1/4/16 请求 x 8K-98K tokens)
tests/kernels/test_cp_gather_fp8.py: 新增正确性测试(乱序 block table、8K-128K 长序列)
```

## 性能声明

内核级(1-req,98304 tokens):3117.02 → 29.31 us;16384 tokens:245.64 → 6.06 us;16-reqs 98304 tokens:212.54 → 31.02 us。e2e:`vllm serve nvidia/DeepSeek-V3.2-NVFP4 -tp 4 --kv_cache_dtype fp8`,`vllm bench serve` input-len 96000 / output-len 1 / 32 prompts / 并发 1:mean TTFT 556.13 → 383.60 ms(PR 称约 45% TTFT 提升),total token throughput 172566.48 → 250134.01 tok/s。GSM8K 评测 accuracy 0.924 → 0.931。PR 未注明 e2e benchmark 所用 GPU 型号。

## 验证与回归风险

新增内核正确性测试:FP8 反量化段 atol 1e-3/rtol 1e-2,RoPE 段要求 bit-exact;覆盖单 token、跨 block、乱序物理块、8K-128K 长序列。风险集中:内核假定 DeepSeek MLA cache 布局(656B entry、128 元素 scale 分组)与 block_size 64(FlashMLA sparse 唯一支持值);host 接口签名变更(参数删减),`_custom_ops` 调用方需同步。

## 关联

同区域:DeepSeek-V3.2 / MLA sparse attention 与 FP8 KV cache 相关 PR(同分类 Quantization 下的 DeepSeek 系列);`concat_and_cache_ds_mla_kernel`(写侧,测试中模拟其布局)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
