# PR #5113: Support MHA with chunked prefix cache for DeepSeek chunked prefill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2025-04-16 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | high priority, performance, deepseek |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 可用,1 个文件因超长被截断) |
| 链接 | https://github.com/sgl-project/sglang/pull/5113 |

## 一句话结论

DeepSeek 长前缀 chunked prefill 时把 MLA 吸收策略换成"分块 KV 的 MHA + 状态合并",32k 输入下 prefill 吞吐从 7253 提升到 9937 tok/s(8xH200)。

## 动机

此前 MLA 在长前缀(如 32k input)下很慢(issue #5031):profiling 显示 attention kernel 运行时间是主要瓶颈。chunked prefill 期间走的是 absorb(权重吸收)路径,但 q 中 token 很多时 MLA-with-absorption 计算强度过大;直接改用 MHA 计算上高效,可物化 KV head 又容易 OOM。方案是把长前缀 KV cache 切成多个 chunk,逐块做 MHA,把每块的输出与 log-sum-exp (LSE) 状态合并成单一累积输出——既享受 MHA 的计算友好,又避免 OOM。设计参考 vllm 的 `mla/common.py` 与 flashinfer 的 recursive attention 教程。

## 优化点

新增第三种 attention 分派方式 `MHA_CHUNKED_KV`,与既有 `MHA` / `MLA`(absorb)并列:`dispatch_attn_forward_method` 在 fa3 backend、extend 模式、prefix 长度总和超过阈值(8192 token)时触发。执行时先对 extend 部分做一次普通 MHA 得到初始 output/LSE,再对每个 prefix chunk:用预计算的 chunk kv indices 从 MLA latent pool 取出 latent cache,经 `kv_b_proj` 还原成 K/V,跑 `flash_attn_varlen_func`(causal=False,返回 LSE),然后用 sgl-kernel 的 `merge_state_v2` 把 output/LSE 累积合并。chunk 元数据(starts/seq_lens/cu_seq_lens/kv_indices)在 `ForwardBatch` 上准备一次、跨层复用,chunk 大小按 `128*1024` 总容量除以 batch size 切分;GPU/CPU 各算一遍 chunk lens 以避免 stream sync。KV indices 用专门的 Triton kernel `create_chunked_prefix_cache_kv_indices` 批量生成。短输入(如 128/256 token)会引入开销,因此仅在超过阈值时激活。该功能对 DeepSeek 模型默认开启(page_size > 1 或非 MLA backend 自动禁用),可用 `--disable-chunked-prefix-cache` 关闭;当前仅支持 FA3 backend。

## 关键改动

```text
python/sglang/srt/models/deepseek_v2.py: 新增 AttnForwardMethod.MHA_CHUNKED_KV 分派、forward_normal_chunked_kv 与逐 chunk 合并循环
python/sglang/srt/layers/attention/flashattention_backend.py: forward_extend 增加 attend/not-attend prefix cache 两条 varlen MHA 分支
python/sglang/srt/model_executor/forward_batch_info.py: 新增 prefix chunk 元数据字段、prepare_chunked_prefix_cache_info 与 Triton kv-indices kernel
python/sglang/srt/model_executor/model_runner.py: 非 MLA 或 page_size>1 时自动禁用 chunked prefix cache
python/sglang/srt/server_args.py: 新增 --disable-chunked-prefix-cache 开关
python/sglang/test/attention/test_prefix_chunk_info.py: 新增 chunk 切分逻辑与 Triton kernel 的单元测试
```

关键片段(deepseek_v2.py,逐 chunk 合并):

```python
output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
lse = torch.transpose(lse, 0, 1).contiguous()
tmp_output = torch.empty_like(accum_output)
tmp_lse = torch.empty_like(accum_lse)
merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
accum_output, accum_lse = tmp_output, tmp_lse
```

## 性能声明

基准均在 8xH200、DeepSeek-V3、`--attention-backend fa3 --disable-radix-cache --max-prefill-tokens 32768`、DeepGemm 关闭(注明开启并预热 DeepGemm 可获更高吞吐):

- 32000-100(random-input 32000/output 100,120 prompts,request-rate 8):Prefill 吞吐 7253.07 -> 9937.69 tok/s,Decode 22.67 -> 31.06 tok/s;Mean TTFT 259006.92 -> 185310.57 ms,Mean TPOT 280.73 -> 214.13 ms
- 4096-512(512 prompts):Prefill 5751.74 -> 6157.18 tok/s;Mean TTFT 164542.75 -> 158411.01 ms
- 2000-100(120 prompts,request-rate 2):Mean TTFT 2773.21 -> 2596.57 ms,Mean TPOT 79.69 -> 69.06 ms
- 并发基准(input=32000, output=100, 并发 1-32,input 32000/output 100,无 DeepGemm):并发 32 时 input throughput 7238.651 -> 10008.304 tok/s,mean TTFT 97750.335 -> 70225.980 ms
- Profiling(chunked prefill,input 32000,chunked_prefill_length 8192):原 absorb 策略下 flash attention kernel 单次最长约 8ms,改后由多个小 kernel 组成,总计约 3ms

准确性:GSM8K(8-shot, 1319 题)0.948,Invalid 0.000;MMLU 0.876。

## 验证与回归风险

新增 `test_prefix_chunk_info.py` 覆盖三种 chunk 切分场景(等长/不等长/不规则前缀)与 Triton kv-indices kernel 的正确性;`test/srt/test_fa3.py` 有相关改动。回归面:仅 FA3 backend 且限定 DeepSeek(M LATokenToKVPool assert);非 MLA backend 与 page_size > 1 强制禁用;target_verify/draft_extend 模式不走该路径;短前缀(<8192)保持原 absorb 路径以免性能回退。PR 自述 TODO 中"更好的 chunk 切分方式"与"更好的触发条件"(#5624)留待后续。

## 关联

前序 issue #5032 同主题问题由 #5031 提出;TODO 提及触发条件优化在 #5624;`flash_attn_varlen_func` 合入 sgl-kernel 的 #5315。同区域为 DeepSeek MLA prefill 优化(同分类 Overlap / Scheduler 下其他 PR 编号 pack 未列出)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
