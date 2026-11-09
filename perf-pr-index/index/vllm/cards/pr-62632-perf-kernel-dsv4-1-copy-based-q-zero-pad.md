# PR #62632: [Perf][Kernel][DSv4.1] Copy-based Q zero-pad for large eager mega-attention steps

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @roborluo |
| 合并 | 2026-11-02 |
| 分类 | Attention / Kernel(major) |
| Label | ready, deepseek, verified, DSv4.1 |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/62632 |

## 一句话结论

DeepSeek-V4.1 mega-attention 大 eager 步(≥2048 token,TP2)把 Q zero-pad 从融合 insert op 中拆出,用连续 Triton copy 实现,每层 Q pad + KV insert 快 1.9–2.15×。

## 动机

Mega attention 自己做 Q RoPE 并以 chunk-interleaved 布局读 Q:每 token 32 个 `[heads, 16]` chunk,padding heads 位于每个 chunk 尾部。V4.1 层在 `wq_b` 前已对 `qr` 归一化,所以 `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 的 Q 半部分是纯 zero-pad。32 live heads pad 到 64 时,Q 半部分每 token 读 32 KiB、写 64 KiB,而共享同一次 launch 的 KV 半部分只有约 1.8 KiB——大步长下 op 时间几乎全是 Q pad。且 fused op 按 warp 逐 (token, head) slot 写 Q,16 元素片段相距 `padded_heads * 16` 个元素,访存极不连续。

## 优化点

符合条件的层(读 interleaved Q、32/64 heads、`max_num_batched_tokens ≥ 2048`)在初始化时置 `_pads_q_by_copy` 并注册 `_pad_interleaved_q` 的 JIT warmup。每步派发在 `_fused_qnorm_rope_kv_insert`:步长 ≥ `_Q_PAD_COPY_MIN_TOKENS`(2048,由 B300 上 64–16384 token 扫描确定)且当前 stream 非 capturing 时,fused op 以既有 `q_head_padded=0`(KV-only)模式运行,`_pad_interleaved_q` 将 [N, 32, 512] 的 Q 连续拷贝进 [N, 64, 512](每 (token, 16-dim chunk) 一行:512 个 live 元素 + 512 个 +0.0)。捕获的 CUDA graph、小于阈值的步、TP1/TP4/TP8 shard、其他 attention backend 与平台完全不变;无新 flag/env/config。输出 bit-identical(NaN-free Q 下 padded Q 与 sliding-window KV cache 字节均与 fused launch 相同)。

## 关键改动

```text
vllm/models/deepseek_v41/attention.py: _pad_interleaved_q Triton copy kernel + _pads_q_by_copy 派发
tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py: 新增 KV-only+copy vs fused 等价测试
tests/kernels/test_compressor_kv_cache.py: stand-in 层声明 _pads_q_by_copy=False
vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py: 注释更新
```

## 性能声明

每层 Q pad + KV insert kernel 时间(bf16,32 live/64 padded heads,`bench_q_pad_copy.py`,CUDA graph replay + CUDA events 取中位,两次 run 差 < 0.1%):2048 token 1.93×、4096 2.06×、8192 2.10×、16384 2.15×;512 token 1.61×、1024 1.87×(低于阈值,不派发)。发行配置 `BLOCK_ROWS=4`、4 warps 为 12 个扫过的配置中最快。serving 侧以 torch profiler 确认 8192-token prefill 走 `_pad_interleaved_q_kernel`(TP2、FLASHMLA_MEGA_ATTN_DSV41、Engram CPU offload);PR 未给出端到端吞吐数字。

## 验证与回归风险

验证:新单测在 17/1024/2049 token(NUM_TOKEN_CUTOFF 两侧)× 0/5 DP-padding 行下断言 cache 字节与 padded-Q bits 完全一致(rtol=0, atol=0,copy 写入 NaN 填充 buffer);整文件 221 passed;PCP/SWA cache 测试 2 passed;`VLLM_GPU_SYNC_CHECK=error` 下完整 GSM8K 无 host sync;lint/mypy 通过。风险:eligible 步从一次 launch 变两次(额外 host 端 launch 开销,由大步隐藏);仅 32-of-64 shard(TP2)符合 2 的幂 chunk 宽度要求;CI 无 TP2 mega attention 任务,serving 检查为手动;`source_file_dependencies` 未列 `attention.py`,后续仅改 copy 不会重跑测试(作者已提出可补)。

## 关联

属于 #57448 的一部分,基于 #56935(FlashMLA mega attention 与被绕过的 fused interleaved pad)和 #61371(V4.1 KV record);相关:#58872(token-sharded fused_wqa_wkv)、#55654 与 #60076(KV-only insert ops,均不重复)、#59857(prefill context parallelism 走 FlashMLA 层,不经 copy)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
