# PR #30514: [DSA] Integrate Q8KV8 FP8 Sparse MLA Prefill into the DSA Backend (DeepSeek-V3.2)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang |
| 合并 | 2026-07-19 |
| 分类 | Quantization (tier: major) |
| Label | documentation, quant, deepseek, run-ci, jit-kernel, bypass-fastfail |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,7 个文件,另有 5 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/30514 |

## 一句话结论

把 SM90 Q8KV8 FP8 sparse-MLA prefill kernel 集成进 DSA 后端,DeepSeek-V3.2 prefill 全程原生 FP8,H200 单请求延迟提升 4.6-7.7%、饱和吞吐提升 1.8-2.7%。

## 动机

PR-1(#25751)已交付独立的 Q8KV8 sparse prefill kernel,但 DSA prefill 仍走 bf16 路径:fp8 KV 需要先 dequant 回 bf16 再喂给 bf16 kernel,产生 paged-fp8->bf16 的往返、KV 侧内存流量翻倍,且用不上 FP8 tensor core 的 2x 吞吐。把 kernel 接入 `dsa_backend.py` 才能转化为端到端 serving 收益(roadmap #25746)。

## 优化点

四个层面:(1) 集成:新增 `flashmla_sparse_q8` 作为一等 DSA prefill 选项(`_forward_flashmla_sparse_q8kv8` 路由、RAGGED gate、启动期 FP8+SM90 fail-fast 校验、拒绝作为 decode backend),decode 保持 `flashmla_kv` 不变;prefill context <= index_topk(2048)时走 dense one-shot 路径,两后端一致。(2) 原生 FP8 KV:新增 `gather_dequant_requant_fp8_paged` 单个 Triton kernel 完成 paged fp8 KV 的 gather + per-group dequant + per-tensor requant(nope 512B fp8 + 4 组 fp32 scale + 64B rope bf16 -> 576B fp8),消除 fp8->bf16->fp8 往返与中间 bf16 缓冲;per-tensor scale=1 恒等,依赖真实 DeepSeek-V3 数值幅度。(3) 融合 q-prep:`concat_and_cast_q_fp8_pad` 把 concat(q_nope, q_rope) 与 bf16->fp8 cast 融合(bit-exact,比两条 strided `copy_` 快约 3.7x,贡献约 +3.5% c=1 吞吐),仅支持 2 的幂头数并带回落。(4) e2e 暴露的 kernel 并发/正确性修复:full-topk dispatch 把 -1 哨兵映射到互不相同的零 pad 行(统一控制流,修复长上下文下 DP cross-rank collective 死锁;pad 槽必须在 softmax 掩成 -INF,否则 few-valid 行输出被压低最多约 2048x)、`is_kv_valid`/`sM` 按迭代奇偶双缓冲(barrier 复用竞态)、named barrier 改 arrive_and_wait 完全汇合、WG1 peer P 的 `peer_scale` 帧对齐、`fence.proxy.async` 排序 STSM->WGMMA、移除多余的输出反置换。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa_backend.py: flashmla_sparse_q8 路由/派发/校验。
python/sglang/jit_kernel/csrc/sparse_mla_q8kv8_prefill_sm90/kernel.cuh: 并发/正确性修复与 Vt half-buffer 拆分。
python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py: 新增 gather_dequant_requant_fp8_paged 融合 kernel。
python/sglang/kernels/ops/kvcache/cache_ops.py: 新增 concat_and_cast_q_fp8_pad 融合 q-prep。
python/sglang/srt/server_args.py: DSA_CHOICES 增加 flashmla_sparse_q8。
docs_new/docs/advanced_features/attention_backend.mdx: 后端文档表格更新。
test/registered/jit/..._sm90.py: 单测扩到 21 项(+6 个 e2e corner-case)。
```

-1 哨兵处理(消除数据相关循环):

```cpp
const bool t_is_pad = (t < 0);
t = t_is_pad ? (pad_base + offs) : t;  // 互异零 pad 行,load 恒安全
is_cur_token_valid = !t_is_pad;        // softmax 掩 -INF,不得计分
```

## 性能声明

H200,TP8/EP8/DP8,两臂均 fp8 KV,重复 A/B 取中位(每行 Δ 波动约 0.3%):chunk32768 时单请求延迟 +4.6%~+7.7%(峰值 +7.7% @ il=16k:7379 vs 6855 tps;64k 为 +6.9%、160k 为 +5.3%),饱和吞吐 +1.8%~+2.7%(如 il=120k c16:21082 vs 20521 tps);chunk16384 时单请求 +2.6%~+6.9%、饱和 +1.6%~+2.7%。融合 q-prep 贡献约 +3.5% c=1 / +1.7% 饱和(il=16k/32k 实测)。profiler 分解(il=64k):c=1 每.rank GPU-busy 总节省 2156 ms(+6.4%,其中 attention kernel 直接省 154 ms、NCCL barrier 放大 1876 ms),c=16 省 1962 ms(+2.4%),与 E2E 表一致——c=1 收益来自未重叠 all-reduce 的 overlap/makespan 效应。精度:gsm8k 3.6k-65k 七档全 1319 题,|Δ|<=0.004、pooled -0.002,双臂稳定 0.95-0.96;LongBench-v2(10k-147k 独特文档,334 题,thinking-ON)q8 0.602 vs q16 0.616,Δ -0.014±0.015 在采样噪声内。

## 验证与回归风险

正确性:`tests/test_sparse_mla_q8kv8_prefill_sm90.py` 21/21 通过(PR-1 的 15 项 + 6 项新 corner-case:masked -1 哨兵 few-valid 行 vs fp32 参考 cos 0.9998、s_q 2048/4096/6144 NaN 监视、S_KV=65536 大索引)。CI:Base 失败、Extra 通过(pack 标注)。风险:(1) 仅 SM90 + `--kv-cache-dtype fp8_e4m3`,启动期 fail-fast,其他配置需走 bf16 `flashmla_sparse`;(2) 融合 q-prep 要求 2 的幂头数(DeepSeek 128 头及所有 TP 切分满足,否则回落通用 concat+cast);(3) `gather_dequant_requant_fp8_paged` 按恒等 per-tensor scale 假设重定量,依赖模型数值幅度;(4) 正文记录上游 DSA fused top-k v2 在 ~160k prefill 有非法内存访问(影响两个后端,il<=120k 不受影响,后续由 #30512 解决),160k 基准用 `SGLANG_OPT_USE_TOPK_V2=0` 规避;(5) kernel 修复涉及 barrier/parity 协议,注释详尽但依赖复杂内存序推理。

## 关联

前序:PR #25751(PR-1,独立 SM90 kernel + 微基准)、roadmap #25746、PR #30512(DSA fused top-k v2 崩溃修复);同区域:PR #14781/17554(NSA/DSA metadata 优化)、PR #21511(AMD FP8 attention)。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
