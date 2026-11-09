# PR #30971: [minimax-m3] fp8 attention GEMMs on SM100 (fp8_e4m3 KV + trtllm_mha)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @alumkal |
| 合并 | 2026-08-01 |
| 分类 | Quantization(tier: major) |
| Label | performance, blackwell, run-ci, jit-kernel, run-ci-extra, kernel |
| Diff | 素材未提供 diffstat(16 个文件因体积原因从 diff 中省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/30971 |

## 一句话结论

MiniMax-M3 在 SM100 上把 sparse/MSA/indexer/dense attention GEMM 全链路跑在 fp8_e4m3,32k 输入 prefill 吞吐 +65.9%,同 KV 预算省约 20 GB/GPU。

## 动机

MiniMax-M3 此前所有 attention GEMM 均为 bf16:开 `--kv-cache-dtype fp8_e4m3` 时 fp8 cache 会在读取时被拉宽回 bf16(widening-dequant 契约),lightning-indexer cache 更是完全 bf16。而 Blackwell 上 MSA 的 fmha_sm100 内核与 trtllm-gen 的 dense 内核都有原生 fp8 路径,可以把 sparse/MSA/indexer/dense 的 attention GEMM 端到端跑 fp8_e4m3(fp8 tensor-core MMA + fp8 index KV cache),主要加速 prefill。

## 优化点

三个独立有用的 commit:(1) `minimax_decode_topk` 把 block id 输出契约从"front-packed、无序"改为升序排序(smem 内 O(k²) rank 置换,k≤64;`kMaxTopK` 32→64 并加 RuntimeCheck)——这是 MSA fmha_sm100 消费方的硬性要求,`kv_block_indexes` 必须严格升序,否则其 sorted-order early-exit 会错误掩掉部分尾块;顺带修了 HIP wave64 的 shuffle 宽度。(2) `trtllm_mha` 经 trtllm-gen 的 dynamic tokens-per-page 内核(flashinfer ≥ 0.6.12)接受 `page_size == 128`,并在构造时校验 trtllm-gen 前置条件(GQA、QK/V head dim 相等、2 的幂 page),不支持的组合在初始化即报错。(3) fp8 attention-GEMM 模式本身,无 opt-in flag:`m3_fp8_attn_gemm_enabled()` 由 `kv_cache_dtype==fp8_e4m3` + `attention_backend==trtllm_mha` + SM100 推导并尽可能激活;`_minimax_m3_overrides` 把 SM100 上 fp8_e4m3 KV 的默认后端翻到 trtllm_mha(page 128),`fp8_e5m2` 则告警留在 widening Triton 路径(fmha_sm100 的变体查找会静默把 e5m2 派发到 e4m3 内核)。精度回退开关 `SGLANG_DISABLE_M3_FP8_ATTN_GEMM=1` 强制回 pre-fp8 数值路径。模式开启后:main 与 lightning-indexer KV 池均为 fp8_e4m3,sparse 后端在 KV store 与 DP trim 之后把 q/idx_q cast 成 fp8,Triton sparse 内核跑 fp8x8 `tl.dot`(P 量化到 e4m3 做 PV MMA,与 fmha_sm100 fp8 内核同契约),MSA 用 uniform-e4m3,dense 层走 trtllm_mha 既有 fp8-q 路径。

## 关键改动

```text
python/sglang/kernels/jit/csrc/minimax/minimax_decode_topk.cuh: block id 升序输出(warp rank 排序)、kMaxTopK 64、HIP wave64 shuffle 修复
python/sglang/kernels/ops/attention/minimax_sparse/(common|decode|prefill)/*: IS_FP8 分支(fp8x8 tl.dot)、q/k/v_scale 传递、_sort_ids_ascending
python/sglang/srt/arg_groups/overrides.py: SM100 + fp8_e4m3 时默认后端翻到 trtllm_mha(e5m2 告警、kill switch 处理)
python/sglang/srt/server_args.py 等: m3_fp8_attn_gemm_enabled() 推导与 trtllm_mha page-128 约束(部分文件在省略的 diff 中)
```

```python
if (
    server_args.kv_cache_dtype == "fp8_e4m3"
    and not envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get()
):
    # fp8 attention GEMMs activate whenever possible;
    # only trtllm_mha serves the dense fp8-q path
    overrides["attention_backend"] = "trtllm_mha"
else:
    overrides["attention_backend"] = "fa4"
```

## 性能声明

`sglang.bench_one_batch`,TP4,batch 1,output 64,两侧同 flag(`--mem-fraction-static 0.60 --cuda-graph-max-bs 8`),A/B 仅由 kill switch 区分:8192 输入 prefill 0.209 s / 39,201 tok/s 对 0.250 s / 32,724 tok/s(+19.8%);32768 输入 0.606 s / 54,089 tok/s 对 1.005 s / 32,598 tok/s(+65.9%)。decode 约 parity(median 104.5 对 108.5 tok/s,该模式瞄准 prefill,decode 每步多一次 fp8 q-cast 且 MSA decode 在 CUDA graph 下仍由 env 门控)。显存:同 2,929,408-token KV 预算下,fp8 indexer cache 使 init 后多出约 20 GB/GPU 可用(post-capture 91.9 对 71.9 GB)。精度:GSM8K(`run_eval --eval-name gsm8k --num-examples 1319`,chat API,M3-Sparse mxfp8 checkpoint,TP4,`--attention-backend trtllm_mha --kv-cache-dtype fp8_e4m3`)fp8 ON 0.970 对 OFF 0.968。

## 验证与回归风险

内核级 parity 与单测:`test_fp8_attn_gemm.py` 10/10(sparse prefill/decode fp8 对 bf16 parity 及 per-tensor scale 覆盖)、`test_msa_fp8_parity.py` 5/5(含 q_scale)、`test_minimax_decode_topk{,_page_table}.py` 148/148、`test_trtllm_mha.py -k page128` 4 passed + 4 subtests(decode/extend/CUDA-graph parity 与 init 拒绝用例)、`test_model_overrides.py` 67/67;CI Base 与 Extra 均通过。风险面:模式无 flag、满足条件即默认激活(有 kill switch 供精度 triage);行为变化——显式 `--page-size 128` + trtllm_mha 在非 GQA 模型上由"回退 64 并告警"改为 init 时 raise;e5m2 KV 留在 widening 路径、fp8 Q 下 e5m2 被 sparse op 拒绝;per-tensor scale 字段尚无 checkpoint loader 填充(None = unit scale);仅影响 MiniMax-M3 在 SM100 的路径,`SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH` 门控保持不变。

## 关联

依赖 flashinfer ≥ 0.6.12 的 trtllm-gen dynamic tokens-per-page 内核与 MSA fmha_sm100 的 fp8 变体;素材未引用其他 PR 编号(同区域为 MiniMax-M3 sparse attention / Blackwell fp8 KV 相关工作)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
