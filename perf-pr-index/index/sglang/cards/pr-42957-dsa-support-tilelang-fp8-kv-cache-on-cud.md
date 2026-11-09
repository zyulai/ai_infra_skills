# PR #42957: [DSA] Support TileLang FP8 KV cache on CUDA

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Jzz1943 |
| 合并 | 2026-10-11 |
| 分类 | Speculative Decoding(pack 分类;实际为 DSA attention 的 FP8 KV cache 支持) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/42957 |

## 一句话结论

TileLang DSA 在 CUDA 上支持 FP8 KV cache:GLM-5.3-Flash 4×H200 KV token 容量提升 1.94 倍,固定并发下 TTFT/TPOT 增加 5.3%/4.6%。

## 动机

此前 CUDA 上 `--kv-cache-dtype fp8_e4m3` 与 tilelang DSA 后端组合被校验直接拒绝(旧回归:能启动但 decode CUDA graph 捕获时崩,报 "kernel main input KV dtype expected bfloat16, but got float8_e4m3fn"),CUDA 只能 BF16 KV。TileLang DSA 在 ROCm 已有真 FP8 kernel;GLM-5.3-Flash 这类模型在 CUDA 上无法用 FP8 KV 省 KV 内存。

## 优化点

四点:(1) 分派——`tilelang_sparse_fwd` 去掉 FP8 分支的 HIP 限制,CUDA FP8 KV 走既有 partial/combine kernel,CUDA 参数取 `block_I=32, threads=128, block_per_cu=1`,SM 数从 device properties 读取(gfx95 仍用 64/256/2);(2) 布局——KV 容量计算 `calculate_mla_kv_cache_dim` 对 tilelang/tilelang 组合直接返回原始维度,分配不带 per-block scale 元数据的 raw FP8 行(TileLang 消费 raw FP8,与 flashmla_kv/trtllm 的 scaled FP8 布局不同);(3) 关闭不兼容路径——`dsa_backend.supports_mha_one_shot = False`(one-shot FP8 dequantizer 要求 scaled KV 行),HiSparse 明确拒绝;(4) 校验收紧——CUDA FP8 + tilelang 要求 prefill 与 decode 都是 tilelang(raw/scaled 布局不可混用)、不支持 `--dcp-size>1`、要求 SM89+(FP8 tensor core)。BF16 CUDA 路径的 tail kernel 每 128 行成对处理,topk 须为 2304(测试中体现)。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py: FP8 分派去除 HIP 限制,CUDA 走 partial/combine kernel
python/sglang/srt/arg_groups/overrides.py: CUDA TileLang FP8 校验(tilelang-only、SM89+、无 DCP、无 HiSparse)
python/sglang/srt/layers/attention/dsa_backend.py: CUDA FP8+tilelang 禁用 one-shot MHA 路径
python/sglang/srt/mem_cache/kv_cache_configurator.py: raw FP8 行不带 scaled 元数据
test/registered/unit/test_dsa_tilelang_fp8_validation.py: 校验矩阵单测更新
```

## 性能声明

测试条件:GLM-5.3-Flash,4×H200,TP4/EP4,TileLang DSA(prefill+decode),DeepGEMM MoE,EAGLE 5/1/6 + Prefill BCG;workload 32,768 输入 / 1,024 输出,并发 8,64 请求,三次预热运行取中位:

| KV dtype | KV 容量 (tokens) | TTFT (ms) | TPOT (ms) | 输出吞吐 (tok/s) |
|---|---|---|---|---|
| BF16 | 607,232 | 2,664.34 | 12.76 | 518.63 |
| FP8 E4M3 | 1,177,856 | 2,804.25 | 13.35 | 495.48 |

FP8 提供 1.94 倍 KV token 容量;同一并发下 TTFT +5.3%、TPOT +4.6%、输出吞吐 -4.5%(内存换容量的权衡,非延迟加速)。

## 验证与回归风险

精度:GSM8K 1,314 题 5-shot、temperature 0、两轮、Prefill BCG 开启——BF16 无 MTP 均值 95.852%,FP8 无 MTP 95.738%,BF16 MTP 95.472%,FP8 MTP 95.205%;FP8 平均接受长度 4.16 与 BF16 的 4.15 相当;prefill 与投机 CUDA graph 通过。测试:Configuration/GPU 测试(13 tests, 4 subtests)与 pool-configurator 测试(51 tests, 11 subtests)通过;kernel 单测扩展到 CUDA(SM89+)并新增 FP8 CUDA graph 捕获/回放逐位一致(atol=0)断言;校验单测覆盖 tilelang+trtllm 拒绝、SM80 拒绝、DCP 拒绝。风险面:tilelang 成为 CUDA FP8 KV 下 DSA 的唯一可用后端(prefill/decode 必须同时是 tilelang),DCP、HiSparse、one-shot MHA 均不可用;Requires TileLang for DSA prefill 与 decode;PR CI 三项(Base/Extra/AMD)均为失败(x)。

## 关联

GLM/DSA 区域:PR #37889([AMD] GLM DSA prefill top-k v2)。同 KV cache 量化方向:PR #7278(FP8 blockwise MoE)、#12316(CPU FP8)。DSA TileLang FP8 kernel 此前仅 ROCm(gfx950)可用,本 PR 为 CUDA 侧对齐。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
