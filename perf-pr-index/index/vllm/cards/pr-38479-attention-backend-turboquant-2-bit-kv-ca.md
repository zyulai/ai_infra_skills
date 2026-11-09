# PR #38479: [Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vibhavagarwal5 |
| 合并 | 2026-04-15 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, rocm, intel-gpu, ready, ci/build, v1, nvidia, quantization |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,列出的源码文件约 10 个,另有 18 个文件因超长被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/38479 |

## 一句话结论

新增 TurboQuant 在线 KV cache 压缩注意力后端,slot 压缩 2.6x-4.9x,KV 容量约 4x,k8v4 预设保 79-100% 基线吞吐。

## 动机

KV cache 容量与显存带宽是长上下文/高并发 decode 的主要瓶颈。已有方案多依赖离线校准或改模型权重;TurboQuant(PolarQuant,ICLR 2026)可以在 store 时在线量化:key 用 WHT 旋转 + Lloyd-Max 标量量化,value 用均匀量化,全部由 fused Triton kernel 在写入时完成,无需离线校准、模型改动或权重修改,只需 `--kv-cache-dtype turboquant_k8v4`。

## 优化点

store 路径(K→WHT 旋转→Lloyd-Max 量化→bit-pack;V→均匀量化→bit-pack)与 decode 路径(分 KV split 的 Triton kernel:unpack、dequant、Q·K、score·V)均为 Triton 实现;prefill 用 `flash_attn_varlen_func` 走 O(N) 内存高效路径,小块 continuation-decode(阈值 128 token)直走 TQ decode kernel。关键性能机制:(1) 用 WHT+随机符号翻转替代 QR 分解正交矩阵,自逆、可为后续 in-kernel butterfly 融合铺路,并带来 +0.5-2.5% 的结构化 Hadamard cache 模式收益;(2) `_tq_fused_store_mse` 把 bucketize/centroid gather/residual norm/index packing/value 量化融进单 kernel,消除每层 4 次 PyTorch kernel launch,+18-21% decode 吞吐、-10-12% prefill TTFT;(3) FP8 cast 移入 kernel(`tl.float8e4nv`/`tl.float8e4b15`,按 SM 能力自动选 Ampere/Hopper 格式);(4) slot 尺寸向上取偶而非 pow2,消除最多 47% padding 浪费(t4nc 136B vs 256B);(5) KV store 在副 CUDA stream 上与下一层 forward 重叠(CUDAGraph capture 时禁用);(6) 静态 `NUM_KV_SPLITS` grid 维度使 CUDAGraph 可 capture,GPU 显存估计 33 GiB→8.7 GiB。

## 关键改动

```text
vllm/model_executor/layers/attention/attention.py: 每层初始化 TQ 缓冲(WHT 符号矩阵按 layer_idx*1337 种子、Lloyd-Max centroids、decode 中间 buffer 预分配以避免 memory profiler 误判后 OOM);get_kv_cache_spec 返回 TQFullAttentionSpec
vllm/model_executor/layers/quantization/turboquant/config.py: TQ_PRESETS 四个预设(k8v4/t4nc/k3v4nc/t3nc)与 TurboQuantConfig
vllm/model_executor/layers/quantization/turboquant/centroids.py: Lloyd-Max 最优标量量化器求解(N(0,1/d) 分布,梯形积分,lru_cache)
vllm/config/attention.py, vllm/config/cache.py, vllm/engine/arg_utils.py: kv-cache-dtype 预设与 kv_cache_dtype_skip_layers 边界层保护参数
```

关键代码(每层独立旋转矩阵种子):

```python
_TQ_LAYER_SEED_STRIDE = 1337
layer_idx = extract_layer_index(prefix)
seed = tq_config.seed + layer_idx * _TQ_LAYER_SEED_STRIDE
self.register_buffer("_tq_signs", generate_wht_signs(head_size, seed=seed))
self.register_buffer("_tq_centroids", get_centroids(head_size, tq_config.centroid_bits))
```

## 性能声明

质量(Qwen/Qwen3-4B,head_dim=128;GSM8K 5-shot 200q、NIAH 512-32K 77 probes;基线 GSM8K 0.900/NIAH 100%):k8v4 0.860/100%(2.6x)、t4nc 0.840/100%(3.8x)、k3v4_nc 0.780/100%(4.3x)、t3nc 0.720/100%(4.9x)。吞吐/TPOT/TTFT(Qwen3-4B,4x RTX PRO 6000 Blackwell,cudagraphs+compile):k8v4 在 6 个场景(short-decode 128→512、long-prefill 4096→128、mixed 512→512、high-load 512→128 n=500、very-long-prefill 8192→64、decode-heavy 64→1024)达基线 79-100%,very-long-prefill 234 vs 233 tok/s 持平;k8v4 TPOT 在长序列反而更快(long-prefill 135.2ms vs 138.1ms);更深压缩预设吞吐为基线 65-96%。NC(norm correction)在 4-bit 带来约 0.8% PPL 改善。

## 验证与回归风险

新增 `tests/quantization/test_turboquant.py`(570 行,配置/预设定值/中心点求解等单元测试)与 GSM8K eval 配置;PR 自述已验证 CUDAGraph capture(51 FULL + 51 PIECEWISE)、混合 batch 路由正确性,LM Eval harness 集成测试未完成。风险面:改了共享 attention 层初始化与 kv cache spec 路径(影响所有 attention backend 的 spec 合并语义),`TQFullAttentionSpec` 需通过 uniform-type 检查;仅支持 full-attention 与 uniform sliding-window,hybrid(mamba+attention、interleaved SWA)留待后续 PR。

## 关联

同区域:其他 KV cache 量化/压缩 PR(fp8 kv-cache、int8_per_token_head 等 TRITON_ATTN 量化路径);后续计划中的 hybrid 架构支持 PR(编号 PR 未注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
