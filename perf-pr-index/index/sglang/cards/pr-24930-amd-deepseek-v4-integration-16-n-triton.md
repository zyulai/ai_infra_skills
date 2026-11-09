# PR #24930: amd/deepseek_v4 integration 16/N Triton sparse MLA kernel 0508

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-danli103 |
| 合并 | 2026-05-11 |
| 分类 | Attention / Kernel(tier: major) |
| Label | amd, deepseek |
| Diff | diffstat 缺失(pack 未记录);diff 可用,另有 8 个文件被截断(详见 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/24930 |

## 一句话结论

为 DeepSeek V4 稀疏 MLA 新增 Triton 实现,作为 TileLang 的可切换替代,端到端吞吐提升 8-50%,prefill kernel 约 2.3 倍。

## 动机

DpSk V4 在 AMD 平台上的 sparse MLA 此前只有 TileLang 实现。作者称其作为已有 TileLang 实现的替代方案引入:通过环境变量即可切换,零改动 serving 逻辑。背景是 #23600、#23608 两个 PR 冲突集很大,amd/deepseek_v4 集成分支用本 PR 平行推进集成。除替换 kernel 外,还顺带处理 TileLang 路径遇到的实际工程问题:autotune 反复重编译、首次启动编译开销大、CUDA graph 模式缺少提前退出。

## 优化点

新增 `python/sglang/srt/layers/attention/nsa/triton_decode/` 包,含多个 decode kernel 变体(common 共享工具、dsv4 专用、fused、optimized 入口、split-K)。机制要点:(1) `debug_flash_mla_adapter.py` 增加 `triton` backend dispatch,并把 TileLang 的 import 改为惰性加载(选中才 import);(2) `__init__.py` 用轻量 `_KVScopeAdapter` 桥接 flash_mla 接口到 Triton kernel,lse 转置对齐 tilelang/flash_mla 约定;(3) autotune key 用 `_bucket_total_tokens` 把 total_tokens 上取整到 2 的幂——serving 中每个 batch 的 token 数都不同,精确值作 key 会对每个唯一值重编译,bucket 化把唯一 key 数压到约 15 个,大幅降低 autotune 开销;(4) DSV4 gather+dequant kernel 按工作负载大小分档:元素数 <32K 用固定 BLOCK_TK=128 kernel(避免 autotune 开销),<150K 用 1D fused kernel(单次 launch 省开销);(5) 为 CDNA4/gfx950(MI355X,256 CUs)增加 BLOCK_H=64、BLOCK_N=256 等大块配置,支持 attn_sink 与 split-K 大 topk 场景。

## 关键改动

```text
python/sglang/srt/layers/attention/debug_flash_mla_adapter.py: 增加 triton backend dispatch,TileLang 惰性加载
python/sglang/srt/layers/attention/nsa/triton_decode/__init__.py: flash_mla 接口到 Triton kernel 的适配层
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py: 统一稀疏 decode kernel、autotune bucket、chunked/split-K runner
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py: DSV4 gather+dequant kernel(d_qk=512)与分档阈值
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py / _optimized.py / _splitk.py: fused / 优化入口 / split-K 变体(截断部分)
```

autotune bucket 关键逻辑:

```python
def _bucket_total_tokens(total_tokens: int) -> int:
    # Round total_tokens up to the nearest power of 2 for autotune key stability.
    # Bucketing to powers of 2 limits the number of unique keys to ~15,
    # dramatically reducing autotuning overhead.
```

## 性能声明

正文:"8-50% TPUT improvement on e2e",高并发下更明显;"main benefit comes from the prefill kernel, ~2.3x boost",decode kernel 在高并发下有提升。精度:gsm8k 在开启 Triton kernel 前后 "no obvious diff"。PR 未注明测试所用模型规模、硬件具体型号(除 AMD 语境与代码中 MI355X/CDNA4 注释)、精度与 workload 明细。

## 验证与回归风险

验证:gsm8k 前后对比(正文一句,未给数字)。切换完全由 `SGLANG_HACK_FLASHMLA_BACKEND` 环境变量控制,默认仍是 `tilelang`,不开启则服务路径不变。风险:Triton 路径对 flash_mla 调用参数做子集匹配(`**_unused` 静默忽略 block_table、cache_seqlens 等参数),参数语义若有出入不易察觉;CUDA graph early-exit 与 autotune bucket 影响两个 backend 共用的调度代码;新增 Triton cache 目录写入包路径(`.triton_cache`)。

## 关联

同区域:#23600、#23608(正文说明因冲突过大改用本 PR 与 amd/deepseek_v4 分支平行集成)。同为 AMD/DeepSeek V4 sparse MLA(Attention / Kernel)区域的 TileLang 前序实现。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
