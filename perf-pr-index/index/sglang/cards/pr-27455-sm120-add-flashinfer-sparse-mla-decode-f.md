# PR #27455: [SM120] Add FlashInfer sparse MLA decode for DSv4-Flash

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AliceChenyy |
| 合并 | 2026-06-29 |
| 分类 | Attention / Kernel(major) |
| Label | deepseek, run-ci, run-ci-extra |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 3 个文件(flash_mla_sm120.py、deepseek_v4_backend.py、environ.py、测试重命名扩展) |
| 链接 | https://github.com/sgl-project/sglang/pull/27455 |

## 一句话结论

在 RTX PRO 6000(SM120)上为 DeepSeek-V4-Flash 接入 FlashInfer CUTLASS sparse MLA decode kernel,decode TPOT 较 Triton 快 2.2-3.7x,并设为默认后端。

## 动机

#24692 合入的 Triton FlashMLA decode kernel 功能可用但性能有余量;FlashInfer 原生 SM120 `decode_dsv4` kernel 用 CUTLASS block-scaled MXFP8 MMA 实现,可带来 2.2-3.7x decode 提速。接入难点:SGLang SWA KV cache 用 page_size=256 且 footer 布局(scale 存页尾),而 FlashInfer `decode_dsv4` 把 `page_block_size=64` 硬编码为 CUDA 模板参数。

## 优化点

两个层面:(1) 后端切换——`flash_mla_with_kvcache_sm120` 入口默认走 FlashInfer,经 `SGLANG_SM120_FLASHMLA_BACKEND=triton|torch` 可强制回落,FlashInfer 不可用时无缝回退 Triton(零回归);(2) 页格式适配——新增融合 Triton `_page_split_kernel`,在每次 FlashInfer 调用前把 256-token 页(footer:256×576B data + 256×8B scale)单次 launch 拆成 4 个虚拟 64-token 页(footer:64×576 + 64×8,补齐到 37440B 对齐),替代 8 个独立 copy kernel、每步省 344 次 launch;页拆分保持线性 token 序,索引无需重映射;输出 buffer 按设备惰性预分配复用;`mid_out`/`mid_lse`/`output`/`out_lse` split-K scratch 显式预分配。KV cache pool 与 compressor 不动,非 SM120 路径不受影响。extra cache(C4/C128)按原样直传。

## 关键改动

```text
python/sglang/srt/layers/attention/flash_mla_sm120.py: FlashInfer 分支 + _page_split_kernel(单 launch 页拆分)+ lazy scratch
python/sglang/srt/layers/attention/deepseek_v4_backend.py: swa_page_size 从 pool 读取,断言放宽为整除
python/sglang/srt/environ.py: 新增 SGLANG_SM120_FLASHMLA_BACKEND(默认 flashinfer)
test/registered/kernels/test_flash_mla_backends.py: 新增 FlashInfer vs Triton 对比测试(atol=5e-2)
```

关键页拆分注释:

```python
# SGLang SWA KV cache footer layout per 256-token page:
#   [data: 256 * 576 bytes] [scale: 256 * 8 bytes] [padding]
# FlashInfer decode_dsv4 expects per 64-token page:
#   [data: 64 * 576 bytes] [scale: 64 * 8 bytes] [padding to 37440]
```

## 性能声明

PR 注明条件:TP=4,4x RTX PRO 6000 96GB,triton MoE,CUDA graph,ISL=8K / OSL=32,FlashInfer vs Triton:

- BS=1:TTFT 6.17→0.92s(6.7x),TPOT 82.7→22.2ms(3.7x),吞吐 3.21→17.71 tok/s(5.5x)。
- BS=4:TPOT 148.8→64.0ms(2.3x),吞吐 14.87→45.77 tok/s(3.1x)。
- BS=8:TPOT 253.8→114.4ms(2.2x),吞吐 19.83→52.61 tok/s(2.7x)。
- 综合声明:decode TPOT 2.2-3.7x(ISL=8K,TP=4,跨 batch size)。

## 验证与回归风险

GSM8K 10q 0-shot TP=4 FlashInfer 100%;单元测试 FlashInfer vs Triton cosine similarity(atol=5e-2)通过;CUDA graph capture BS=1/2/4/8/12/16 全通过。依赖 FlashInfer 0.6.13 GitHub main(flashinfer-ai/flashinfer#3395 已合;PyPI 0.6.13rc2 尚不含 `_sparse_mla_sm120.py`),首次调用 JIT 编译约 2 分钟。风险:测试项中 CI 未勾选(SM120 硬件不在 CI 矩阵中,仅靠 `is_sm120_supported()` + try-import 保护);CI Base/Extra 快照失败;页拆分 buffer 惰性分配按最大需求增长、不复缩;默认后端变为 FlashInfer,若环境装的是不含该 kernel 的 FlashInfer 版本需显式回退 triton。

## 关联

前序 PR #24692(SM120 Triton FlashMLA decode);参考 vLLM vllm-project/vllm#43477(同一 FlashInfer kernel,vLLM 原生 block_size=64 无需页拆分);上游依赖 flashinfer-ai/flashinfer#3395。其余同分类 PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
