# PR #48602: [Perf] dsa: Add an opt-in FlashInfer Prims-TS sparse MLA backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @nvpohanh |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci, release-highlight, highest-priority |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/48602 |

## 一句话结论

新增 opt-in DSA backend `--dsa-prefill-backend prims_ts` / `--dsa-decode-backend prims_ts`(FlashInfer Prims-TS 稀疏 MLA,复用 `trtllm` raw-FP8 路线):decode kernel 比 trtllm-gen 快 1.15–1.8x、比 FlashMLA 最高快 3.35x,raw FP8 cache 每 token 576 B(对比 fp8_ds_mla 656 B,Kimi-K3.1 多容纳约 9% KV tokens);默认后端不变。

## 动机

移植 Kimi-K3.1 分支(#48414)的 opt-in DSA backend:在现有 `trtllm` DSA 路线(raw FP8 MLA cache、q quantize/concat、KV store、top-k transforms、row plan、DCP page-table localization 全不变)上,把 attention kernel 从 trtllm-gen 换成 FlashInfer Prims-TS(`BatchSparseMLADecodePagedTSWrapper`:D576、native E4M3 KV、E4M3 query、FP8 MMA、natural-log LSE)。decode 比 trtllm-gen 和 FlashMLA 都快,cache 每 token 字节数比 FlashMLA 少。需要 FlashInfer 0.7.4(#49771),其中 D576 native-FP8 sparse MLA 来自 flashinfer-ai/flashinfer#7410、#7984,SM107(Rubin)来自 #8186。

## 优化点

- 新 leaf module `layers/attention/dsa/prims_ts.py`:`PrimsTSSparseMLA` 持有 planned launches——行数 ≤1024 精确逐行 plan,之上用 power-of-two packed capacity;plan 在 eager 调用时创建(CUDA-graph warmup 会为每个 captured row count 提前 plan),replay 调 `run(validate=False)`。`check_prims_ts_sparse_mla_support` 为启动检查(D576 MLA、SM100/SM103/SM107、已安装 FlashInfer 能 plan 该 kernel,失败原因进报错)。
- `dsa_backend.py`:`prims_ts` 加入 `_DSA_IMPL_T`;`uses_raw_fp8_dsa_kv_layout(backend)`(`trtllm` 或 `prims_ts`)替换各处 `== "trtllm"` layout 判断;backend init 拒绝非 `fp8_e4m3` KV cache、不能跑 kernel 的 GPU/FlashInfer、replicated index-K DCP(`_validate_prims_ts_dcp`,Prims-TS partial 是 natural-log LSE,combine 只在 sharded index-K 下按 base e 合并);`_forward_trtllm(use_prims_ts=...)` 在 q、KV store、row plan、page table 就绪后移交 `_forward_prims_ts`;Prims-TS 支持任意 head 数,q 不做 head padding;拒绝 checkpoint KV scale(`k_scale != 1`,raw FP8 store 不带 scale 写入)。
- `arg_groups`/`kv_cache_configurator`/`forward_mha`/`forward_mla`/`layers/cp/interleave`:`prims_ts` 作为两个 DSA backend flag 的合法选择并约束 CUDA + `fp8_e4m3`;Kimi-K3.1 在 `flashmla_kv`、`trtllm` 之外接受 `prims_ts`(警告 numerics 与参考 `fp8_ds_mla` 不同、PD 两侧必须都用 raw-FP8-KV backend);overlap 检查改为按 backend 类型 narrow。未设置的 phase 沿用已设置的 phase,其他 Blackwell FP8 模型默认 `trtllm`(同一 576 B/token layout)。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa/prims_ts.py(新增): PrimsTSSparseMLA plan/run 封装 + check_prims_ts_sparse_mla_support 启动检查
python/sglang/srt/layers/attention/dsa_backend.py: prims_ts 加入 impl 类型;uses_raw_fp8_dsa_kv_layout;_validate_prims_ts_dcp;_forward_trtllm(use_prims_ts)→_forward_prims_ts
python/sglang/srt/arg_groups/fields/exec_.py / overrides.py: prims_ts 选择项与约束(需 CUDA + fp8_e4m3);Kimi-K3.1 接受 prims_ts
python/sglang/srt/mem_cache/kv_cache_configurator.py、forward_mha.py、forward_mla.py、layers/cp/interleave.py: 按 uses_raw_fp8_dsa_kv_layout 把 prims_ts 当 trtllm 处理
test/registered/kernels/ops/attention/test_dsa_prims_ts.py(新增): DCP8 partial merge 与 LSE base 测试,注册在 4-gpu-b200
docs(attention_backend.mdx / server_arguments.mdx): 两张 DSA backend 表列出 prims_ts
```

```python
_EXACT_PLAN_MAX_ROWS = 1024
_SUPPORTED_DEVICE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))

def _plan_capacity(rows: int) -> Optional[int]:
    """None plans exactly ``rows`` fixed rows; otherwise the packed row capacity."""
    if rows <= _EXACT_PLAN_MAX_ROWS:
        return None
    return 1 << (rows - 1).bit_length()
```

## 性能声明

(均在 Kimi-K3.1 分支 #48414 上测得)

- Kernel 每层 DSA 时间(µs,GB300,96 heads,每行约 256 keys(DCP8 share of 2048 top-k),6 query rows/request,cold L2,CUDA graph):FlashMLA → Prims-TS:1 req 16.0→11.7(1.37x)、4 req 21.2→9.9(2.14x)、16 req 35.2→14.5(2.43x)、32 req 62.6→18.7(3.35x)、64 req 97.8→29.5(3.32x)。trtllm-gen 在该区间为 13–38 µs,故 Prims-TS 比它快 1.15–1.8x。
- Raw cache 576 B/token 对比 fp8_ds_mla 656 B:Kimi-K3.1 上多约 9% KV tokens。
- 端到端(Kimi-K3.1,agg TP8/DCP8):每 decode step 的 sparse-MLA 时间 891 µs → 332 µs(约一个 ~32 ms step 的 1.7%)。
- 准确度:kernel 对 FP32 reference 相对 L2 1.1–1.6%(FlashMLA 0.2%),误差来自 softmax 概率四舍五入到 E4M3,LSE 匹配到 2e-6;Prims-TS 对 FlashMLA 输出误差 4.1–4.5%,无 NaN/Inf mismatch;GSM8K(全部 1319 题,只换 backend flag)0.9704(Prims-TS)vs 0.9697(FlashMLA)。

## 验证与回归风险

本 head 上仅 CPU 环境:新增 `test_dsa_prims_ts.py` 2 passed / 2 skipped(需 SM100/SM103/SM107 GPU),相关单元测试 248 passed,pre-commit 通过;FlashInfer 0.7.4 的 wrapper `plan`/`run` 参数已对照 `v0.7.4` tag 核对。Kimi-K3.1 分支 GB300(FlashInfer main 的 Prims-TS 拷到 0.7.2 上):FlashInfer D576/native sparse MLA 测试 80 passed,DCP8 partial-merge(24 行)通过,`check_prims_ts_sparse_mla_support` 对 12/24/48/96 heads 通过。1025-row packed-plan case 尚未在 GPU 上跑过;本 port 未在 main 的 GPU 上运行过;SM107 硬件上什么都没跑过;除 Kimi-K3.1 外没有模型用 `prims_ts` 服务过(包括本 port 无 DCP 放行的 DeepSeek-V3.2 与 GLM-5.2);GPU cases 由本 PR 的 B200 CI 覆盖。未覆盖项:未见过行数(≤1024)的 eager batch 会多建一个 plan,其 workspace 在 `--mem-fraction-static` 定好 KV pool 之后分配;首次使用要编译 kernel(CUDA-graph capture 约多 100 s),可用 `FLASHINFER_WORKSPACE_BASE` 指向持久路径缓存;prefill 速度未与其他 prefill backend 对比。

## 关联

#48414(源分支移植)、#49771(FlashInfer 0.7.4 依赖)、flashinfer-ai/flashinfer#7410 / #7984 / #8186(D576 native-FP8 sparse MLA 与 SM107)、#39117 / #40433(DCP 相关上游)、`trtllm` / `flashmla_kv` / FlashMLA(被对比/共存的 backend)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
