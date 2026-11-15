# PR #49771: [Chore] deps: Bump FlashInfer to 0.7.4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @nvpohanh |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | dependencies, run-ci, release-highlight, bypass-fail-fast, parallel-stages, run-ci-nv-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/49771 |

## 一句话结论

FlashInfer 0.7.3→0.7.4 依赖升级:四个 pin 文件更新,并把 UNO 的 `flashinfer.top_k` 移出 `torch.compile(dynamic=True)`(0.7.4 的 stream-keyed scratch lookup 在 dynamic 模式下炸);正文性能声明一节明确写 N/A(dependency bump),无性能数字。

## 动机

Bump FlashInfer 0.7.3 → 0.7.4(release notes)。0.7.4(flashinfer-ai/flashinfer#8696)用 `torch._C._cuda_getCurrentRawStream(index)` 构建每 stream 的 scratch-buffer key;在 `dynamic=True` 下该调用收到 symbolic device index,失败 `invalid argument to getCurrentStream`,破坏了 `test_uno_compact_preserves_original_top_p_nucleus` 与 UNO compact verify path。`top_k` 是 opaque kernel,Dynamo 反正会在其上 graph-break,所以把它移出编译函数零成本。

## 优化点

- 经 `scripts/release/bump_flashinfer_version.py 0.7.4` 更新四个常规 pin 文件:`docker/Dockerfile`、`docker/Dockerfile.cu134`、`python/pyproject.toml`、`engine.py` 版本检查。
- `python/sglang/srt/speculative/uno_utils.py`:`_build_sparse_target_support_tensors` 现在先 eager 调 `flashinfer.top_k`,其后的 normalization 包进新的 `@torch.compile(dynamic=True)` helper `_sparse_target_support_from_topk`;签名不变。
- 兼容性核查(正文):PyPI `flashinfer-python==0.7.4`;`flashinfer-cubin==0.7.4` 与 `flashinfer-jit-cache==0.7.4`(cu129/cu130,x86_64 与 aarch64);0.7.4 依赖列表与 0.7.3 一致,SGLang 的 pin 已满足(`nvidia-cutlass-dsl[cu13]==4.8.0`、`apache-tvm-ffi==0.1.11`);AST 扫描 `python/sglang` 与 `test/` 的全部 FlashInfer import/attribute/call(299 symbols、512 call sites),无移除、无签名变化;SGLang 用到的 C++ 头部分(`DeviceSamplingFromProb`、`SamplingTempStorage`、`FusedAddRMSNorm`、`SegmentPackBits`)未变;私有内部(FA2 paged-prefill `_cached_module.plan` 的 21 参数、`_eagle_verify_fp8kv_sm90_state` 重置、MegaMoE `KNOB_CACHE_VERSION = 2`)不变;两个 `backend="auto"` 行为变化(flashinfer-ai/flashinfer#5188、#5286)不触及 SGLang(SGLang 显式传 `fa2`/`cutlass`);Prims-TS 参数重命名与 `flashinfer.experimental` 源码安装要求均不影响 main;torch.compile traceability 审计(逐个 reaching FlashInfer per-stream scratch helpers 的编译区)只有上述 UNO site 受影响(GEMM call sites 已 `register_custom_op`);deterministic MoE workaround(#48468)保留(flashinfer-ai/flashinfer#8831 仍 open)。

## 关键改动

```text
docker/Dockerfile: ARG FLASHINFER_VERSION 0.7.3 -> 0.7.4
docker/Dockerfile.cu134: ARG FLASHINFER_VERSION 0.7.3 -> 0.7.4
python/pyproject.toml: flashinfer_python[cu13]==0.7.3 -> ==0.7.4
python/sglang/srt/entrypoints/engine.py: assert_pkg_version "flashinfer_python" 0.7.3 -> 0.7.4
python/sglang/srt/speculative/uno_utils.py: _build_sparse_target_support_tensors 移出 torch.compile;top_k 之后 normalization 拆到新 @torch.compile(dynamic=True) 的 _sparse_target_support_from_topk
```

```python
# Runs outside torch.compile: Dynamo cannot trace FlashInfer's stream-keyed
# scratch-buffer lookup, which calls torch._C._cuda_getCurrentRawStream since 0.7.4.
topk_logits, topk_ids = _flashinfer_top_k(
    next_token_logits.contiguous(), max_top_k, sorted=True, deterministic=False,
)
return _sparse_target_support_from_topk(
    topk_logits=topk_logits, topk_ids=topk_ids, next_token_logits=next_token_logits,
    temperatures=temperatures, top_ks=top_ks, top_ps=top_ps,
    batch_size=batch_size, forward_width=forward_width, max_top_k=max_top_k,
)

@torch.compile(dynamic=True)
def _sparse_target_support_from_topk(*, topk_logits, topk_ids, ...) -> ...:
    rows = batch_size * forward_width
    expanded_temperatures = torch.repeat_interleave(temperatures, forward_width, ...)
```

## 性能声明

- 正文 "Speed Tests and Profiling" 一节原文:**N/A (dependency bump)**。pack 的 regex 未检出任何 perf claim;本卡不引用任何加速数字。

## 验证与回归风险

- L40S(torch 2.14.1、FlashInfer 0.7.4):`test/registered/kernels/ops/speculative/test_speculative_sampling.py` 在无 `uno_utils.py` 改动时 11 例失败 1 例,改动后 11/11 通过。新 compiled helper 与 eager reference 在四个 shape(含 vocab 151936、k=64、无 top-p)匹配,无 graph break。作者未在本地跑 GPU 端到端模型测试(依赖 CI)。
- 风险(正文自述):与以往 FlashInfer bump 一样,首次 jit-kernel 运行可能命中冷 JIT cache;性能声明适用于该配对 stack 而非所有依赖/后端组合的说法不适用本 PR(未做性能声明),但 API 兼容性核查覆盖面如上,AST 扫描抓不到 torch.compile traceability 类破坏(故单独审计)。

## 关联

flashinfer-ai/flashinfer#8696(0.7.4 stream-keyed scratch-buffer 行为)、#9354(让 helper 重新可 trace 的长期修复)、#8831(deterministic MoE,仍 open)、#5188 / #5286(不触及 SGLang 的行为变化);SGLang #48468(deterministic inference workaround,保留)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
