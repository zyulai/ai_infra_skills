# PR #34620: [Diffusion][ERNIE] Fuse QKNorm with full-width RoPE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-13 |
| 分类 | Attention / Kernel | tier: major |
| Label | run-ci, diffusion, jit-kernel, run-ci-extra, mergeable |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 6 个文件:`qknorm_rope.cuh`、`qknorm_rope.py`、`layernorm.py`、`ernie_image.py`、两个测试文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/34620 |

## 一句话结论

把 ERNIE-Image 的 Q/K RMSNorm 与全宽 rotate-half RoPE 融合进单个 JIT kernel,denoise 对比 torch.compile 提速 5.41%,输出像素级一致。

## 动机

ERNIE-Image 的 attention 块在 QK RMSNorm 之后接 RoPE,此前是三条独立路径:先 `apply_qk_norm`,再对 Q、K 各调一次 `_ernie_rope`,多出两轮全量读写。SGLang 已有通用 JIT 融合 kernel `fused_qknorm_rope_warp`(一次完成 QKNorm+RoPE),但它只支持"半宽" cos/sin cache(每行 `rope_dim`,cos 占前半、sin 占后半);而 ERNIE hoist 的是 concatenate 的全宽 cache(每行 `[full cos, full sin]`,宽 `2*rope_dim`),布局不匹配导致融合 kernel 无法启用。差距虽是两三个小 kernel 的 launch 与访存,但在 50-step 的 diffusion 去噪里逐层累计。

## 优化点

在通用 JIT kernel 上新增 `kCacheHasFullWidth` 编译期模板参数:cache 行宽变为 `2*kRopeDim`,sin 指针偏移从 `kRopeDim/2` 改为 `kRopeDim`,NeoX 路径的 cache 索引直接用 `lane_id` 而非 `lane_id % kHalfRotaryLanes`;pack-KV 模板以 `static_assert` 拒绝 full-width。Python 侧 `fused_inplace_qknorm_rope` / `can_use_fused_inplace_qknorm_rope` / `apply_qk_norm_rope` 全链路传递 `cache_has_full_width`:调用方省略 `rope_dim` 时按 cache 宽度推断(全宽时减半);非 NeoX 或 pack-KV 组合 full-width 时拒绝融合;`apply_qk_norm_rope` 中 full-width 且 fused 不可用时报 RuntimeError 而非静默回退。ERNIE 模型侧:hoist `rope_cache = cat(rope_cos, rope_sin)` 与 `rope_positions`,把 `apply_qk_norm` + 两次 `_ernie_rope` 替换为单次 `_ernie_qknorm_rope`;经 `BitExactFusionGate` 门控——首次使用与 split 路径逐 bit 比对,通过后置 verified,失败自动回退 split kernels;`round_norm_before_rope=True` 保持 BF16 的精确 round。另修两处 fallback 正确性:kernel 部分改写 Q/K 后抛异常时,参考路径从 pristine clones 出发。

## 关键改动

```text
python/sglang/kernels/jit/csrc/diffusion/qknorm_rope.cuh: kCacheHasFullWidth 模板参数,cache 步长/sin 偏移/索引按全宽重排
python/sglang/kernels/ops/diffusion/qknorm_rope.py: API 传递 cache_has_full_width,rope_dim 推断与门控
python/sglang/multimodal_gen/runtime/models/dits/ernie_image.py: _ernie_qknorm_rope 融合调用 + BitExactFusionGate 验证;hoist rope_cache/positions
python/sglang/multimodal_gen/runtime/layers/layernorm.py: apply_qk_norm_rope 接受 full-width cache 并强制 fused
```

```cpp
constexpr int64_t kCacheRotaryDim = kCacheHasFullWidth ? 2 * kRopeDim : kRopeDim;
constexpr int64_t kCosSinStrideBytes = kCacheRotaryDim * sizeof(CacheDType);
...
const auto sin_ptr = cos_ptr + (kCacheHasFullWidth ? kRopeDim : kRopeDim / 2);
```

## 性能声明

条件:B300、ERNIE-Image Turbo、1024x1024、seed 42、50 steps、native backend、`quality=lossless`、prompt enhancement 关闭(保证文本形状确定性;对比基线为 main `dbebc1de`):

- latest main eager:denoise 6.5414 s / e2e 6.6666 s;PR eager:6.4820 s / 6.6063 s——PR eager 比 main eager 快 **0.91%**。
- PR `torch.compile`:denoise 6.8526 s / e2e 6.9215 s——PR eager denoise 快 **5.41%**(e2e 快 4.55%)。
- 输出:同 prompt/seed/分辨率/步数下 before/after PNG 像素级一致,SHA-256 相同(`3d0eae0d...`)。
- 端到端服务吞吐:PR 未提供。

## 验证与回归风险

测试:B300 上 ERNIE + 完整 generic QKNorm/RoPE + Cosmos pack-KV 矩阵 **1259 passed**,覆盖 model-level first-use exact gate、omitted-`rope_dim` full-width API 行为、partial-mutation 异常回退;改动文件 pre-commit 通过。新增单测断言融合路径与 split 参考逐 bit 相等(`torch.equal`)、首试异常时输出仍与参考一致且 gate 置 disabled。风险:full-width cache 仅支持 NeoX RoPE;pack-KV 路径不支持 full-width;ERNIE 融合走 BitExactFusionGate,若平台不满足 bit-exact 会自动回退 split kernels(性能收益消失但正确性保持);Cosmos pack-KV 与既有 cache 布局行为不变。

## 关联

Cosmos pack-KV 路径(同一 fused kernel 的另一模式,full-width cache 对其明确拒绝,保持正交)。无正文列出的其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
