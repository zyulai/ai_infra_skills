# PR #41830: [diffusion] kernels: take the Wan VAE decoder's remaining first-decode work off the request path

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-06 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 未提供(pack 素材无增删行数字) |
| 链接 | https://github.com/sgl-project/sglang/pull/41830 |

## 一句话结论

把 Wan VAE decoder 首次请求剩余的 Triton 编译与校验开销移出请求路径,first-decode extra 从 +0.17 s 降到 -0.005 s。

## 动机

#41710 之后,空 Triton cache 下 Wan2.2-TI2V-5B 720p 的首次请求 decode 仍比后续慢 0.17 s,且 server warmup 无法吸收:(1) DupUp3D-add 的两个变体要到 decode 时才编译——channels-first 分支的 stride 是分辨率的乘积(C*H*W、W 等),divisibility class 随形状变化,warmup 没编到这一类;(2) `FusedWanRMSNormSiLU` 的无损 norm+SiLU 融合把首次精确性校验键控在 shape+stride 上,新分辨率会重跑 eager 参考链加 `torch.equal` host sync 达 46 次(46 个 `cudaStreamSynchronize`)。warmup 之所以覆盖不了,是 TI2V 的输出尺寸由条件图长宽比决定:64x64 warmup 图得 928x960,nightly 的 1024x704 图得 1152x768,没有固定 warmup 图能穷举所有比例。

## 优化点

三处特化策略修正。(1) `_dup_up3d_add_kernel`:直接对这些 stride `do_not_specialize` 会让 kernel 慢最多 98%(其可整除性正是编译器为每四个输出生成一次 128-bit 访问的依据);改为 wrapper 从 layout 推导向量宽度 `VEC` constexpr,把 `main`/`out` 的非 unit stride(及 W 为最内维时的 `out_w`)除以 VEC 传入,kernel 内乘回——不依赖具体值地重述同一连续性;unit stride 保持特化但改用 `do_not_specialize_on_alignment` 摘除对齐绑定。(2) `_cat_pad_cl3d_kernel`:为 channels-first 输入新增第二个 input strides unspecialized 实例(其 load 沿 channel stride gather,与值无关);channels-last 输入保留全特化 kernel(unspecialize 会有 10-12% 开销);#41710 的 per-layer `pad`/`cache_t` 特化保留。(3) `FusedWanRMSNormSiLU`:精确性校验签名改为 `(device, dtype, x.is_contiguous(), gamma.dtype)`——这恰好是选择 kernel 模板的量,逐元素运算与尺寸无关。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/layout/wan_causal_cache_triton.py: DupUp3D/cat-pad kernel 特化策略重构与 VEC 推导
python/sglang/multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py: FusedWanRMSNormSiLU 校验签名去 shape/stride 化
test/registered/kernels/ops/diffusion/test_shape_specialization.py: 新增 4 个首请求编译回归 launch
test/registered/kernels/ops/diffusion/test_layout.py: bitwise 测试扩到 cl/cf 四种 layout pair 与 VEC 1/2 宽度
test/registered/kernels/ops/diffusion/test_wan_norm_silu_post.py: 校验每 layout 只跑一次的测试
```

VEC 推导(每线程持有 BLOCK/(32*num_warps)=4 个连续输出,VEC 由内维与公共 stride 的可整除性决定):

```python
_MAX_VEC = 4

def _vec_width(inner: int, strides) -> int:
    vec = _MAX_VEC
    while vec > 1 and (inner % vec or any(s % vec for s in strides)):
        vec //= 2
    return vec
```

## 性能声明

H200,Wan2.2-TI2V-5B at 1280x720,空 `TRITON_CACHE_DIR`,对照 #41710:

- 首次/第二/第三次请求 decode(nightly harness):5.244/5.069/5.078 s → 5.061/5.067/5.062 s。
- First-decode extra:+0.17 s → **-0.005 s**。
- perf dumps 的 `DecodingStage`:5693.7/5518.5/5532.8 ms → 5514.6/5520.7/5516.1 ms。
- 首次请求 decode 期间的编译次数:2 → **0**;进程内 Triton layout-kernel 变体:22(12 cat/pad、6 DupUp3D)→ 15(7 cat/pad、4 DupUp3D);warmup 自身 decode:7.30 s → 6.96 s。
- Kernel time 按真实 decode 的每个 layout-kernel call site/layout/shape 汇总,两版本 ABBA 交错同进程:lossless fp32 -0.05%,`quality=high` fp32 +0.27%,`quality=high` bf16 +0.11%。
- 输出与 #41710 完全一致(值与 strides);实测请求端到端延迟不变。(正文中的 "up to 98%" 指若简单 do_not_specialize stride 会使 kernel 变慢的上限,非本 PR 加速比。)

## 验证与回归风险

验证:`test_shape_specialization.py` 新增 `wan_cat_pad_channels_first` 与 `wan_dup_up3d` 的 cf/cf、cf/cl、cl/cf 三个 launch,四者在 #41710 上失败、本 PR 通过;`test_wan_norm_silu_post.py` 新增 `test_first_sight_check_runs_once_per_layout`(#41710 验证 6 个签名,本 PR 2 个);`test_layout.py` bitwise cat/pad 与 DupUp3D 扫 channels-last/first 四种 layout pair 加两种宽度(VEC 2 与 1);四个测试文件合计 286 passed、1 skipped。风险:Triton 特化/对齐策略改动若 VEC 推导或 stride 缩放有误,会波及 Wan VAE 输出正确性(bitwise 测试钉住);channels-first 路径走新的 gather kernel 实例;CI 状态显示 Base、Extra、AMD ROCm 10 三个 run 均失败。

## 关联

前序/依赖:#41710(本 PR stacked on it,是其 first-decode 清理的续作)。同区域:diffusion/wan VAE layout kernel 优化;同属 Attention / Kernel 分类。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
