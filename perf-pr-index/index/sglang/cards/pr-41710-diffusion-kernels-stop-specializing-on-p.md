# PR #41710: [diffusion] kernels: stop specializing on per-request sequence lengths

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-05 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/41710 |

## 一句话结论

把 diffusion Triton 内核的 per-request 尺寸参数改为不特化,消除冷缓存下首请求的 JIT 重编译卡顿。

## 动机

Triton 对每个整型参数按类别(被 16 整除 / 等于 1 / 其他)各编译一个内核变体。sequence length 与行数随分辨率、prompt、输入图像变化,若请求的尺寸落在 warmup 未覆盖的类别,就会在请求现场编译内核。Qwen-Image-Edit 在 nightly 上首请求即可复现:server warmup 用方形图(8192 tokens,可被 16 整除),而 nightly 的 1024×704 输入 token 数不被 16 整除,冷 Triton 缓存下首个 denoising step 耗时 1.95 s,后续步骤仅 0.34 s。

## 优化点

核心改动是给一组 diffusion Triton 内核加 `do_not_specialize`,把随请求变化的尺寸参数(seq_len、行数、帧数、B/L、VAE 的 T/H/W 等)移出特化集合。这些参数只用于行索引和行 mask,load/store 沿 hidden(或 channel)维向量化,而该维的尺寸与 stride 保持特化,故向量化不受影响。Wan VAE 内核保留了 channel 数、元素总数与 stride 的特化(channel 向量化依赖),per-layer padding 与 cache length 也保留(常量折叠值 cat/pad 内核时间的 1.5%)。算术不变,输出 bit-identical。`mrope.py` 与 `vision_rope.py` 已对各自的 sequence length 采用同样做法。

## 关键改动

```text
python/sglang/kernels/kda_kernels/layernorm_modulate_triton.py: _layernorm_modulate_kernel 与 _qk_ln_head_kernel 加 do_not_specialize=["seq_len","n_rows"]
python/sglang/kernels/ops/diffusion/modulate/scale_shift_triton.py: select01/residual/scale_shift_4d/BLC 各内核对 seq_len、stride_i_b、num_frames、frame_seqlen、B、L 不特化
python/sglang/kernels/ops/diffusion/layout/wan_causal_cache_triton.py: _cat_pad_cl3d_kernel 对 T/H/W/out_t/out_h/out_w、_dup_up3d_add_kernel 对 out_t/out_h/t_offset 不特化
python/sglang/kernels/ops/diffusion/layout/ulysses_qkv_triton.py: Ulysses QKV pack 内核对 rows 不特化
test/registered/kernels/ops/diffusion/test_shape_specialization.py: 新增,断言换长度类别不触发编译
```

## 性能声明

H200、`triton.testing.do_bench`、production-like 形状(一个长度可被 16 整除、一个不可):per-head q/k LayerNorm 不特化后快 12%(61.2→53.9 µs),其余内核均在 ±1% 内;单独跑时进程间漂移可达 3.5%。

端到端(nightly harness,空 Triton 缓存):`qwen_image_edit_2511` 首请求额外开销 +2.03 s → +0.38 s;`wan22_ti2v_5b_720p` +0.68 s → +0.14 s,首次 VAE decode 由慢 0.64 s 降为慢 0.17 s,Wan 的 Triton 变体编译数从 45 降到 22。FLUX.1 与 Z-Image 本无需重编译。measured 请求耗时不变。

## 验证与回归风险

新增 `test_shape_specialization.py`:每个 launch 先在可被 16 整除的长度运行,再在 57 和 1(VAE 内核仅 frames)运行,用 `triton.knobs.runtime.jit_post_compile_hook` 断言后两者不触发编译;11 个用例在 main 上全部失败、本 PR 后通过。现有内核单测(`test_modulate.py`、`test_layernorm_modulate.py`、`test_layout.py`、`test_wan_temb_table_slices.py`)及 bitwise cat/pad、DupUp3D、chunked causal-conv 测试不变,特化/非特化变体输出 `torch.equal`。风险面:依赖"channel 数为 16 的倍数"这一前提保持 VAE 参数单类别;若其他分辨率打破该假设,需重新审视保留特化的参数集。

## 关联

#41711(harness 首请求额外开销中的 perf dump);`mrope.py`/`vision_rope.py` 的既有同型做法。同批无其他 diffusion 形状特化 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
