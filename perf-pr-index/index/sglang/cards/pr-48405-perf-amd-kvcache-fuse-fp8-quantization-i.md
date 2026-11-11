# PR #48405: [Perf] [AMD] kvcache: Fuse fp8 quantization into the vectorized_5d KV-cache write

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @siliangchen-amd |
| 合并 | 2026-11-09 |
| 分类 | Quantization(tier: major) |
| Label | run-ci, jit-kernel, memory-pool |
| Diff | +307/-17,5 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/48405 |

## 一句话结论

vectorized_5d + fp8 KV cache 时把除 scale 与 fp8 cast 融进 5D 写入 kernel,省掉每层 4 次 launch,输出 tok/s +6.4%~10.9%。

## 动机

`vectorized_5d` KV-cache 布局(`SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`)加 `--kv-cache-dtype fp8_e4m3` 时,`MHATokenToKVPool.set_kv_buffer` 先把 K、V 除以各自 scale 并 cast 成 fp8(每层 4 个 elementwise launch),然后才调用 5D writer,写路径存在一串本可融合的量化前处理。

## 优化点

`reshape_and_cache_shuffle_5d` Triton kernel 增加 `QUANT`/`K_SCALE_IS_PTR`/`V_SCALE_IS_PTR` 编译期分支:cache 为 fp8 而 K/V 不是时,在写入途中以 fp32 除以 per-tensor scale、饱和到 ±fp8_max 再 cast;scale 可为 device tensor(kernel 内读,无 host sync,CUDA graph capture 安全)、host tensor 或 float,经 `_scale_arg` 归一。`set_kv_buffer` 在该布局+fp8 组合下直接把 scales 交给 writer,不再原地除 caller 的 K/V。同时修复 aiter_backend gfx950 head-dim-256 fresh fp8 prefill:改为从 cache 读回刚写入的 K/V(5D 用 `launch_gather_shuffle_5d_to_linear`,NHD 用 uint8 view `index_select`),q 用 `_quantize_q_fp8` 量化;main 上该分支在非单位 KV scale 下本就错误(NHD fp8 写不原地除)。

## 关键改动

```text
python/sglang/kernels/ops/kvcache/shuffle_5d_cache.py: 5D writer 融合 fp8 量化(QUANT/scale ptr 分支)
python/sglang/srt/mem_cache/memory_pool.py: set_kv_buffer 的 vectorized_5d+fp8 融合分支
python/sglang/srt/layers/attention/aiter_backend.py: _written_kv_fp8 读回 cache,hd256 fresh prefill 改读回+量化 q
test/registered/kernels/ops/kvcache/test_shuffle_5d_fp8_write.py: 新增注册测试
test/registered/attention/test_aiter_fp8_kv_scales_paths.py: 新增 head-dim-256 fresh prefill 用例
```

```python
if QUANT:
    if K_SCALE_IS_PTR:
        k_scale = tl.load(k_scale_ptr).to(tl.float32)
    if V_SCALE_IS_PTR:
        v_scale = tl.load(v_scale_ptr).to(tl.float32)
    k = tl.clamp(k.to(tl.float32) / k_scale, -fp8_max, fp8_max)
    v = tl.clamp(v.to(tl.float32) / v_scale, -fp8_max, fp8_max)
k = k.to(key_cache_ptr.dtype.element_ty)
v = v.to(value_cache_ptr.dtype.element_ty)
```

## 性能声明

单卡 MI355X(gfx950),rocm720-mi35x image(2026-11-03,aiter 0.1.26.dev144),`sglang.benchmark.serving` random-ids 1024 in/256 out,每点两轮取均值,`--mem-fraction-static 0.85`。输出 tok/s(main→PR):Llama-3.1-8B-Instruct C1/C16/C64 = 219.14→235.51(+7.5%)/2871.89→3168.26(+10.3%)/7592.76→8154.68(+7.4%);Qwen3-8B +8.6%/+10.3%/+6.6%;Qwen2.5-7B-Instruct +6.4%/+10.9%/+7.7%;Llama-3.3-70B-Instruct-FP8-KV +6.9%/+9.7%/+7.8%。TPOT 降 6.1%-11.3%。one_batch decode step(Llama-3.1-8B):bs1 4.77→4.50 ms(-5.7%)、bs16 -10.3%、bs64 -8.5%。GSM8K 1319 题:Qwen3-8B 0.905→0.904,Llama-3.1-8B 0.756→0.744;Qwen2.5-7B 与 Llama-3.3-70B-FP8-KV 在 main 的 5D+fp8 KV 配置本已损坏(0.025/0.000),本 PR 后仍坏(0.033/0.010),需与 #48434 合修(0.945)。

## 验证与回归风险

新测试校验融合写与 fp32 除法+饱和+cast bit-exact(三种 scale 传法)、超范围饱和、bf16 缓存往返;hd256 fresh prefill 用例在 main 上相对 L2 误差 0.998(NHD)/0.915(5D),本 PR 通过。风险:aiter_backend prefill 分支改为读回 cache(行为修正而非纯优化);caller 的 K/V 不再被原地除,依赖原地除语义的调用方需注意;NHD fp8 写不原地除的既有错误由 #48434 处理。

## 关联

#48434(prefill 中 q 用自己的 scale,合用后 Llama-3.3-70B-FP8-KV 得 0.945);同批 #48406(同 vectorized_5d 布局的 fused QK-norm+RoPE+KV write)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
