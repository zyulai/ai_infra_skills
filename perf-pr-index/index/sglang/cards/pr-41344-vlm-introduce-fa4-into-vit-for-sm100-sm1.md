# PR #41344: [VLM] Introduce FA4 into ViT for SM100/SM103

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-09-27 |
| 分类 | Attention / Kernel(tier: major) |
| Label | Multi-modal, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及约 9 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/41344 |

## 一句话结论

B300 上 ViT 注意力默认切到 FA4(内核比 Triton 快 2–20x),并以 tcgen05.ld.red 硬件行 max 再提速 FA4 前向 1.01–1.075x。

## 动机

填补 FA4 在 Blackwell 上的三个缺口:(1) FA4 前向没有使用 sm_103 的 `tcgen05.ld.red` 指令——TMEM load 可同时返回所加载值的 per-thread 归约,能"免费"拿到 softmax 行 max,而当前 B300 上每个 softmax step 仍用 FMNMX3 归约每行 128 个值;(2) #25570 以"FA4 尚未在 B300 验证"为由把 SM103 的默认多模态 attention 后端从 `fa4` 改为 `triton_attn`,实测 FA4 内核在 ViT 形状(head dims 64/72/80/128)上比 Triton 快 2–20x 且数值一致,B300 的 ViT 一直付出可避免的代价;(3) ViT CUDA graph runner 不支持 `fa4`(`SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 时 fa4 服务器启动即死 `AssertionError: cu_seqlens must be a torch.Tensor`),不修它就改默认会破坏开启 ViT graph 的每个 B300 部署。

## 优化点

① `flash_fwd_sm100.py`:sm_103 上用 `tcgen05.LdRed32x32bOp(redOp=MAX)` 构造带归约的 TMEM load;`softmax_step` 中无 mask、无 score_mod/bias 的 tile 用它读 S 并同时取得每 32 列 chunk 的 max,交给 softmax;masked tile 保留普通 load 与软件 max(硬件 max 取的是 raw S)。在启用的 tile 上硬件 max 与软件归约完全相等,输出 bit-identical;首个迭代 tile 总是 masked,仅稳态 tile 改变。② `softmax.py` 的 `update_row_max` 接受可选预计算 `tile_row_max`,其后逻辑(fast exp2、rescale 阈值)不变。③ `interface.py` 新增 `SGLANG_FA4_TMEM_LOAD_RED_MAX`(默认 1,0 回落 FMNMX),进 compile key 使两变体共存;非 sm_103 忽略。④ `vision.py` 把平台默认多模态后端从"SM100 为 fa4"扩为全部 compute capability 10.x(SM100/SM103);显式 `--mm-attention-backend` 与 per-model override 不变。⑤ `VisionFlash4Attention` 按"参数是否为 list"接受 graph runner 的 `[cu_seqlens, max_seqlen]` 布局;ViT graph runner 给 fa4 与 fa3 相同 layout,并在 capture 前同时预热 windowed 与 full-attention 两种 layout(FA4 JIT 首见配置即编译,不能发生在 stream capture 内)。

## 关键改动

```text
python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py: sm_103 用 tcgen05.ld.red 取 softmax 行 max
python/sglang/kernels/ops/attention/flash_attn/cute/softmax.py: update_row_max 接受预计算 tile_row_max
python/sglang/srt/layers/attention/vision.py: SM103 默认 mm 后端改 fa4;支持 list 形式 seqlens
python/sglang/srt/multimodal/vit_cuda_graph_runner.py: ViT graph runner 支持 fa4 并预热两种 layout
benchmark/kernels/attention/bench_flash_attention_tmem_red_max.py: red-max 开关 A/B benchmark
```

```python
self.use_tmem_load_red_max = (
    tmem_load_red_max and is_sm103 and score_mod is None and not has_bias
)
...
tmem_load_red_atom = cute.make_copy_atom(
    tcgen05.copy.LdRed32x32bOp(
        tcgen05.copy.Repetition(32), redOp=tcgen05.TmemLoadRedOp.MAX
    ),
    self.qk_acc_dtype,
)
```

## 性能声明

全部 8x B300(sm_103,148 SMs),CUDA 13.0,torch 2.13.0,nvidia-cutlass-dsl 4.6.2。① tcgen05.ld.red.max(ViT 形状,CUDA-graph 计时,10 轮配对、输出 bit-identical):8 images×4096 tokens d=80 1.075x(648.8→588.0 us)、d=72 1.039x、d=64 1.023x;1 image×32768 tokens d=80 1.058x。② fa4 vs triton_attn 内核(head-dim sweep):d=64 4.2–5.4x、d=72 16.2–20.5x、d=80 6.1–7.9x、d=128 5.4–6.6x;GLM-5.3/DeepSeek-V4.1 ViT 形状 graph 模式 2.05x–5.29x。③ 端到端(ViT CUDA graph,Qwen2.5-VL-7B,p25 TTFT):4K 图 866.7/901.7→673.9/747.5 ms(-150~−190 ms),1080p -13~−16 ms,448² 无差异。④ MMMU 900 题:eager/graph 与三模型下 fa4 与 triton_attn 差异在 ±0.4pt 内,最终答案一致率 92.2–97.2%。作者如实列出 fa4 落后场景:eager 微小输入(~60us host 开销)与超多 64-token window(4K 700 windows 时 windowed 层慢 1.7x,靠 full-attention 层弥补)。

## 验证与回归风险

新增 `test_flash_attention_4_sm103.py`(4-gpu-gb300 CI):与 FMNMX 路径 bit-exact 对照(output+LSE `torch.equal`),覆盖 causal/varlen/chunked prefill/GQA/pack_gqa/head dims/sliding window/learnable sink/softcap(该处禁用 red load)/split-KV/paged KV(page 128 TMA、page 64 cp.async)/fp8 e4m3;PTX 检查含 4 条 `tcgen05.ld.red`。FA4 既有单测 330 passed/24 skipped。`test_vision_backend_selection.py` 扩展(9.0→fa3、10.0/10.3→fa4、12.0→triton_attn;list/tensor seqlens)。回归面:fa4 成为所有 SM10.x 的默认,影响全部 B300 多模态部署;graph capture 前必须预热两种 layout。pack 中 CI:Base 通过,Extra 与 AMD ROCm 10 失败标记。

## 关联

前序:#25570(将 SM103 默认改为 triton_attn 的变更,本 PR 反转之)。同批无其他 FA4 相关卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
