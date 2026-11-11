# PR #46213: [Perf] flashinfer-megamoe: Add an optional decode workspace

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Jiminator |
| 合并 | 2026-11-10 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, quant, run-ci, bypass-fail-fast |
| Diff | +128/-35,6 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/46213 |

## 一句话结论

为 FlashInfer MegaMoE 增加可选小容量 decode workspace,autotune 关闭时 decode TPOT 最高 1.78×。

## 动机

MegaMoE 的 per-call 开销随 workspace 容量增长。startup autotuning 会为 decode 准备小容量 profile,但当 `--disable-flashinfer-autotune` 或 `--flashinfer-autotune-skip-ops flashinfer_megamoe` 时,每次 forward 都用主 workspace,小 decode batch 也要为满容量买单。PR 从 Beam 工作(#44747/#45256)中拆出,专门解决这个问题;早期版本里的 MegaMoE startup autotune 修复已移入 #49096(FlashInfer 0.7.3)。

## 优化点

新增环境变量 `SGLANG_FLASHINFER_MEGAMOE_DECODE_MAX_TOKENS_PER_RANK`(默认 0=off)。`MegaMoeTunedForward` 在某层第一个未调优 forward 时懒创建第二个小容量 workspace,且只有这类 untuned forward 使用它;选择条件是"没有任何 DP rank 在 extend,且所有 rank 的 DP-gathered token 数都放得下"——MegaMoE 是 lockstep collective,所有 rank 读同一份 gathered counts,因此选到同一 profile。startup 校验拒绝负值和不低于主容量的值;非 ModelOpt NVFP4 MoE 层(唯一会建 decode workspace 的路径,含 `nvfp4_online`)只告警。默认 autotune 开启时 workspace 不被使用(每 rank 记录一次),因为 tuned profiles 已覆盖小 batch;统一 `forward_megamoe` helper 服务 plain/tuned/decode 三条路径。

## 关键改动

```text
python/sglang/srt/layers/moe/flashinfer_megamoe_autotune.py: _untuned_workspace 懒创建与选择逻辑,forward_megamoe 统一入口
python/sglang/srt/layers/moe/flashinfer_megamoe.py: _resolve_decode_max_tokens_per_rank,仅 NVFP4 路径传 with_decode_workspace=True
python/sglang/srt/arg_groups/moe_hook.py + environ.py: startup 校验与新 EnvInt
docs/docs/references/environment_variables.mdx: 环境变量文档
```

```python
if self._decode_workspace is None:
    self._decode_workspace = mega.create_workspace(
        self._decode_max_tokens_per_rank
    )
if get_is_extend_in_batch():
    return None
if self._common_tokens(tensors) > self._decode_max_tokens_per_rank:
    return None
return self._decode_workspace
```

## 性能声明

4×GB300 节点,FlashInfer 0.7.3,torch 2.14.1+cu130;Beam-1.0-501B-A23B-NVFP4,TP4/attention DP4/EP4,modelopt_fp4,flashinfer_megamoe,trtllm_mha,page 64,CUDA graph+overlap,无投机解码;ShareGPT、512 输出、median TPOT(ms)。autotune off 时开 64-token decode workspace:c=1 12.37→6.94(1.78×)、c=16 15.94→10.59(1.51×)、c=64 20.56→15.02(1.37×,两轮均值比);仅跳过 MegaMoE 调优时接近(7.07/10.71/15.12);默认 autotune 下 workspace 未用,tuned profiles 给出相近 TPOT(7.10/10.67/15.20)。收益实为启动时间:autotune-off 105/120 s 就绪,冷启动调优 900 s。GSM8K(1319 题)四配置 95.98%-96.44%。

## 验证与回归风险

GSM8K 1319 题 ×4 配置加 10 请求 smoke;单元测试 test_flashinfer_megamoe.py(GB300 47 passed,CPU 29 passed)、test_flashinfer_autotune_sync.py、test_server_args.py 的 TestFlashinferMegaMoeConfig;PR 自述"No unit test covers the decode workspace"。插桩确认 decode graphs(1-64 token/rank)用该 workspace 捕获,更大 batch 与 eager prefill 用主 workspace。风险:仅 ModelOpt NVFP4 MoE 生效;依赖 FlashInfer 0.7.3 行为;错误配置在 startup fail fast(不加载权重)。

## 关联

拆分自 #45256(Beam,#44747);依赖 #49096 落地的 FlashInfer 0.7.3 修复。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
