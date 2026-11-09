# PR #47201: [Perf] [NVIDIA] speculative-decoding: Bind generic provider strategies

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-11-05 |
| 分类 | Speculative Decoding(classify.py 给出,tier: major) |
| Label | documentation, quant, speculative-decoding, blackwell, npu, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 片段约 8 个文件,另有 11 个文件因超长被裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/47201 |

## 一句话结论

为 NVIDIA megakernel 插件建立 speculative decoding 的 provider 绑定层(spec 契约 + 恒存在 facade),使插件可接管 draft/verify 局部路径并在 PD 部署下带来大幅解码加速。

## 动机

可选的 speculative-decoding provider 需要一个窄的、模型无关的边界:provider 发现应只发生一次,graph 调用点应调用恒存在的 strategy,而 provider 缺席或拒绝时必须完整保留 SGLang 原生 RNG、collective、验证与状态生命周期行为。此前 NVIDIA megakernel 相关入口分散在 EAGLE/DSpark 各调用点,依赖 provider-presence 分支。本 PR 只含 provider 契约、绑定与 stock facade,与伴随的运行时优化 PR #47200 相互独立、可分别落地。

## 优化点

- B1a 定义冻结契约:`VocabShardView`、`ShardedLogitsView`、verify/sampling/watermark 视图、`PostVerifyIO/Output/Strategy`、`SpeculativeRuntimeHooks`、`HooksFactory`;factory 由 `BuiltMegakernel` 可选发布,旧 provider 用 `getattr` 发现保持兼容。
- B1b 绑定 provider factory(`bind_speculative_runtime_hooks`):受 `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` 主开关控制,off 时按 identity 返回 stock bundle;provider 只能替换 rank-local `local_keys` 构建,不能替换 runtime verifier runner;rank 间 provider 一致性用 `megakernel_agree` 校验。
- B1c 绑定恒存在的 DSpark draft strategy:一次解析 stock/特化 strategy,每个 batch 走同一 facade,移除 sampler 里的 provider-presence 分支;不支持的 batch size 回退 stock Markov block。
- B1d 引入 `EagleDraftRuntime`,把 draft-loop/draft-extend/round-start 等入口经无条件 facade 路由,graph capture 前完成入口选择,消除 EAGLE 调用点重复检查。
- B2 为 provider 发布 data-only linear prefill 视图(`prefill_view()`),支持 Qwen3.5 GDN prefill 的 split convolution;另注册 `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`(插件读取的独立 slot 选择器,非主开关节点)。

## 关键改动

```text
python/sglang/kernels/plugin_kernels/nvidia/spec.py: 冻结 provider 契约类型(VocabShardView/PostVerifyStrategy 等)
python/sglang/srt/kernel_plugins/nvidia/binding.py: bind_speculative_runtime_hooks 绑定与校验、prefill view 分派
python/sglang/srt/kernel_plugins/nvidia/dspark_binding.py: 新增,DSpark draft strategy 绑定
python/sglang/srt/kernel_plugins/nvidia/speculative_runtime.py: 新增,stock 默认 local_keys/post_verify
python/sglang/srt/kernel_plugins/nvidia/eagle_binding.py: LinearPrefillView/prefill_view() 发布 prefill 数据
python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py: 改走恒存在 draft_runtime facade
```

```python
def bind_speculative_runtime_hooks(model, defaults) -> SpeculativeRuntimeHooks:
    if not envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get():
        return defaults
    built = built_megakernel(model)
    factory = None if built is None else built.speculative_runtime_hooks
    megakernel_agree(factory is not None,
                     what="speculative runtime hooks factory presence")
    if factory is None:
        return defaults
    hooks = factory(defaults)
    ...
    if hooks.post_verify.run is not defaults.post_verify.run:
        raise ValueError("a provider cannot replace the runtime post-verify runner")
    return hooks
```

## 性能声明

绑定层本身把所有调用路由到 stock 路径,收益仅在 NVIDIA 插件(sgl-nvidia-mega#33)提供 megakernel 入口时体现,故数字为整个 stack 级别。AgentX,Qwen3.5 on GB300,PD(P/D 分离),每臂 1200s,run-to-run 噪声约 2%:TPS/user P50——TP8 decode c16:main 686.7 → 本 PR+插件 1180.4(+71.9%);TEP8 decode (EP4) c272:86.2 → 104.1(+20.7%);再加 #47200 分别为 1381.5(+101.2%)与 123.8(+43.6%)。插件关闭时本 PR 与 main 在噪声内(TPS/user P50 +0.7%)。更早的静态 decode 测试(TP8/TEP8)报 bs=1 P50/P90 TPS/user +101.9%/+106.1%(773.39→1561.443)、AgentX TEP8 c272 TPS/user P50 +51.3%、TTFT P50 -80.1%、总 tok/s/GPU +17.2% 等(作者注明这些是 stack 级测量,且部分 AgentX 数据后来被修正重测)。GSM8K(1319 题 greedy,无 draft-acceptance 模拟,seed 1234):TP8 0.978、TEP8 0.980 strict accuracy。

## 验证与回归风险

绑定层专注测试(binding-only 36 passed,1 skipped;组合 runtime+binding 302 passed,18 skipped,2726 subtests);契约测试冻结字段名/类型/编码/形状/回退语义。实测两请求 TEP8 verify batch 中 8 rank 的 434,560 个 provider-slot logits 与 stock 计算零 bitwise 差异;逐层定位在 ReplaySSM 提交态发现 5,898,240 个 BF16 元素中 233 个差异,均在 `atol=1e-6, rtol=2^-7` 内(tolerance-equivalent)。风险:与 #47200 的 EAGLE 改动在 5 个文件重叠,重 base 需按文档化方案解决;`speculative_runtime_hooks` 成为必须声明的成员,配套插件提交(sgl-nvidia-mega#33)需先于或同时落地;CI 快照显示 Base/Extra/AMD ROCm 均未通过(以合并时为准)。

## 关联

伴随 PR:#47200(runtime 优化,与本 PR 独立)。同区域:EAGLE/DSpark megakernel 路径的后续 provider PR;依赖 sgl-project/sgl-nvidia-mega#33 插件提交。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
