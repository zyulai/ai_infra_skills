# PR #44036: [Feat] FlashInfer: Add NVFP4 W4A16 and SiTU support to MegaMoE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zianglih |
| 合并 | 2026-10-16 |
| 分类 | Speculative Decoding(classify.py 给出,tier: major) |
| Label | documentation, quant, blackwell, run-ci, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 片段涉及约 10 个源文件,另有 6 个文件因超长被裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/44036 |

## 一句话结论

为 FlashInfer MegaMoE 补齐 NVFP4 W4A16(激活保持 BF16)与 SiTU 激活支持,扩展 Blackwell 上低精度 MoE 的可用配置面。

## 动机

此前 FlashInfer MegaMoE 的 NVFP4 路径只有 W4A4,要求静态 activation scale,序列化 ModelOpt checkpoint 与 `nvfp4_online` 在线转换均无法走 W4A16;Kimi K3 使用的 SiTU 激活也未接入。本 PR 是 #39210 的剩余集成,叠在已合并的 #43789 之上,复用 `SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=1` 开关;默认 NVFP4 MegaMoE 路径保持 W4A4 不变。依赖 FlashInfer 0.7.1 pin(含 FlashInfer #5019),无 Dockerfile/依赖改动。

## 优化点

- 在 MegaMoE 适配层加入 W4A16 config 与预处理:保留 packed NVFP4 权重/block scales、per-expert FP32 weight scales、BF16 激活/EP transport/combine 与 post-FC2 routing;对序列化 checkpoint 的 activation scales 做中和,在线 W4A16 则绕过激活量化。
- 激活接线:gated SiLU/SwiGLU 与 SiTU 覆盖 NVFP4 W4A4/W4A16;Kimi K3 的 GEMM1 alpha/clamp 字段映射为 SiTU beta/linear-beta(与 SwiGLU clamp 分离)。MXFP8/DeepGEMM 路径显式拒绝未接线的 SiTU,Nemotron ReLU-squared 不支持。
- 权重 reload:在四个 prepared weight/block-scale allocation 上恢复 canonical loading views 再拷入新数据,W4A4/W4A16 共享 loader 生命周期,保持 live layer、tuned profiles 与 workspace owner 以稳定 CUDA graph 指针;W4A4 reload 若 input scale 变化则要求重启。
- 单节点 NVSHMEM 默认值:单节点且无 speculative decoding(或 draft A2A=none)时默认 `NVSHMEM_REMOTE_TRANSPORT=none` 等,选择 direct peer memory access,规避 nvshmem 未初始化 proxy global-exit 问题。
- FC2 reduction 默认 `SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=0` 用外部 FP32 accumulation,置 1 才允许非确定的 BF16 atomic 候选。

## 关键改动

```text
python/sglang/srt/layers/moe/flashinfer_megamoe.py: W4A16 config 选择、SiTU 参数映射、reload 权重视图管理
python/sglang/srt/layers/quantization/modelopt_quant.py: W4A16 中和 activation scale、SiTU 允许 megamoe backend
python/sglang/srt/arg_groups/moe_hook.py: W4A16 校验(BF16 combine/dtype)、单节点 NVSHMEM 默认值
python/sglang/srt/arg_groups/overrides.py: nvfp4_online 放行 flashinfer_megamoe runner(opt-in)
python/sglang/srt/model_loader/weight_utils.py: dummy weights 将 weight_scale_2 初始化为 1
```

```python
def _nvfp4_megamoe_activation_kwargs(config: MoeRunnerConfig) -> dict[str, Any]:
    if config.is_gated and config.activation == "silu":
        return {"activation": "swiglu", "gate_up_clamp": config.swiglu_limit}
    if config.is_gated and config.activation == "situ":
        # Kimi K3 stores SiTU's tanh constants in the existing GEMM1 fields.
        return {
            "activation": "situ",
            "situ_beta": config.gemm1_alpha,
            "situ_linear_beta": config.gemm1_clamp_limit,
            "gate_up_clamp": None,
        }
    raise ValueError("FlashInfer NVFP4 MegaMOE only supports gated SiLU or SiTU.")
```

## 性能声明

PR 正文未提供 speedup 或吞吐对比数据,作者明确写明 "no ... speedup claim"。给出的是精度/功能验证(8×B300 SXM6 AC GPU,FlashInfer 0.7.1,Qwen3-30B-A3B-Instruct-2507-FP8,TP4/attention-DP4/EP4):online NVFP4 W4A16 GSM8K 191/200 correct(95.5%),阈值 >90%;serialized W4A16/W4A4 reload 两种 cache-flush 模式均通过,logprobs 误差 ≤1e-4。harness 顺带输出的 latency/throughput 字段被作者注明为 "raw incidental output, not a controlled performance comparison",不构成性能结论。

## 验证与回归风险

新增 registered 测试 `TestFlashinferMegaMoeBackendNvFp4OnlineW4A16`(online NVFP4 GSM8K)、serialized W4A16 reload(两种 cache-flush 模式)及本地 W4A4 对照;CPU adapter harness 29 tests、autotune/cache 15 tests(1 skip)、C2 registered units 65 passed。作者注明 GPU 结果取自 rebase 前的两个 revision,rebase 后 head 未重跑 GPU 验证。影响路径:flashinfer_megamoe runner/A2A(限 SM100/SM103、attention-DP=TP=EP)、Kimi K3、`nvfp4_online` 的 runner 自动选择(W4A16 opt-in 时 auto 改选 megamoe);W4A16 额外要求 BF16 model dtype 与 combine dtype。

## 关联

前序:#39210(原始 W4A16 集成)、#43789(MegaMoE startup autotuning 等已上游部分)。依赖 FlashInfer 0.7.1 pin(含 FlashInfer #5019);同批 bump FlashInfer 0.7.1/0.7.2 见 PR #43494、#44591。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
