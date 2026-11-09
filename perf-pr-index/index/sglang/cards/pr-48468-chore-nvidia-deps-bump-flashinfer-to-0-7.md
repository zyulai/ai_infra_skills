# PR #48468: [Chore] [NVIDIA] deps: Bump FlashInfer to 0.7.3 and migrate the NVLink one-sided all-to-all to flashinfer.moe_ep

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @leejnau(含 @trevor-m 的 #46885 cherry-pick) |
| 合并 | 2026-11-06 |
| 分类 | MoE(tier: major) |
| Label | dependencies, blackwell, run-ci, jit-kernel, release-highlight, bypass-fail-fast, highest-priority, run-ci-nv-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留部分文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/48468 |

## 一句话结论

升级 FlashInfer 到 0.7.3 并把 NVLink one-sided all-to-all 迁到 flashinfer.moe_ep,同时修复 EAGLE 采样 kernel 编译失败与确定性推理的 MoE finalize 顺序问题。

## 动机

两个改动互为依赖、必须一起落地:0.7.3 删除了 `flashinfer.comm.MoeAlltoAll` 与 `moe_a2a_get_workspace_size_per_rank`(现 `FlashinferDispatcher` 在用),不迁移则 bump 后启动即失败;而迁移(#46885)给 `flashinfer.moe_ep` 传的 `runtime_max_tokens_per_rank` 参数只在 0.7.3 存在(0.7.2 叫 `max_tokens_per_rank`),不 bump 则迁移无法编译。另需依赖 #48467(DeepSeek-V4 trtllm counter buffer)先行,因为 0.7.3 不再有 `_get_trtllm_gen_multi_ctas_kv_counter_buffer` 这个内部名。

## 优化点

(1) `NVLinkOneSidedAllToAll` 切换到 `flashinfer.moe_ep` 新 API(BootstrapConfig/MoEEpCommParams/NVLinkOneSidedConfig/create_communication),dispatch 传 `runtime_max_tokens_per_rank`;workspace namespace 机制改为按 token 容量 padding 出不同持久 workspace。(2) SGLang JIT 树投机采样 kernel(`sgl_kernel/speculative/sampling.cuh`)给 `DeviceSamplingFromProb` 传新增的 `last_valid_index` 参数(flashinfer#5352)——不传则 kernel 编译失败,第一个 EAGLE verify 步就使 scheduler 崩溃。(3) FlashInfer backend 的 `fast_prefill_plan` 适配 FA2 paged-prefill `plan()` 从 20 参变 21 参(新增 `fixed_uniform_q_len`),并像上游 plan() 一样每次清 `_eagle_verify_fp8kv_sm90_state`。(4) MiniMax-M3.1 fused NVFP4 KV store 的 quantize guard 从 `amax != 0` 改为 `sf != 0`,对齐 0.7.3 量化器(flashinfer#6996)。(5) flashinfer#8831 临时 workaround:0.7.2.post1/0.7.3 的小批量 MoE finalize kernel(4 < top-k ≤ 64)与大 batch kernel求和顺序不同,导致 token 输出依赖 batch size、prefill/decode logprob 不一致;确定性推理模式下 pre-routed NVFP4 trtllm MoE 改 `do_finalize=False`,用 SGLang 的 `moe_finalize_fuse_shared`(新增 `out=` 参数)以固定顺序求和;非确定性模式不变。(6) Dockerfile/pyproject/engine.py 版本 pin 0.7.2 → 0.7.3。

## 关键改动

```text
python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py: MoeAlltoAll → flashinfer.moe_ep 迁移
python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh: DeviceSamplingFromProb 加 last_valid_index
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: 确定性模式 _finalize_in_order(do_finalize=False + 固定顺序求和)
python/sglang/srt/layers/attention/flashinfer_backend.py: fast_prefill_plan 适配 21 参 + 清 sm90 state
docker/Dockerfile, pyproject.toml, engine.py: 0.7.3 pin
```

## 性能声明

无速度类 perf 声明;全部为正确性/精度验证数据(4×GB200):

- 端到端 GSM8K 全量 1319 题(DeepSeek-V4-Flash,GB200 低延迟 recipe:TP4 + flashinfer_mxfp4 MoE + EAGLE 3 步 4 draft + `--dsv4-attn-backend trtllm`,temp 1.0/top_p 1.0,sgl-eval 0.1.3):main/0.7.2 = 96.66%、#48467/0.7.2 = 96.74%、本 PR+#48467/0.7.3 = 96.59%,三组 request errors 均 0%。修 spec-sampling 前 0.7.3 得分 0%(JIT 编译错误使 scheduler 崩溃)。
- 确定性推理 logprob 一致性测试(KL 阈 1e-9,TP=2):main/0.7.3 KL 5.6e-6~1.6e-3;本 PR + workaround = 0.0(2/2 runs)。
- @trevor-m 的 moe_ep 迁移验证(0.7.3,4×GB200):GSM8K 96%,a2a/dispatcher/spark 相关测试通过。

## 验证与回归风险

GB200 上 #48467 的 DSV4 trtllm prefill 测试在 0.7.3 过、无 fix 不过;`test_speculative_sampling.py` 11 passed。风险面:所有 FlashInfer 依赖路径(MLA/MoE/sampling)行为随 0.7.3 变化;workaround 仅限确定性模式且依赖 `is_config_namespace_published`,#8831 上游修复后需移除;jit-kernel 套件首跑会遇冷 JIT cache;CI 状态截图时仍在运行(hourglass)。

## 关联

前序:#46885(moe_ep 迁移)、#48467(dsv4 counter buffer)、#46926(0.7.2.post1 fallback,仍 open);上游 flashinfer#5352、#6047、#6621、#6546、#6996、#8831。同区域:PR #31927(0.6.15.post1 bump)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
