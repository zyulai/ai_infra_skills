# PR #47653: [Perf] kimi-k3: Seed a validated FlashInfer prefill MoE tactic

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Codoge |
| 合并 | 2026-11-02 |
| 分类 | MoE(major) |
| Label | run-ci |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 为新增 tactic JSON、加载模块、autotune 接线与 2 个测试文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/47653 |

## 一句话结论

为 Kimi-K3 TP1/PP8 16K prefill 播种实测验证过的 FlashInfer 0.7.2 MoE tactic(token tile 192→128),serving 吞吐 +1.44%(standalone 筛选对)。

## 动机

FlashInfer 0.7.1 在 Kimi-K3 TP1/PP8 prefill 的 16K-token MoE candidate enumeration 中只枚举出 192/256 两种 token tile;经小形状枚举发现、再在真实 16K 形状下用捕获路由回放验证的 tile128 tactic 更快。本 PR 把该做法移植到仓库 pin 的 FlashInfer 0.7.2:在 B30Z 上重新枚举得到 tile128/config4,并用真实捕获输入回放验证。tactic ID 是版本相关的,旧 0.7.1/config0 记录不被安装也不被接受。

## 优化点

不改 `AutoTuner.choose_one`、不加 monkey patch、不加新配置开关,只做"启动时种子":在 autotune context 里、用户持久 cache 加载之前,经现有 `load_configs` 机制装入一条 16K prefill MoE 调优记录;随后加载的显式用户 cache 保持优先(可覆盖),关闭 cache reuse 时保持全新调优行为。加载有严格门控:仅 Kimi-K3、TP1/PP8/EP1、chunk16K、BF16、SM103(device capability (10,3))、FlashInfer 0.7.2;FlashInfer 自身的 env 元数据校验(cuda/cublas/cudnn/cutlass 版本、GPU=B30Z)进一步限制到实测组合。cache key 包含 activation/weight 格式、本地/全局 expert 数、top-k、hidden/intermediate 维度、布局与 unpacked routing 表示。PR 明确该记录不是任意序列长度/模型/硬件/EP/decode 的通用 tile128 策略。

## 关键改动

```text
python/sglang/srt/layers/moe/configs/kimi_k3_sm103_fi072_prefill.json: 单条 16K MoE tactic 记录((128, 4),含完整 _metadata 版本指纹)
python/sglang/srt/layers/moe/flashinfer_kimi_k3_tactics.py: load_kimi_k3_prefill_tactics,六重门控后才 tuner.load_configs
python/sglang/srt/model_executor/runner/flashinfer_autotune.py: 在用户 cache 之前先种 serving-derived 默认
test/registered/unit/layers/moe/test_kimi_k3_prefill_tactics.py: 门控矩阵测试(拒绝其他 model/并行/chunk/dtype/硬件/版本)
```

## 性能声明

PR 注明条件:Kimi-K3,单机 8x B300/B30Z,TP1/PP8/EP1,chunk 16384,client C4,ISL 65,536 / OSL 1,CUDA graphs 与 radix cache 关闭,每正式 run 128 requests(8 warmup),上游 `main` base `592754fc5e` + FlashInfer 0.7.2:

- MoE-only(本 PR)vs baseline A1:42,203.78→42,813.05 input tok/s,**+1.44%**(单次筛选对,非重复结果)。
- combined(本 PR + #47651 KDA pack)重复 A/B:+3.27%(两轮 +3.24%/+3.31%),pooled 256 requests 42,167.15→43,547.02 tok/s;P50/P99 TTFT 降 3.28%/3.17%。PR 强调 combined 收益不可单独归因于本 PR。
- trace 佐证:8 个 PP rank 上 736 个 MoE GEMM 的 token tile 均由 192 变为 128;早期 0.7.1 原型 NSYS 显示 MoE GEMM1+GEMM2 每次 10.330→9.622ms(约 -6.85%)。
- 历史 0.7.1/config0 原型筛选对约 +1.41%,PR 声明其不等于 0.7.2 记录的测量。

## 验证与回归风险

真机验证:B300/B30Z(SM103)、PyTorch 2.14.1+cu130、FlashInfer 0.7.2,真实 AutoTuner 加载记录后回放捕获的 Kimi-K3 输入 `[16384, 3584]`(原始 expert weights/scales/top-k routing),实际选中 (128,4) 且输出与捕获的服务参考 **bitwise equal**;随后加载显式用户记录,同一操作选中 (192,4),验证用户 cache 优先。单测:2 个新测试通过(含拒绝 0.7.1/0.7.3 与其他硬件的子例);修复既有 `TestAutotuneCachePhases` fixture 缺 `is_draft_worker` 字段后 12 tests passed。全服务验证:6 次 full-service run,60 次 A1 对比输出 token ID 与输入 SHA256 全部一致(finite output-regression check,非 logits 级或任务级精度基准)。风险:记录与 FlashInfer 版本/库指纹/GPU 强绑定,换环境自动不生效;CI Base/Extra 快照失败(AMD 通过)。

## 关联

伴生 PR #47651(KDA input-copy→pack kernel,combined +3.27% 的另一半);机制同源:FlashInfer AutoTuner 的 tuning-bucket 与 cache 语义。其余同分类 PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
