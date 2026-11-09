# PR #47976: [Perf] [NVIDIA] moe: Localize DeepGEMM Mega MoE weights across locality domains

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-11-03 |
| 分类 | MoE (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/47976 |

## 一句话结论

在双 locality domain GPU(CC 10.7)上用 `deep_gemm.localize` 把 Mega MoE 权重按域落位,decode 输出吞吐 +2.6%(batch 128)。

## 动机

在有两个 locality domain 的 GPU 上,DeepGEMM 的 Mega MoE 把每个 expert 的行 `[d*N/2, (d+1)*N/2)` 分给域 `d` 的 SM;若这些行同时存放在域 `d` 的内存中,权重读取即可保持本地。DeepGEMM 自 sgl-project/DeepGEMM#119 起支持接收 localized 权重(原生 CUDA 13.4 locality domains),本 PR 把它接入 SGLang。难点是 expert 参数在 Mega MoE 之外还有两处读取:deep-ep 回退路径与 EPLB(按 expert 切片,domain-major 张量会骗过 `filter_moe_weight_param_global_expert` 的 shape 检查,导致 EPLB 只搬 scale 不搬 weight)。

## 优化点

新增默认关闭的 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE`,仅作用于 compute capability 10.7,要求 sgl-deep-gemm >= 0.2.4 且以 CUDA >= 13.4 构建。`build_mega_moe_experts_weights` 对 interleaved 的 L1 与 L2 权重调用 `deep_gemm.localize`,kernel 拿 domain-major `(domains, experts, N/domains, K)` 张量(`mega_l1_weights`/`mega_l2_weights`),而 `w13_weight`/`w2_weight` 持有同一内存的 expert-major 视图,`get_moe_weights` 仍按 expert 切片,不复制第二份权重。`build_mega_moe_shared_weights` 同样 localize 融合 shared expert 的 L1/L2(DeepGEMM 要求 shared 与 routed 一致)。`update_expert_weights_single_layer` 处理 P2P:localized expert 的两半分布在不同域、切片不 dense,改为从连续 staging copy 发送、收进连续 temp buffer,三者布局允许时统一以 int64 视图搬运。`should_use_mega_moe`:localized 后没有非 mega 回退,超过 token cap 直接 assert。

## 关键改动

```text
python/sglang/srt/layers/moe/mega_moe.py: localize 路由/shared expert 权重、expert-major 视图、超 cap assert
python/sglang/srt/eplb/expert_location_updater.py: _is_dense/_int64_view,非 dense 切片的 staging copy P2P 搬运
python/sglang/srt/environ.py: 注册 SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE(EnvBool(False))
test/manual/test_expert_location_updater.py: 增加 padded domain-major 存储的 expert-major 视图用例
```

```python
mega_l1, mega_l2 = deep_gemm.localize(mega_l1), deep_gemm.localize(mega_l2)
w13_interleaved, w2 = mega_l1.transpose(0, 1), mega_l2.transpose(0, 1)
experts._mega_moe_weights_localized = True
```

## 性能声明

条件:DeepSeek-V4.1-Flash,单机,DP-attention 4 / EP 4,`--moe-a2a-backend megamoe`,DSPARK,main `e152f531d9` + 本 PR,sgl-deep-gemm v0.2.5.post3(CUDA 13.4),token cap 16384。

- Decode,128K-token 上下文,输出吞吐(per-DP 中位求和),off/on/on/off 交错测量:batch 128 **+2.6%**,batch 224 +0.9%;每个 batch 下两个 on 跑均快于两个 off 跑。
- GSM8K(200 题):off/off 0.990,on/off 0.995,off+EPLB 0.980,on+EPLB(评测中 5 次 rebalance)0.995,无精度回退。
- EPLB 单层权重更新(4 GPU,104 物理 experts/rank,36 次中位):localized 比 plain 慢 +3.5%;按域切片单独发的替代方案慢 +22%(瓶颈是 per-message 开销);server 内一次 rebalance localized 比 plain 长 13.5%(各 5 次均值);rebalance 期间瞬态显存增加 staging copy(约权重显存的 2%)。
- 显存:routed experts 只留 localized 一份,权重显存读数 +1.7%,KV 容量 -1.47%(PyTorch segment 洞无法归还驱动);shared expert 的 L2 被持有两份。

## 验证与回归风险

`test/manual/test_expert_location_updater.py` 通过 gloo(8/32 rank)与 NCCL(4 GPU);`test_waterfill_eplb.py` 通过。Not covered(作者自列):prefill 吞吐(此前 kernel 测量无收益,未复测)、Elastic EP 磁盘重载与 `update_weights_from_disk` 进 localized 参数、多节点 EPLB。风险:默认关闭、仅 CC 10.7; localized 后超 token cap 直接 assert(无回退);显存轻微增加与 rebalance 变慢 13.5%;CI 三项(Base/Extra/AMD ROCm 10)均未通过。

## 关联

依赖 sgl-project/DeepGEMM#119(localized weights 支持);同区域为 NVIDIA Mega MoE / EPLB 路径;正文未列其他 sglang PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
