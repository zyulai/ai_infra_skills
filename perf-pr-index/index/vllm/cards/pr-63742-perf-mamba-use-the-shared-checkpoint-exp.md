# PR #63742: [Perf][Mamba] Use the shared checkpoint exporter for Mamba2 and drop ssd_checkpoint

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @roikoren755 |
| 合并 | 2026-11-03 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | (none) |
| Diff | pack diffstat 缺失(+None/-None);PR 正文自述 vllm/ 内 +53/-157(含删除的 118 行文件)、tests 内 +35/-17 |
| 链接 | https://github.com/vllm-project/vllm/pull/63742 |

## 一句话结论

Mamba2 prefill checkpoint 改用 GDN/KDA 共享的 MambaPrefillCheckpointExporter 并删掉 ssd_checkpoint,48 例微基准中 39 例更快,还修掉写 null block 的隐患。

## 动机

#60050 为 GDN/KDA 引入共享 MambaPrefillCheckpointExporter 后,共享 kernel 与 Mamba2 专属的 ssd_checkpoint.py 已基本等价(差异只剩 launch 配置、null-block/短 offset mask 与 conv window 寻址),却因 launch 配置不同会让 Mamba2 回退约 1.19-1.27x。同时旧 kernel 没有 null 目的块检查:默认 sparse retention 下跨多个 Mamba block 的非最终 prefill chunk 拿到 checkpoint offset 但目标是 null,旧 kernel 会把 checkpoint 写进 block 0(潜在踩到 attention block 0)。

## 优化点

把 Mamba2 的 store 路径切到共享 exporter 并删除 ssd_checkpoint.py,对 store_cache_checkpoints_kernel 做三处改动:一是 block size 256 -> 512 统一所有 caller(256 时大 state 下比旧 Mamba2 kernel 慢,512 最差情况优于 1024);二是 8 个 stride 从 tl.constexpr 改为运行时参数,免去特化开销(平均约 6%);三是给 conv 段加 guard(`program_id(1) * BLOCK_SIZE < WIDTH * STATE_LEN`),不覆盖 conv window 的 program 直接跳过,大 state 上省 15-22%。另外 export() 增加两个 host 侧断言(非零 offset 必须覆盖 conv window、recurrent_rows 数量匹配),Mamba2 mixer 传 x.transpose(0,1) 适配共享 kernel 的 (tokens, width) 布局。

## 关键改动

```text
vllm/model_executor/layers/mamba/ops/ssd_checkpoint.py: 删除,Mamba2 专属 store kernel(118 行)
vllm/model_executor/layers/mamba/checkpoint.py: block_size 256->512、stride 运行时参数、conv window guard、export() 断言
vllm/model_executor/layers/mamba/mamba_mixer2.py: conv_ssm_forward 改调 MambaPrefillCheckpointExporter.export
tests/kernels/mamba/test_checkpoint_exporter.py: 新增 mamba2 caller(sliced_qkv)与 wide recurrent shape
```

## 性能声明

硬件:B200 单卡,torch 2.14.0+cu130,输出与基线逐位相等后计时。Mamba2(exporter vs 被删 kernel):48 例中 39 更快、6 在噪声内、3 更慢(全为 TP8,每步最多 +2.9us);Super TP1 1-6 rows 每 step 省 23-49us;GDN/KDA TP1/2 在 4+ rows 全部 24 例快 13-37%(Kimi K3 TP1 6 rows 每 step 省 625us);GDN/KDA 全 96 例中 25 例更慢(最多 ratio 1.144、+15us/step,集中在 TP4/8 的 1-2 rows)。kernel guard 单独消融:大 state 14.2 -> 11.1us(Super TP1 6 rows)。GSM8K(main 0.928/0.926 vs 本 PR 0.926/0.926,prefix-cache 命中率约 86%)。未验证 E2E serving 吞吐与 B200 以外硬件。

## 验证与回归风险

列出 pytest 清单(test_mamba2_checkpoint_state、test_checkpoint_exporter、test_gdn_prefill_flashinfer、kimi_k3/glm5next KDA 系列等),VLLM_GPU_SYNC_CHECK=error 下 serving Nemotron 3.5 Super TP1 无 host-device sync;export() 新断言为 host 侧、无 device sync。回归面:共享 kernel 也被 ROCm Kimi-K3 KDA exporter(gfx950,未实测)与 Qwen4Exp PLE 短 conv(慢 2-6%,最多 +0.19us/step)共用;TP8 小 shape 的 1-2 rows 场景在启动延迟主导下可能小幅变慢。

## 关联

前序:#60050(共享 exporter 引入,加 HAS_RECURRENT_ROWS)、#59175(非最终 chunk 的 null checkpoint 列)、#52959(内部 prefill checkpoint 追踪)、#61601/#61497(已合并、本 PR rebase 于其上);邻近:#60659、#61951、#61501、#60888、#62232。
