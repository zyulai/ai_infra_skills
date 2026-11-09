# PR #40374: [Diffusion] Fuse lossless SenseNova RoPE for 5% faster H200 inference

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-23 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/40374 |

## 一句话结论

SenseNova-U1.5 三轴 RoPE 复用现有 lossless bf16 rotate-half kernel,H200 上 2048x2048/50 步 worker E2E 快 5.17%,输出字节级一致。

## 动机

SenseNova-U1.5 以 eager 方式应用三轴 RoPE,依赖多个独立的 multiply、negate、concatenate、add kernel,launch 数量与 GPU 时间开销大。复用已有的 lossless bf16 rotate-half kernel 可以在保持逐位一致的前提下减少这部分 device 工作。

## 优化点

在 `apply_rotary_pos_emb` 前加 guard:仅当 Q/K 为 CUDA bf16、cos/sin 同设备同形状、无梯度、非 torch.compile、`unsqueeze_dim==1`、4-D、`shape[2]>=128`、head_dim 属于 {32, 64} 且 cos/sin 形状为 `(B, L, head_dim)` 时,把 bf16 CUDA Q/K 张量路由到已有 `fused_rope_rotate_half_bitexact`——该 kernel 对每个乘积先舍入到 bf16 再相加,与 eager 链路的舍入顺序一致,保证 lossless。空间轴 head 切片先 materialize 成 kernel 要求的 BSHD 布局,返回 BHSD view 接回原 attention 路径。短前缀、不支持的形状/dtype/table broadcasting、梯度与编译场景保留 eager fallback;不改动 normalization、RoPE table 计算、权重或 attention backend。另将 `--profile` 等 profiling 参数透传进 SenseNova sampling whitelist(此前被静默丢弃),仅影响插桩。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/sensenova_u1/neo_unify/modeling_qwen3.py: apply_rotary_pos_emb 增加 fused kernel guard 与 BSHD/BHSD 布局转换
python/sglang/multimodal_gen/configs/sample/sensenova_u1.py: sampling whitelist 加入 profile/perf_dump_path 等字段
python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py: profiling 参数透传回归测试
```

```python
q_embed = fused_rope_rotate_half_bitexact(
    q.transpose(1, 2).contiguous(), cos_rows, sin_rows
)
return q_embed.transpose(1, 2), k_embed.transpose(1, 2)
```

## 性能声明

单张 H200,`sensenova/SenseNova-U1.5-8B-MoT`(revision 固定),bf16、`quality=lossless`、2048x2048、50 steps、CFG 4、seed 42、eager、无 torch.compile,同卡 A-B-B-A:

- Worker request E2E 均值 20.4079 s → 19.3537 s(-5.17%);generation stage -5.19%;client E2E(含 PNG 保存)-5.13%;峰值显存不变(35.6836 GiB)。
- Trace 配对分析(第三个 denoising cycle 切片,252 次 RoPE 调用):RoPE 应用链 launch 2,520 → 840,累计 GPU 时间 28.9995 ms → 5.6673 ms。
- 提交函数 microbench(B=1、4096 tokens、Q/K heads 32/8、bf16,含 guard/transposes/materialization):t 轴 165.90 → 31.59 µs,h 轴 99.05 → 50.18 µs,w 轴 99.05 → 51.93 µs。
- 精度:全部 8 个未插桩输出与 2 个插桩输出 byte-identical(解码像素 max diff 0、SSIM 1、PSNR 无穷)。

## 验证与回归风险

验证:历史记录 98 tests passed(17 个新 CUDA 用例覆盖真实 strided t/h/w heads、奇数长度、非变异、fallback/gradient 行为 + 81 个既有 SenseNova 测试);当前 diff 保留 model 测试文件中的 profiling 透传回归,专用 RoPE 测试文件已在清理中移除(不影响 runtime 代码)。CI follow-up 在合并 main 后 97 个 kernel/model 测试通过;Wan 2-GPU lane 出现超过基线的性能问题,PR 记录为未能定位根因(阈值/断言保留)。风险面:guard 未覆盖的配置(短前缀、其他 dtype、KV-shared 等)走 eager 原路径;merge head 包含独立的 SenseNova shared-RMSNorm dispatch 上游改动,故 5.13% 不代表 merged head 的新测量。

## 关联

#34004、#38530:同款共享 kernel 复用/舍入方法的前序 PR;独立于 #38927(RoPE-table 复用)、#39705(RMSNorm dispatch)、#38905(host-sync);CI follow-up 提及 #40411、#40427(fixture 修复)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
