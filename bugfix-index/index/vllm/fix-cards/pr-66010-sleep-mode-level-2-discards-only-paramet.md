# PR #66010: [Sleep Mode] Level 2 discards only parameters

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66010](https://github.com/vllm-project/vllm/pull/66010) |
| 作者 | aoshen02 |
| 合并日期 | 2026-11-06 |
| labels | bug, documentation, frontend, ready, mrv2, minimax |
| 热度 | 👍 1 · 💬 34 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
level-2 sleep 把模型构建期所有显存都丢弃,非参数状态唤醒后损坏;改为只丢弃权重更新会重写的参数,其余原址保留。

## 问题现象
`sleep(level=2)` 唤醒后非参数模型状态全部损坏(GB200 实测):GEMM-RS/AR 的 `symm_mem.rendezvous` 对端指针表读出全零,首次 forward 即 `CUDA_ERROR_ILLEGAL_ADDRESS`(Kimi-K3 placeholder 权重、DeepSeek-V4.1-Flash + GEMM-RS);DeepSeek-V4.1 `permute_on_load` 索引使 GSM8K 0.90→0.0 且后续 reload 不修复;MXFP4 TRT-LLM permute cache 使 gpt-oss-20b GSM8K 0.375→0.0。main 上 DeepSeek-V4.1-Flash 经 NCCL 权重更新(GB300 TP4)GSM8K 0.9007→0.0;Beam NVFP4 磁盘 reload 后 `sample_tokens` RPC 超时挂死;DeepSeek-V4.1-Flash + DSpark 磁盘 reload 160/160 expert 全不一致。同模式的 Inkling/Beam 对端表、Engram handles(#64922)、普通属性(#60009)、attention impl 状态(#62634)在代码中存在但未实测。

## 根因
正文披露:level 2 靠"不保留可重建的东西"释放显存,而唤醒时唯一被重建的是权重更新写入的内容——参数。但 today 它丢弃模型构建期分配的一切,因为 weights pool 包裹了整个模型构建过程;其余状态只能逐点抢救:buffer 按名字存 CPU,其余以别的方式持有的直接丢失。

## 修复方案
- `CuMemAllocator.sleep`/`XpuMemAllocator.sleep` 新增 `discardable` 参数:weights 分配仅当其中所有 live block 的起始地址都在 `discardable` 中才被丢弃,地址不匹配则保留该分配(错匹配只多占 host 内存,不丢状态);
- `Worker.sleep(level=2)` 通过 `SleepModeBackend.suspend(discardable)` 传入权重更新会重写的参数存储地址,替代原 `level` 参数(插件接口变更,树外 backend 需跟进);
- 据此删除逐点补救:runtime pool 里的模型状态(#63548)、LoRA 状态(#62358)、MiniMax-M3 MSA prefill workspace(#64856)的 tag 隔离,`named_buffers()` CPU 拷贝,以及 `sleep_preserve_parameter_names` 的独立 CPU 存取(保留参数改为"不可丢弃",经分配器备份、PIN_MEMORY 时 pinned)。Level 1 行为不变;KV-init 与 CUDA graph runtime pool 不在 `load_model` 内,保持原样。

```python
offload = data.tag in offload_tags and ptr not in discarded
```

注:pack 的 diff 在 40000 字符处截断(文件粒度),截断部分未纳入本卡。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:改动触及 sleep/wake 全链路(分配器、SleepModeBackend、gpu_worker、model_runner、LoRA mixin)并变更 `suspend` 插件签名;level-2 备份的 host 内存上升(DeepSeek-V4.1-Flash 每卡 0.97→2.35 GiB,Beam NVFP4 0.39→1.55 GiB)。未覆盖:更新不写、checkpoint 中缺席的参数仍被丢弃(用 `sleep_preserve_parameter_names`);`discard(tags)` 不变;权重派生张量在 post-load 刷新前保留旧值而非垃圾,属另一类 bug。验证:RL 循环 + 磁盘 reload 下 DeepSeek-V4.1/Beam/MiniMax-M3/Kimi-K3 oracle 一致、GSM8K 无回归;ROCm MI355X 无回归;`test_cumem.py` 22 passed、`test_sleep_mode.py` 等 20 passed;未跑 XPU、完整 Kimi-K3、Inkling/MiniMax-M3 verifier 级,AMD CI 待批。

## 关联
- [#54477](https://github.com/vllm-project/vllm/pull/54477)(direct weight reload 工作,F11)
- [#63548](https://github.com/vllm-project/vllm/pull/63548)、[#62358](https://github.com/vllm-project/vllm/pull/62358)、[#64856](https://github.com/vllm-project/vllm/pull/64856)(被本 PR 移除的逐点修复)
- [#64922](https://github.com/vllm-project/vllm/issues/64922)、[#60009](https://github.com/vllm-project/vllm/issues/60009)(同模式问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
