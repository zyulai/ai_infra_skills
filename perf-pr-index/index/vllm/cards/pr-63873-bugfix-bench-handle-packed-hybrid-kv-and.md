# PR #63873: [Bugfix][Bench] Handle packed hybrid KV and speculative recurrent state

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2026-10-27 |
| 分类 | Speculative Decoding(tier: major) |
| Label | bug, documentation, performance, ready |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/63873 |

## 一句话结论

修复 vllm bench forward 在混合 attention/recurrent 模型上的 KV 初始化与投机 recurrent 状态回放(计时区间不变)。

## 动机

vllm bench forward 在两类混合模型上直接失败:hybrid cache group 会把多个 attention kernel block 打包进一个 manager block 且层交错,按 layer kv_cache 直取张量会报 "No block dim of size ...";recurrent(Mamba 类)状态在 verify 时每个投机 token 都可能被读写,但 benchmark 只按 1 个 slot 分配/恢复,状态被踩导致结果错误。

## 优化点

三处构成一个 benchmark 状态正确性修复:zero_kv_blocks/touched_kv_views 改为按 AttentionGroup 取数,当 group.kernel_block_stride 存在时经 map_kv_cache_to_manager_blocks 重建无拷贝视图;_num_blocks 对 MambaSpec 返回 1 + num_speculative_blocks(verify 请求为每个投机 token 预留独立且互不重叠的 recurrent scratch block);touched_kv_views 对 MambaSpec 覆盖全部 slot(attention tail 仍只取尾部窗口)。快照恢复都在被计时的 forward 区间之外,服务执行与计时区间不变。

## 关键改动

```text
vllm/benchmarks/forward/kv_init.py: _group_tensors 经 manager-block 视图取数;MambaSpec 全 slot 视图
vllm/benchmarks/forward/batch.py: _num_blocks 对 MambaSpec 返回 1+num_speculative_blocks
vllm/benchmarks/forward/main.py: 收集 kv_caches/attn_groups 传入
tests/benchmarks/test_forward_cli.py: packed KV / mamba slots / hybrid verify 三个新测试
docs/cli/README.md: hybrid 模型下 verify 状态预留与恢复说明
```

```python
if isinstance(spec, MambaSpec):
    return 0, 1 + spec.num_speculative_blocks
```

## 性能声明

属正确性修复,PR 明确"not a speedup claim"。GB200(4 GPU,native MRV2,dummy weights)验证:GLM-5.3-Flash TP4 + FP8 KV + FlashInfer sparse MLA + ngram5 复现器在 base 上全 rank 报 ValueError,修复后 Decode/Verify/Mixed 完成、有限且确定;decode=1*1@512 / verify=2*6@512:5 / mixed=128@512,2*6@512:5 的中位延迟分别为 4.723 / 8.443 / 20.134 ms(spread 0.03%-0.05%),仅作执行检查。

## 验证与回归风险

新增 tests/benchmarks/test_forward_cli.py 三个测试(packed KV 零化不污染相邻层/块;mamba slot 容量与隔离;hybrid verify 有限且确定)+ tiny Qwen3Next 公开 CLI 回归(真实 LLM engine,dummy 权重)。风险限定在 benchmark 工具路径;快照占用与实际投机状态成比例的额外 GPU 显存。

## 关联

同分类 Speculative Decoding 卡:#34457、#59082、#32542。PR 自述 duplicate-work 检索未发现处理同类 benchmark 状态失败的 open PR;无关联 issue 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
