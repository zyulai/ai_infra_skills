# PR #37938: [Perf] Vectorize alloc_extend_naive to remove the per-request Python loop

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @alexnails |
| 合并 | 2026-09-05 |
| 分类 | Attention / Kernel (tier: major) |
| Label | memory-pool |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/37938 |

## 一句话结论

把 paged KV 分配器 `alloc_extend_naive` 的逐请求 Python 循环全向量化,decode 型大批量快 20-100x、长 prefill extend 约 5x,行为不变。

## 动机

`alloc_extend_naive` 为每个请求填充其 extend 区间的 KV-cache slot 索引,原实现是对 batch 逐请求的 Python 循环:开销主要是 O(batch) 的 Python/torch dispatch,几乎与每请求 token 数无关,大 batch 下成为分配路径瓶颈。

## 优化点

重写为纯张量运算,语义与循环逐位等价。三阶段契约(补齐当前部分页 num1、整新页 num2、最后一页开头 num3)整批并行计算:`num1 = clamp(seq_lens, max=ceil_prefix) - prefix_lens`;`done_after_1` 对应循环的 `continue` 条件,把 num2/num3 掩成 0,使 `pages_per_req = num2//page_size + (num3>0)` 与循环实际页消耗一致,页分配顺序经 cumsum 保持不变。逐输出 slot(而非逐请求)用 `searchsorted(end_pos, j, right=True)` 找 owner 请求,`local = j - start_pos[owner]` 复现循环变量,再按 `is_phase1/is_phase2` 三选一公式填充。`torch.where` 不短路,两条死分支的 `free_pages` 索引必须全 slot 合法——由此发现并修复两个越界缺陷(page_idx2/3 对空 free_pages 的索引),最终修复是 host 侧形状 guard:`free_pages.numel()==0` 时(调用方契约保证此时全 batch 无需新页)直接写 phase-1 值返回,`numel()` 是静态 shape 元数据,不引入 host-device sync;非空路径再对索引做防御性 min/max clamp。

## 关键改动

```text
python/sglang/srt/mem_cache/allocator/paged.py: alloc_extend_naive 由逐请求循环改为 searchsorted + where 的全向量化实现(含空 free_pages guard)
```

```python
j = torch.arange(extend_num_tokens, device=device, dtype=torch.int64)
owner = torch.searchsorted(end_pos, j, right=True)
local = j - start_pos[owner]
...
out = torch.where(
    is_phase1, val_phase1, torch.where(is_phase2, val_phase2, val_phase3)
)
```

## 性能声明

CPU(`torch.set_num_threads(1)`),microbench:

- 按 batch size(page_size=16):bs 1 为 1.15x,bs 8 为 5.20x,bs 64 为 12.61x,bs 256 为 17.24x,bs 1024 为 15.40x。
- bs=1 两实现约 40-55 微秒/次,在测量噪声内;跨重复运行 vectorized 对 page_size {1, 128} 略慢(0.84-0.94x)、对 {16, 64} 略快(1.09x-1.15x),PR 如实报告 bs=1 无一致收益。
- 按 extend 长度(page_size=16):1-8 tokens(decode-like)bs 64 为 20.3x、bs 1024 为 109.5x;32-64 tokens 为 24.5x/47.9x;256-512 tokens(prefill-like)为 5.5x/4.9x。PR 结论:长 prefill extend 约 5x,大 decode 型 batch 约 100x,无单一答案。

## 验证与回归风险

本 PR 按任务要求不添加测试文件;正确性靠一次未提交的临时 randomized differential 脚本:20,000 个用例(与 `origin/main` 的原循环逐位 `torch.equal`,覆盖 bs 1-256、page size {1,4,16,64,128}、prefix 为 0/整页倍数、extend 含 0、tight/slack free-page pool,以及后补的 all-zero-new-pages、混合 batch、page_size=1 空 pool 形状)——0 mismatch、0 crash。两个真实调用方(NPU `allocator_npu.py` 的 `alloc_extend` 与 sglang-simulator 的 `_alloc_extend_cpu`)均有 pool >= demand 前置检查,是空 pool guard 正确性的前提;违反契约的输入对旧实现同样崩溃。风险面:bs=1 时两种 page size 有轻微回退;guard 路径含 `assert`(依赖调用方契约);合并时 CI Test (Base)/(Extra)/AMD ROCm 7.2 均为失败状态。

## 关联

同区域:paged KV cache 分配路径(`PagedTokenToKVPoolAllocator`)及其 NPU/simulator 调用方;pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
