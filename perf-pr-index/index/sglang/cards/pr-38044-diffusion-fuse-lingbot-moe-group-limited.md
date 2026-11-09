# PR #38044: [diffusion] fuse LingBot MoE group-limited top-k index selection

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-05 |
| 分类 | MoE |
| Label | documentation, run-ci, diffusion, jit-kernel, run-ci-extra |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见新增 routing/group_limited_topk_triton.py 并更新文档/registry,另有 3 个文件因超长未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/38044 |

## 一句话结论

把 LingBot Video MoE 路由的 group-limited top-k 专家选择从 6 段小 CUDA 算子链融合成单个 Triton kernel,微基准 4.87x 快(0.2207ms → 0.0453ms)。

## 动机

LingBot Video MoE 路由器用一串小 CUDA 算子做 group-limited 专家选择:per-group top-2 归约、group top-k、scatter_ 构造零掩码、expand/reshape 广播、masked_fill(-inf)、最终 expert top-k。在 launch-bound 的单 GPU 上这条链全是开销:每个中间张量都很小,整个计算受带宽与 launch 次数约束。由于选择可以在不改变选中 expert-id 集合的前提下融合,值得做成单 kernel。

## 优化点

新增 `routing/group_limited_topk_triton.py`:每个 token 一个 Triton program,一次性加载该 token 的 score 行(scores + bias 后的 scores_for_choice,fp32 contiguous [T,E]),在寄存器里 reshape 成 [group, experts-per-group] 并做 per-group top-2 求和(group score = 组内最大两值之和;对重复最大值只移除一份,避免丢掉第二个 top-k 项),用显式左 tie-break 的 argmax 循环选出 TOPK_GROUP 个组,把非选中组的专家掩为 -inf 后在展平分数上做静态 top-k 循环写出 expert ids。选中集合与参考 CUDA `torch.topk(..., sorted=False)` 链等价,输出顺序不作为契约。配套:`can_use_group_limited_topk` 显式谓词(CUDA 非 HIP、fp32 contiguous、n_group>1、每组至少 2 个 2 的幂个专家等),不满足配置走 eager PyTorch 路径,kernel 失败不再被静默吞掉;以 `register_custom_op` 注册(带 fake impl),覆盖 `torch.compile(fullgraph=True)`;在 diffusion kernel registry 注册 `diffusion.group_limited_topk` 并经 facade 导出。同时更新 fused kernel 清单(45 operators / 51 implementations)与 benchmark/profile skill 的 fast-path 映射。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/routing/group_limited_topk_triton.py: 融合 kernel + 谓词 + custom op 注册
python/sglang/kernels/ops/diffusion/__init__.py: registry 增加 diffusion.group_limited_topk
docs/docs/sglang-diffusion/fused_kernels.mdx: 新增 MoE routing 一节,更新 inventory
python/sglang/kernels/ops/diffusion/README.md: routing/ 子包说明与 MoE routing 表
test/registered/kernels/ops/diffusion/test_routing.py: 正确性测试入 CI(矩阵含 128 专家/4 组/选 2 组/top-8)
```

per-group top-2 的去重手法:

```python
m1_idx = tl.min(tl.where(g == m1[:, None], group_e, BLOCK_EPG), axis=1)
g2 = tl.where(group_e == m1_idx[:, None], float("-inf"), g)
m2 = tl.max(g2, axis=1)
group_scores = m1 + m2
```

## 性能声明

PR 正文 microbenchmark:单卡 NVIDIA H200,8192 tokens,生产 LingBot Video 路由配置(128 experts / 4 groups / topk_group=2 / top_k=8),100 warmup + 500 测量迭代。参考 PyTorch 选择链 0.2207 ms,融合 Triton 选择 0.0453 ms,**4.87x faster**。正文明确声明这只是 routing-selection microbenchmark,不声称端到端模型加速。

## 验证与回归风险

正确性测试 28/28 通过(H200):六种 expert/group/top-k 配置 x 四种序列长度、均匀并列、重复非均匀分数、不支持布局的拒绝用例、`torch.compile(fullgraph=True)`;9/9 facade/registry 一致性测试通过。断言按"选中 expert-id 集合"比较(参考实现 sorted=False,顺序忽略)。风险点:融合路径有严格形状约束(fp32、contiguous、每组专家数为 2 的幂且 ≥2),不满足即回退 eager;tie-break 行为(argmax 左优先)与 torch.topk 的并列处理需保持一致,测试矩阵已覆盖并列情形;该 kernel 仅 CUDA(`torch.version.hip is not None` 即拒绝),ROCm 走 eager。

## 关联

同区域优化:diffusion kernel registry 中的其他融合算子(norm/modulate/rope/attention/layout),以及 `sglang/srt/layers/moe/topk.py` 的 LLM 侧 group-limited topk 实现(LingBot 的参考来源);同批 PR #36592(Wan FFN GELU epilogue)同属 diffusion kernel 系列。PR 正文未给出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
