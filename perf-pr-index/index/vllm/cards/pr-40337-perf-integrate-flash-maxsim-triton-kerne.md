# PR #40337: [Perf] Integrate flash-maxsim Triton kernels for late-interaction scoring

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @roipony |
| 合并 | 2026-10-01 |
| 分类 | Attention / Kernel(tier: major) |
| Label | frontend, mrv2, ready, v1, verified |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动约 10 个文件(新增 flash_maxsim 包,另有正文列出的若干文件未含于 pack diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/40337 |

## 一句话结论

ColBERT/ColPali 的 MaxSim 打分从 padded-bmm 换成 vendored flash-maxsim Triton kernel,变长文档 kernel 层最高提速约 3000 倍,`/v1/score` 端到端 +15%~23%。

## 动机

#35330 引入、#38620 重新启用 GPU 的 vanilla `compute_maxsim_score_batched` 是子批 padded-bmm 路径:变长文档要逐文档 pad 到统一长度、fp32 cast、堆叠张量再 bmm,`[B, Lq, Ld]` 相似度矩阵整体物化。profile 显示变长 workload 下 vanilla 超过 90% 的时间花在 Python 侧 padding/cast/分配而非 bmm 本身。issue #38282 明确请求集成 flash-maxsim,#38620 也为此预留了 API。

## 优化点

三条打分路径自动回退:(1) 零拷贝路径(GPU、d≥16、无 chunked prefill、默认 pooling 参数时默认):`TokenEmbeddingPoolerHead.project_batch()` 一次性投影全部 hidden_states,每个 doc 的 pooler 输出是 `[L, d]` 视图,`flash_maxsim_rerank_direct` 按 (offset, length) 直接从投影张量读 doc,不做任何拷贝;(2) flash-packed 回退(zero-copy 被禁用或 matryoshka/`use_activation=False`):`torch.cat` 一次打包所有 doc,经 `cu_seqlens` 跳过 padding;(3) vanilla 保留(CPU 张量、embedding 维度 <16(Triton `tl.dot` 下限)、`VLLM_FORCE_VANILLA_MAXSIM=1`)。核心是融合 Triton kernel:每 doc 一个 program,Q 分块与 D 分块做 `tl.dot` 后取 max 累加,分数矩阵永不物化、走 SRAM 分块。autotune key 按 Lq/Ld 桶({32..4096})与 `next_pow2(d)` 分桶,`LateInteractionRunner.__init__` 预编译 72 个桶组合避免首请求触发 autotune;配置剪枝按 SMEM 预算(优先 d_pad 而非真实 d 估算),全不合适时回退单一最小 footprint 配置。kernel 运行时失败会禁用 flash 并对本进程回退参考路径。

## 关键改动

```text
vllm/v1/pool/flash_maxsim/: vendored Triton 内核(flash_maxsim_rerank.py 零拷贝 rerank、_common.py autotune 配置/SMEM 剪枝等)
vllm/v1/pool/late_interaction_runner.py: _score 分派 + 共享 query 走 flash + 失败回退
vllm/config/pooler.py: PoolerConfig.enable_flash_late_interaction(--no-enable-flash-late-interaction 传播)
vllm/v1/worker/gpu/pool/pooling_runner.py: 按 pooler_config 构造 LateInteractionRunner
tests/v1/worker/test_late_interaction_runner.py: flash vs fp64 oracle 等新测试
```

```python
S = tl.dot(Q_block, tl.trans(D_block))
S = tl.where(d_valid[None, :], S, float("-inf"))
m = tl.maximum(m, tl.max(S, axis=1))
...
score_acc += tl.sum(m)  # [B, Lq, Ld] 永不物化
```

## 性能声明

A100 80GB、ColBERT 真实 embeddings。Kernel 级(变长文档,vanilla / flash-packed / zero-copy):B=1K Ld=10-100:42.6ms / 0.58ms / **0.05ms(~788x)**;B=5K:220.3 / 2.16 / **0.08ms(~2830x)**;B=10K Ld=10-100:437.6 / 4.04 / **0.14ms(~3034x)**;B=10K Ld=50-500:472.9 / 4.22 / **0.53ms(~892x)**;ColPali B=1K Ld约1030:40.4 / 0.72 / **0.22ms(~184x)**。E2E `/v1/score`(并发客户端,best-of-3):吞吐 +15%~+23%(如 2000 docs/req 并发 4:1.6 vs 1.3 TPS,+23%),P95 +7%~+24%。正确性:kernel vs fp32 bmm `max_abs_err=4e-6`;E2E 5K 真实文档 `max_abs_diff<0.001`、top-3 排名完全一致(位置 4-5 尾部噪声源于 fp16 tensor core 对相差 <5e-4 分数的不确定性)。warmup 冷启动约 160s(无 Triton 磁盘缓存)、热启动约 15s。

## 验证与回归风险

`pytest tests/v1/worker/test_late_interaction_runner.py -v` 4 passed:flash 与逐对 fp64 MaxSim oracle 一致(含长度 1 的 doc)、多 query 走参考路径且完全一致、内核失败自动禁用并回退、禁用标志生效;另有 autotune 剪枝的两个单元测试(d_pad 预算正确性、全不合适时回退最小配置);demo/bench 脚本与 5K 文档 `/v1/score` parity 验证。风险点:零拷贝路径在 matryoshka(`pooling_params.dimensions`)、`use_activation=False`、chunked prefill 下回退(`project_batch` 先归一化再截断,matryoshka 会产生非单位向量);`flash_maxsim_advanced.py`(q-reuse/split-K)已 vendored 但默认路径不分派;`--no-enable-flash-late-interaction` 需经 `propagate_flash_late_interaction` 传到引擎侧。

## 关联

前序:#35330(引入 vanilla MaxSim batched 打分)、#38620(重启用 GPU MaxSim 并预留集成 API,本 PR 为其承诺的后续)、#38282(功能请求)。同区域(检索/打分 kernel):pack 未提供其他同类 PR 编号。
