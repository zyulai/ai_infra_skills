# PR #42297: [Speculative] Add opt-in block verification

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @maocheng23 |
| 合并 | 2026-10-05 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, speculative-decoding, run-ci, jit-kernel, release-highlight, bypass-fail-fast |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留 benchmark/docs/kernel 等部分文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/42297 |

## 一句话结论

为随机采样 EAGLE 加 opt-in block verification:验证不再止于首个被拒 token,可选出最长被接受前缀,sampler 微基准延迟最多降 63.6%。

## 动机

标准 rejection sampling 在第一个被拒的 draft token 处停止;论文 [Block Verification Accelerates Speculative Decoding](https://arxiv.org/abs/2403.10444) 的 Algorithm 2 会测试所有 draft 前缀、选最长被接受前缀,并在更早位置被拒后仍可能接受更长前缀,配合缩放后的 correction 分布保持 target 分布不变。此前 SGLang 的 Triton chain sampler 只实现 token-wise 路径,故需要新增 block 验证分支。

## 优化点

(1) 新增 `--speculative-use-block-verification`,自动启用所需的 proposal sampling 与 draft-probability 存储;支持线性 EAGLE/EAGLE3/NEXTN(`--speculative-eagle-topk 1`),CUDA/ROCm,不支持的配置启动即校验失败。(2) 扩展现有 Triton chain sampler 编译期分支:正向累乘 `r_i = min(1, r_(i-1) * p(x_i)/q(x_i))`,按最长优先测试前缀、首个接受即停;保留每前缀的原始随机数,使被跳过的短前缀测试不会改变选中结果;correction 从归一化 `(r_tau * p - q)+` 采样,全接受时 bonus token 从 target 采样。(3) 两个降本捷径:coin 已 ≥ r_i 的前缀跳过词表归约(利用归一化分布的 `h_i <= r_i`);选中前缀的残余质量直接复用于 correction 归一化;大词表用上限 32768、8 warps 的独立归约 tile,CDF tile 保持 4096(token 验证维持原 4-warp)。(4) 复用 proposal buffer、输出索引、TP 同步与 greedy 路径,不改 scheduler 与 KV cache。

## 关键改动

```text
python/sglang/kernels/ops/speculative/reject_sampling.py: Triton sampler 加 block_verification 编译期分支 + h_i<=r_i 捷径
python/sglang/srt/... (server_args/eagle worker): opt-in flag 自动启用 proposal sampling 与 draft 概率存储
benchmark/kernels/bench_block_verification.py: 可复现 sampler 微基准(支持对比 baseline kernel)
test/registered/kernels/ops/speculative/test_block_verification.py 等: kernel/配置测试
```

## 性能声明

Sampler 微基准(GB300,Torch 2.13.0+cu130/Triton 3.7.1,FP32,5 drafts,vocab 131072,`q=0.7p+0.3softmax(rand)`,3 次 do_bench 中位数):block sampler 优化前→后 batch1 159.69→58.07 µs(-63.6%)、batch16 207.87→132.83 µs(-36.1%)、batch128 250.78→146.25 µs(-41.7%);另测 3 drafts/131072:127.61→57.32、163.45→113.39、184.65→131.07 µs;5 drafts/131073:233.53→79.08、334.14→166.65、334.20→166.38 µs。

端到端(Qwen3.8-27B BF16,AIME25,GB300 TP1,native MTP/EAGLE steps=3 topk=1 draft=4,并发 8,3 seeds×30 题,优化前 commit b40ee2c):acceptance length 2.9802→3.0063(+0.88%);output throughput 988.43→995.36 tok/s(pooled +0.70%,per-seed +2.71%/+0.41%/-0.99%,正文自述"不构成一致的吞吐提升");boxed 分数 84.44%→85.56%,post-`</think>` 分数 77.78%→72.22%,正文声明两者均不确立质量变化。正文明确:剩余成本取决于采样与接受情况,不证明 block 总快于 token;E2E 数字未在优化后 kernel 上重跑。

## 验证与回归风险

25 个 kernel/配置测试 + 12 配置子测通过(GB300);统计测试每例 65,536 样本核对论文期望接受长度与输出联合分布;Compute Sanitizer 17 个非统计 kernel 用例 0 错误;4800 组 per-sequence 与优化前 kernel 对照(3 seeds、0/1/3/5/8 drafts、5 种词表至 131073)完全一致(观测等价,非位级保证);精确有理数枚举复现论文 10/9 vs 11/9 例。风险:ROCm 尚未验证;CI Base/AMD 为红;block 验证会在 verify 期增加词表归约,docs 明确建议按 workload 自行 benchmark。

## 关联

论文 arXiv:2403.10444。同区域:同分类 Speculative Decoding 的 PR #37771(HiSparse 多步 swap)、PR #3822(EAGLE small vocab table);依赖现有 rejection sampling 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
