# RFC #44697: [RFC]: MTP speculative decoding under pipeline parallelism (PP>1)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44697](https://github.com/vllm-project/vllm/issues/44697) |
| 作者 | atassis |
| 创建日期 | 2026-06-05 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 2 · 💬 5 |
| 分类 | Speculative Decoding |

## 一句话结论
提议把 sampler 的 per-request 有效/接受 token 计数广播到非末级 PP rank,修复 MTP 投机解码在 PP>1 下的崩溃与结果发散。

## 提案动机
MTP 投机解码在 PP>1 下不可用:部分配置崩溃,另一些静默偏离 no-spec greedy 基线(实例 #36643 Qwen3.5+PP、#36872 Qwen3.5+spec 乱码;需求追溯到已关闭的 #14044)。根因:投机解码的 token 记账只在末级 PP rank(采样器所在)计算、从不传给非末级 rank——num_computed_tokens 乐观推进后由依赖 valid_sampled_token_count_gpu 的 GPU kernel 修正,该值只有采样器产出,非末级 rank 跳过修正,导致每次 rejection 后位置过度前移、Rope/KV 错位、验证出错;hybrid(GDN/mamba)模型的 num_accepted_tokens(驱动 conv1d/SSM 状态回滚)同样只在末级设置;非末级 rank 嵌入下一步输入所需的 sampled-token 与 draft 值也只在末级。

## 提出的方案
核心不变式:非末级 PP rank 应像 last rank 一样应用 sampler 广播的 per-request 有效/接受计数。五个小块加一个实验性门控:
1. Draft 内存使能(int4 draft-embed 量化 + 跳过 draft lm_head),让大模型的 draft 能放进 last rank
2. pp_spec_broadcast.py:类型化、宽度无关地向非末级 rank 广播 sampler 的 per-request tokens/counts
3. 非末级 rank 输入重建(把真实广播值写入 token_ids_cpu、散布 draft tokens)——修复崩溃的部分
4. 非末级 rank 的 num_computed_tokens 漂移修正
5. hybrid GDN 的 num_accepted_tokens 回滚(以 is_hybrid 门控)
6. 一次性 warning_once,在成熟前把 MTP + PP>1 标为实验性
V1/V2:作者认为 V2(MRV2)是 spec-under-PP 的长期归宿(#42538、#43732 是方向),但 V2 目前在 MiMo PP=2 + MTP 下引擎构建死锁(EngineCore 的 shm_broadcast 卡住),尚不覆盖该配置,故建议先落 V1,后续移植到 V2 的 PPHandler。

## 优化点与收益
在当前 main 上验证:MiMo(纯注意力)与 Qwen3.5-27B(hybrid GDN),num_speculative_tokens 1-3,含 mamba_cache_mode=align 与 fp8 KV cache;1.68-1.89x speedup;27B 上约 94% 接受率;batch=16 与 256-token 运行保持浮点无关不变式(无崩溃、长度精确);残余 token 分歧是 MTP 投机解码的浮点近似平局,非 PP 特有。

## 现状与落地
🟢 open,反馈期 1 周;实现 PR 已就绪并在 issue 中引用,作者愿意按维护者意见调整、拆分或对齐 V2。规模约 544 行非测试代码,超出 500 行指引,作者询问是否把 draft-memory 部分(约 190 行)拆出。正文说明诊断与起草使用了 AI 辅助,作者逐行复核并自行运行测试。

## 关联
- #36643、#36872、#14044、#42538、#40768(作者核实 #40768 非硬依赖)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
