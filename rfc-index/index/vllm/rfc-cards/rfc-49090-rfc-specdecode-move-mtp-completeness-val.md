# RFC #49090: [RFC][SpecDecode] Move MTP completeness validation to the weight-update transaction boundary

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49090](https://github.com/vllm-project/vllm/issues/49090) |
| 作者 | aoshen02 |
| 创建日期 | 2026-07-19 |
| 状态 | 🟢 open |
| labels | stale |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
把 MTP checkpoint 完整性校验从单次 `load_weights()` 移到权重更新事务边界,消除 RL 分桶更新下的假失败。

## 提案动机
多个 vLLM MTP 实现在 `model.load_weights()` 内做 checkpoint 完整性校验,默认一次调用收到完整 checkpoint。而 RL 权重传输的 IPC/NCCL 会把一次逻辑更新拆成多个 transport bucket、逐桶调用 `load_weights()`,于是在第一个部分桶上就报出假的 missing-layer / missing-parameter 错误,即使完整更新其实包含全部所需权重。

## 提出的方案
- 关注点分离:单条目即时校验(name、shape、shard mapping、divisibility、dtype/量化元数据)留在 `load_weights()`;逻辑 checkpoint/update 的完整性校验移到事务边界,在所有桶记账后只跑一次。
- 短期修复(PR #49175):IPC/NCCL 仅对增量 `load_weights()` 包一层 scoped 的 MTP completeness-suppression 上下文,成功或异常都自动恢复;保留启动期校验与逐条目检查,也不声称 RL 更新已有聚合校验。
- 目标架构:待 #48908 的 `WeightLoadSession` 生命周期重构落地后,启动加载与 RL reload 统一为 `prepare() -> load -> finish()` 事务,提供 `record_loaded_items()` / `register_validator()`;RL 更新在 `finish()` 中对跨桶 union 的 ID 集做一次 `expected_ids - loaded_ids` 校验。
- 校验以 canonical weight/parameter-ID 粒度进行,manifest 须覆盖 fused 参数、TP/PP/EP 本地归属、shared/tied 参数、可选 scale、量化映射、checkpoint alias(如共享 MTP head/embedding)等;`set(model.named_parameters())` 不是充分的通用 manifest。
- 受影响 loader 各用三种"完整性"口径:DeepSeek、DeepSeek V3.2/GLM-5.2、DeepSeek V4、MiniMax M3、Bailing(MTP 层覆盖),Step3.5(精确参数覆盖),Inkling(必需子集);正文给出 6 步后续 PR 计划,最终把聚合校验扩展到主模型 RL 更新。

## 优化点与收益
正文未给出量化数字 —— 收益是正确性:消除 RL 分桶更新下的假失败,并让启动加载与 RL reload 获得等价的事务级完整性保证;短期修复不解决静默忽略/错误归一化的 checkpoint key(如 #43456 一类)。

## 现状与落地
仍在讨论/已有原型:open 状态;短期修复已挂 PR #49175(开发分支 aoshen02/vllm#4),事务级方案依赖 #48908 先落地,后续迁移 MTP loader 后再移除临时 suppression;labels 含 stale。文末注明调查与撰写使用了 AI 辅助。

## 关联
- [PR #49175(短期修复)](https://github.com/vllm-project/vllm/pull/49175)
- [PR #48908(WeightLoadSession 生命周期)](https://github.com/vllm-project/vllm/pull/48908)
- [aoshen02/vllm#4(开发 PR)](https://github.com/aoshen02/vllm/pull/4)
- #43456(DeepSeek V4 alias 缺失需另行修复,正文未附链接)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
