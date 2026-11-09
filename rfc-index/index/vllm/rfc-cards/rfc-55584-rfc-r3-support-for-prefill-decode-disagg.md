# RFC #55584: [RFC]: R3 support for prefill/decode disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55584](https://github.com/vllm-project/vllm/issues/55584) |
| 作者 | ShuoleiWang |
| 创建日期 | 2026-09-06 |
| 状态 | 🟢 open |
| labels | RFC, kv-connector |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Disaggregated Serving |

## 一句话结论
在 P/D 分离下用 NIXL 原生传输 R3 路由记录(P→D),保证 Decode 用于 RL 回放的 expert 路由与其实际使用的 KV 精确对应。

## 提案动机
MoE 模型的 RL rollout 需要记录每个 token 实际运行的 expert;vLLM 已能捕获、返回 R3 记录并可配合本地前缀缓存复用。但 P/D 分离时 D 从 P 加载 KV、不重跑全部 prompt token,仅靠自身捕获无法得到完整 prompt 路由。关键规则:路由必须匹配 D 实际使用的 KV,而非仅同 prompt 文本 —— 相同前缀哈希不证明两次执行选了完全相同的路由。48-token prompt 示例:`[0, 32)` 用 D 本地缓存 KV 及其旧路由、`[32, 47)` 用 P 的 KV 及其路由、`[47, 48)` 用 D 新捕获;直接用 P 当前 trace 覆盖所有前缀路由会破坏这一对应。

## 提出的方案
- 作为 PR #45635 已接受的 worker-owned、按 block key 查找的 R3 存储的 PD 扩展(RFC #45221 讨论工作的 PD 延伸),不是第二个 R3 store 或新的通用 Artifact Connector 设计。
- 新增 `KVTransferConfig.enable_routed_experts_transfer`(默认 false),两侧沿用 `--enable-return-routed-experts`;握手带 R3 能力与 data-format 版本,单侧开启或版本/布局不匹配必须失败,不得降级为警告或返回不完整 trace。
- 每个 KV 区间携带指向匹配路由记录的 reference;前缀哈希只用于找可能命中,NIXL 不得凭哈希任选记录;导出用 transfer lease + manifest(绑定源实例、`transfer_id`、cache epoch、模型/权重身份、格式与有效 token 范围),已发布记录不可变,D 校验后经 store 公共 API commit。
- 代理保持普通 PD handoff(至多带小 export ID);路由数据由 worker 直传,不以 JSON 数组走 `kv_transfer_params`。
- 首版范围:MRV2、纯文本 MoE、非流式 completions、`n=1`、DP=1、P/D 匹配 TP 布局;默认禁用;不支持 MTP/DBO/LoRA/多模态/PP/DCP 等;失败遵循 `kv_load_failure_policy="fail"`;快照字节上限等于该 worker 的 R3 store 容量,在途窗口 ≤ `2 * max_num_seqs`。

## 优化点与收益
正文未给出量化数字 —— 收益是正确性/能力:P/D 分离下 RL 训练能拿到与 D 实际使用的 KV 匹配的完整路由记录,且保持前缀缓存、chunked prefill、CUDA graphs、async output 全部开启,不为正确性关特性。

## 现状与落地
仍在讨论/等待评审:open 状态、0 评论;正文明确"征求对原生 NIXL 路径及如何与现有工作配合的同意",不假定取代已有 PR;依赖基础存储改动(PR #45635 或后继)先合入,PD 特性随后作为单个完整 PR 提交。注:pack 正文在 20k 字符处截断,合并标准表格后半未收录,本卡基于现有内容成卡。

## 关联
- [PR #45635(R3 worker-owned 存储基座)](https://github.com/vllm-project/vllm/pull/45635)
- [RFC #45221(R3 相关讨论)](https://github.com/vllm-project/vllm/issues/45221)
- [PR #39289(kv_transfer_params 携带路由的同类工作)](https://github.com/vllm-project/vllm/pull/39289)
- [PR #45419(放宽配置检查 + proxy 合并)](https://github.com/vllm-project/vllm/pull/45419)
- PR #52902(R3 测试重组)与 PR #37094(输出范围相关工作)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
