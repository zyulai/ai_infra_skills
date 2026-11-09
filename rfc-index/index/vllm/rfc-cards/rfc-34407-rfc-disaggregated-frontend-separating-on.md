# RFC #34407: [RFC]: Disaggregated Frontend — Separating Online Serving from Engine

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34407](https://github.com/vllm-project/vllm/issues/34407) |
| 作者 | hyeongyun0916 |
| 创建日期 | 2026-02-12 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 5 · 💬 10 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议两阶段拆分 vLLM:先把前端与 GPU 引擎分离,再在服务层内拆出 renderer,实现无 GPU 前端部署与纯 token-in/token-out 引擎。

## 提案动机
本 RFC 是 RFC #22817(Disaggregated Everything - Token In <> Token Out API Server)的后续,并建立在 #22880 之上。基于 `#sig-frontend` 的讨论,目标是:前端(预处理 tokenization/MM 输入处理,后处理 detokenization/tool call 解析/reasoning 解析)可无 GPU 部署;支持 llm-d、Dynamo 及自定义前端等只需复用 vLLM 预处理逻辑而无需跑完整推理引擎的场景;让 engine 成为与请求预处理解耦的纯 token-in/token-out 服务。

## 提出的方案
采用两阶段拆分,引入新的 CLI 子命令(而非像 PR #33000 那样给 `vllm serve` 加 flag):
- **Phase 1:`vllm online` | `vllm engine`**——把在线服务前端与 GPU 推理引擎分开;两者经 In-proc(默认,同进程部署)或 RPC(拆分部署)通信。前端层无 GPU 运行、承担全部前/后处理,engine 只做 GPU 推理;同进程部署时 `vllm serve` 行为与现在完全一致、零额外开销。
- **Phase 2:`vllm frontend` | `vllm renderer`**——在在线服务层内部进一步拆分;命名沿用 engine client/engine core 的惯例,轻的部分叫 renderer client(位于 frontend 内),重的部分叫 renderer core(位于 renderer 内);同样支持 In-proc(默认)或 RPC。
- **默认行为**:所有组件默认同进程共存(等价于现在的 `vllm serve`),拆分是 opt-in。
- **协议正交**:FastAPI vs gRPC 与拆分层级正交,`vllm online`/`vllm frontend` 均支持 `--server <fastapi|grpc>`;现有接受 TokenizedInput 与 SamplingParams 的内部 gRPC server 保留为面向自行处理预处理的专用客户端的内层接口。

## 优化点与收益
正文未给出量化数字。收益定位:前端可无 GPU 部署、engine 与预处理解耦、支持 tokenization 拆分部署等场景;同进程部署不引入开销。

## 现状与落地
状态 open,labels 含 keep-open,正文未给出明确反馈期;处于提案讨论阶段。

## 关联
- [RFC #22817](https://github.com/vllm-project/vllm/issues/22817)
- [PR #33000: vllm serve --disable-inference](https://github.com/vllm-project/vllm/pull/33000)
- [PR #33693: Renderer architecture draft](https://github.com/vllm-project/vllm/pull/33693)
- [RFC #32713: Unified Parser](https://github.com/vllm-project/vllm/issues/32713)
- https://github.com/vllm-project/vllm/issues/22880

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
