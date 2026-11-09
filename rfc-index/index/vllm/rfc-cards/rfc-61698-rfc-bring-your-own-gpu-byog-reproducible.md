# RFC #61698: [RFC]: Bring-Your-Own-GPU (BYOG) reproducible evals and benchmarks for PRs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61698](https://github.com/vllm-project/vllm/issues/61698) |
| 作者 | LucasWilkinson |
| 创建日期 | 2026-10-14 |
| 状态 | 🟢 open |
| labels | RFC, quantization, rocm |
| 热度 | 👍 15 · 💬 17 |
| 分类 | CI / Infra / Process |

## 一句话结论
由 CI 构建签名 eval 镜像,社区贡献者用自己的 GPU 跑可复现的评测与基准,结果进公共数据库辅助 PR 评审。

## 提案动机
性能与精度敏感 PR 的评审常卡在证据与回归测试上:评审者反复要"e2e 性能数据""eval 结果""H100 上的 8k1k 数字",请求模糊、回复常是截图或手工运行输出,甚至没有。在 vLLM 出资的 CI 上为 PR 跑全量 eval+benchmark 矩阵不可扩展,尤其在 AI 生成 PR 增多的背景下,维护者注意力才是稀缺资源。同时社区拥有的数据中心硬件比 vLLM CI 更多、种类更全(GB300、MI355X、老 Hopper/Ampere 机群等)。

## 提出的方案
- CI 在 commit 上构建标准、版本化的签名 eval 镜像(cosign 签名 + SLSA provenance,按 digest 拉取);任何人一条 `docker run` 在自己 GPU 上运行,产出结果 bundle 上传公共结果数据库
- Run spec(YAML)完整定义一次运行(模型、`vllm serve` 参数、硬件、eval/benchmark);request 的 matrix 展开为多个 job,支持 exclude/include、并行速记(tp4-dp2-ep 等)、展开前严格校验
- Eval 套件:`smoke`(gsm8k,minutes)/ `standard`(默认;gsm8k、mmlu_pro 300 per subject、mrcr 120 samples,up to ~3 hours)/ `extended`(评审要求;gsm8k、full mmlu_pro、mrcr 240 samples,hours);数据集固定 commit 烘入镜像;另设 eval 目录(logprob/KL 检查、ifeval、chartqa/mmmu、gpqa_diamond、aime25 等)
- Benchmark 套件(仿 SemiAnalysis InferenceX):`smoke`(1k1k concurrency 1、64)/ `standard`(8k1k c1/16/64/256 + AgentX 8、32,up to ~2 hours)/ `extended`(1k1k、8k1k、1k8k c1–512 + AgentX,hours to days);保留每请求原始记录;默认报告自然投机解码接受率
- 结果可与 `main` 同 GPU 类型同 spec 的结果比较,明显低于 main 的结果被标记可疑;信任制+问责:上传经 GitHub 设备流登录(无 scope token),记录上传者身份与 bundle sha256,篡改者账号被标记
- 非目标:不替代 pre-merge CI 与现有 post-merge 性能看板;暂不做硬合并门禁(先信息性,原型可靠后再议);正文列有 8 个开放问题

## 优化点与收益
正文未给出量化性能数字。文中数字均为成本/规模描述:standard eval 套件「up to ~3 hours」、standard benchmark 套件「up to ~2 hours」、示例 request 展开为 13 jobs 约 329 GPU-hours(standard 套件下)。定性收益:评审证据标准化、可复现,把产出证据的成本移回希望合并 PR 的一方,并可利用社区闲余硬件。

## 现状与落地
仍在讨论阶段(open)。正文给出三步 rollout:MVP(Dockerfile.byog、tools/byog harness、套件与参考 specs、最小结果服务)→ 服务功能(公共看板与 PR bot)→ 未来可能的硬门禁;多处(套件内容、时间预算、图像保留策略、AMD/TPU/XPU 支持顺序等)在征集社区意见。
注:pack 正文在 20k 字符处截断(截断于「Prior art」一节),本卡基于现有内容成卡。

## 关联
- #59665(fast-track merging RFC)
- SemiAnalysis InferenceX / AgentX(Prior art,正文此处被截断)
- MLPerf Inference(Precedent,惯例参照)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
