# PR #51360: [Frontend] Add reusable TP1 initialized-engine snapshots

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @matteso1 |
| 合并 | 2026-09-21 |
| 分类 | Frontend / Serving(tier: major) |
| Label | documentation, frontend, ready, ci/build, deepseek, kv-connector, nvidia, DSv4 |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 10 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/51360 |

## 一句话结论

新增 `vllm snapshot create/inspect/restore`:TP1 引擎初始化状态经 CRIU 落盘、同机恢复,激活中位延迟降 62.7%(54.70 s → 20.38 s)。

## 动机

针对"无任何存活服务进程、由可信同机操作者在 HTTP bind 前恢复一个准备好的 Linux x86-64 TP1 引擎"这一事件:此前没有复用手段。RFC #34303 与 #37921/#37925 的 CUDA checkpoint 原语位于 CRIU 磁盘持久化之下,#35934/#46089 探索相关生命周期,#52132 只让前端在 EngineCore L3 sleep 期间存活,均不覆盖本场景。

## 优化点

`snapshot create` 初始化引擎并记录确定性 one-token oracle(token ID、text、sampled logprob),用既有 sleep level 2 生命周期释放模型权重与 KV cache 后 CRIU dump;`restore` 在绑定 HTTP 前重载状态并复现 oracle,不匹配则 fail closed。公共 controller 编排快照事务,私有 host runtime 负责 CRIU、PID/会话身份、TCP 检查、io_uring、GPU inventory、pidfd pinning、link remap 与回滚;manifest 为严格 frozen Pydantic 模型,环境值以确定性名字绑定指纹记录(API key 与超时控制排除,字面密钥脱敏);引擎持有打开的生成缓存文件被指纹化,缓存轮换使 restore 早期失败并指名文件。

## 关键改动

```text
vllm/entrypoints/cli/snapshot.py(新): CLI 子命令;create 校验 TP1/PP1/DP1/PCP1、Linux x86_64、无 spec decode/SSL/API key
vllm/snapshot/controller.py(新): create/restore 事务编排,oracle 不匹配 fail closed
vllm/snapshot/manifest.py(新): SnapshotManifest、身份校验、原子写入与私有路径校验
vllm/snapshot/runtime.py(新): LocalSnapshotTools,CRIU/cuda-checkpoint/nvidia-smi 主机操作与缓存指纹
```

## 性能声明

H100 PCIe 80GB、Qwen3-32B BF16 TP1(source 5dd888926f,模型与生成缓存 resident):普通 warm 进程启动中位 54.7029 s,snapshot restore 中位 20.3795 s,节省 34.3234 s(62.7%),每臂 n=3;artifact 5,879,914,496 bytes,创建 78.5275 s(n=1),第三次激活回本。RTX 4090 Laptop Qwen3-0.6B:restore 8 s(n=1);更早 head 同容器跨 docker stop/start 两次 restore 3.233 / 2.931 s。所有被接受试验 oracle 一致(token 12095 / " Paris")。

## 验证与回归风险

H200 验收(create→inspect→restore→容器重启→再 restore,CI 89681/90301);tests/entrypoints/launchers/test_launch_cli.py 等 87 passed;82 个 snapshot tests。限制:Linux x86-64 单 NVIDIA GPU TP1,同机且驱动/内核/Python/torch/vLLM/model revision 一致;root 或 passwordless sudo;artifact 含进程内存与密钥须私有;无断电持久性保证;不支持跨机迁移与 GPU 重定位。风险:CRIU 链路对内核/驱动版本敏感;restore 身份校验严格,环境漂移会拒绝恢复。

## 关联

同分类 Frontend / Serving:#44890(sleep/wake_up 语义)。相关前序:RFC #34303、#37921/#37925、#35934、#46089、#52132;通用 CLI 工作按维护者要求拆出,本 PR 仅注册 snapshot 命令(#54841 为独立工作)。同批 #55422(镜像预编译 bytecode 缩短启动,其 H200 验收使用 bytecode CI 镜像)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
