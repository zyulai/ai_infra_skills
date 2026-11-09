# PR #33374: [AMD][DI][CI] Auto-mount  latest host ionic/ibverbs userspace in the MI355X nightly launcher (fix ABI mismatch)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33374](https://github.com/sgl-project/sglang/pull/33374) |
| 作者 | Lzy17 |
| 合并日期 | 2026-08-04 |
| labels | amd |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
MI355X nightly launcher 未挂载 host ionic/ibverbs userspace,ABI 不匹配致节点 RDMA 失败被剔除;移植 disagg 容器脚本的挂载集修复。

## 问题现象
当节点的 host ionic provider 比容器内 rdma-core 新时,容器内 libibverbs 拒绝该 provider,报 `Driver ionic does not support the kernel ABI ... No IB devices found`,RDMA 失败,该节点被移出 disagg nightly 池(在 `mia1-p01-g20` 上观察到)。此前该节点靠 workflow 中的 `SLURM_EXCLUDE: mia1-p01-g20` 排除,被删注释还提到 MORI 会报 "no active RDMA device"、disagg server 在 init 时挂掉。

## 根因
正文披露:`launch_mi355x.sh` 的 `docker run` 只挂载了 `/it-share` 与 `$HOME`,host 与容器之间 ionic/ibverbs userspace 的 ABI 不匹配因此无法弥合;disagg 容器脚本(`scripts/ci/amd/amd_ci_start_container_disagg.sh`)已通过把 host 的 ionic/ibverbs userspace 挂载进容器解决此问题,但 Slurm launcher 没有同样的挂载集。

## 修复方案
把 disagg 容器脚本已有的挂载集移植到 Slurm launcher:
- `scripts/ci/slurm/launch_mi355x.sh`:生成 `$WORKDIR/ionic_mounts.sh`(literal heredoc),解析 host 的 ionic provider、provider 目录、`/etc/libibverbs.d` 及 libnl/libmnl,导出 `IONIC_MOUNTS`;prefill/decode/bench 三个 role 脚本在 `docker run` 前 `source` 它并传 `${IONIC_MOUNTS[@]}`。
- `.github/workflows/nightly-amd-mi355x-disagg.yml`:删除 `SLURM_EXCLUDE: mia1-p01-g20`,节点重新入池。

正文说明检测的设计约束:检测在 compute node 上运行(ionic `.so` 版本是节点特定的);有存在性守卫,无 ionic 的 host 上 no-op;只挂载 provider 而非核心 `libibverbs.so`;`DOCKER_COMMON`/`EXTRA_DOCKER_ARGS` 保持不变。

```diff
-docker run $DOCKER_COMMON --name mi355x_prefill \
+docker run $DOCKER_COMMON "\${IONIC_MOUNTS[@]}" --name mi355x_prefill \
```

## 影响与回归风险
- 回归面:仅 MI355X nightly disagg 的 Slurm 启动路径(launch_mi355x.sh 与 nightly workflow);容器挂载参数增加为只读挂载,无 ionic 的 host 上行为不变(存在性守卫,正文);p02-only run 预期无回归(测试计划)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
