# PR #24234: [docker] Fix silently-masked cubin download failure; skip prebuilt cubins on aarch64

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24234](https://github.com/sgl-project/sglang/pull/24234) |
| 作者 | mispa-ms |
| 合并日期 | 2026-05-04 |
| labels | (none) |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Dockerfile 中 `|| true` 因 bash 优先级吞掉 cubin 下载失败检查,导致 aarch64 nightly 构建静默失败;改为清理命令单独包子shell并在 aarch64 跳过下载。

## 问题现象
自 #22160(2026-04-09)起,aarch64 nightly Docker 构建一直被静默破坏,已在 NVIDIA 内部 nightly 流水线观察到数周;x86_64 构建不受影响(能找到匹配的预编译 cubin 变体)。典型日志:`kernels download python` 3 次重试全部报 "Cannot find a build variant for this system in kernels-community/sgl-flash-attn3"(仅有 x86_64 变体),但 RUN 层仍报 `DONE 115.4s` 成功;随后 `COPY --from=framework_final /root/.cache/sglang /root/.cache/sglang` 在两阶段之后才报误导性的 `"/root/.cache/sglang": not found` 构建失败。

## 根因
正文披露的根因是两个叠加的问题:

1. **静默失败 bug**:`docker/Dockerfile` 的 `framework_final` 阶段中,cubin 下载重试块末尾的 `... || true`(本意只想保护末尾的 `find` 清理命令),但由于 bash `&&`/`||` 左结合的优先级,把整条链都吞掉了——包括同日 #22322 加上的 `[ "$success" = "1" ]` 快速失败检查。下载永久失败时 RUN 仍报 `DONE 0`,后续 `mkdir -p /root/.cache/sglang && mv python/kernels.lock` 被跳过,最终在 runtime 阶段的 COPY 才失败。
2. **没有 aarch64 cubin 变体**:`python/pyproject.toml` 中 `[tool.kernels.dependencies]` 固定的 `kernels-community/sgl-flash-attn3` 只发布 `x86_64-linux` 变体,在 aarch64 主机(如 Grace Blackwell 构建)上 3 次重试必然全部失败,而上述静默失败 bug 把它藏进 `DONE 0` 的层里。

## 修复方案
只改 `docker/Dockerfile` 的 `framework_final` 可编辑安装 RUN 块(`pyproject.toml` 有意不动,否则会破坏能用预编译 cubin 的 x86):

1. 把末尾 `find ... -exec rm -rf {} +` 包进子shell,使 `|| true` 只作用于清理命令,不再吞掉前面的失败——真正的静默失败修复。
2. aarch64 上显式跳过 cubin 下载并打印日志(说明 `kernels-community/sgl-flash-attn3` 无变体、运行时 JIT 编译),附注释标记上游发布 arm cubin 后移除该分支。
3. 防御性地 `mkdir -p /root/.cache/huggingface /root/.cache/sglang`(此前 HF 缓存目录是 `kernels download python` 的副作用产物,aarch64 跳过后相邻 COPY 会同样失败)。
4. 把 `[ -f kernels.lock ] && mv ... || true` 改为 `if [ -f ... ]; then mv ...; fi`,真实的 mv 错误(文件系统、权限)会大声失败。

关键 diff 摘录:

```diff
-    done; [ "$success" = "1" ] ) \
+             done; \
+         fi; \
+         [ "$success" = "1" ] ) \
-    && mkdir -p /root/.cache/sglang \
-    && mv python/kernels.lock /root/.cache/sglang/ \
-    && find ... -exec rm -rf {} + 2>/dev/null || true
+    && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
+    && ( if [ -f python/kernels.lock ]; then mv python/kernels.lock /root/.cache/sglang/; fi ) \
+    && ( find ... -exec rm -rf {} + 2>/dev/null || true )
```

(摘录有省略,aarch64 分支内含 `echo "Skipping kernels-community/sgl-flash-attn3 cubin download on aarch64 ..."` 提示并置 `success=1`。)Diff 规模:1 file changed, 18 insertions(+), 9 deletions(-)。作者将附上 NVIDIA 内部 aarch64 nightly 绿色构建链接验证。

## 影响与回归风险
- 影响窗口:自 #22160(2026-04-09)引入该 RUN 块起的 aarch64 Docker 构建。
- 回归面:仅影响 `docker/Dockerfile` 构建流程。aarch64 上跳过下载改为显式行为(运行时 JIT 编译内核,与此前静默路径的实际效果一致);x86_64 的重试与快速失败逻辑保持不变。若上游后续发布 arm cubin,需按注释移除跳过分支,否则 aarch64 将一直走 JIT。

## 关联
- #22160(引入问题的 PR)
- #22322(添加 fail-fast 检查的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
