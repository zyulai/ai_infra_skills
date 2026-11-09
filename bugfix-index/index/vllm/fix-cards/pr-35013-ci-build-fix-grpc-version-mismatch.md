# PR #35013: [CI/Build] Fix gRPC version mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35013](https://github.com/vllm-project/vllm/pull/35013) |
| 作者 | DarkLight1337 |
| 合并日期 | 2026-02-21 |
| labels | rocm, ready, ci/build, ci-failure |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

gRPC 三个相关库版本不一致致 CI 失败,统一钉为 1.78.0 修复。

## 问题现象

正文只链接了一个 Buildkite CI 失败记录(vllm/ci build #52563),未描述具体报错信息或复现步骤。标签含 `ci-failure`,说明是 CI 环境中 gRPC 相关依赖版本不匹配导致的安装/运行失败。

## 根因

正文未以文字披露根因,仅给出 CI 失败链接。(由 diff 推断)diff 中新增的注释写明 "The version of gRPC libraries should be consistent with each other"——即 grpcio、grpcio-reflection、grpcio-tools 三个库必须版本一致;此前 requirements 里只钉了 `grpcio-tools==1.78.0`(注释 "Should match `build.txt`"),grpcio 与 grpcio-reflection 未钉版本,安装解析出来的版本与 tools 不匹配,引发版本冲突。

## 修复方案

统一钉版三个 gRPC 库:

- `requirements/rocm.txt`:删除单独的 `grpcio-tools==1.78.0` 行,在文件头部新增一段:
- `requirements/test.in`:同样把单独的 `grpcio-tools==1.78.0` 替换为三行统一钉版。
- `requirements/test.txt`:由锁文件再生成,新增 `grpcio-reflection==1.78.0` 及其依赖来源注释(protobuf 同时被 grpcio-reflection 依赖)。

关键 diff 摘录(requirements/test.in):

```diff
-grpcio-tools==1.78.0 # Should match `build.txt`
+# The version of gRPC libraries should be consistent with each other
+grpcio==1.78.0
+grpcio-reflection==1.78.0
+grpcio-tools==1.78.0
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:仅依赖清单(ROCm 与测试环境),不影响引擎代码。统一钉版后,任何需要更高/更低 gRPC 版本的传递依赖未来可能产生新的冲突;若 `build.txt` 中的 gRPC 版本后续变动,需同步更新这三处钉版。

## 关联

- [Buildkite CI build #52563](https://buildkite.com/vllm/ci/builds/52563/steps/canvas?sid=019c7f00-c4af-4251-b271-e948e7108ba9&tab=output)(正文链接的失败记录)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
