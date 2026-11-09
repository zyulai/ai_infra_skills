# PR #29514: [diffusion] fix --warmup silently downgrading server-based warmup to request mode

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29514](https://github.com/sgl-project/sglang/pull/29514) |
| 作者 | mickqian |
| 合并日期 | 2026-06-28 |
| labels | run-ci, diffusion |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

`serve --warmup` 注入的 `warmup_mode="server"` 因 default_args 不算显式参数而被丢弃,server 预热静默降级甚至整体失效,首请求吃满约 2 倍冷启动延迟。

## 问题现象

`sglang serve --warmup` 本应使用 server-based 预热(`serve` 入口为生产环境注入默认 `warmup_mode="server"`),却静默降级为 request-based 预热;与 `--warmup-resolutions` 组合时预热被整体禁用——`--warmup` 实际上把预热关掉了,第一个真实请求要付完整的 eager 冷启动代价,diffusion 基准中观测到约 2 倍延迟(如 Z-Image-Turbo 无预热 13.46s,正常约 0.74s)。

## 根因

正文披露:`serve` 入口(`runtime/entrypoints/cli/serve.py`)经 `default_args` 注入 `warmup_mode="server"`,但 `default_args` 不会记入 `_explicit_arg_names`。于是 `_adjust_warmup`(`runtime/server_args.py`)中,当 `--warmup` 显式(`legacy_explicit=True`)而 `--warmup-mode` 不显式(`mode_explicit=False`)时,守卫 `if mode_explicit or not legacy_explicit:` 求值为 False,注入的 `warmup_mode="server"` 永不生效,`server_warmup` 回落到 dataclass 默认 False(即 request 模式)。再叠加:`--warmup --warmup-resolutions WxH` 成为死区——合成 server 预热要求 `server_warmup=True` 才跑,request 预热在 `warmup_resolutions is not None` 时又直接跳过,结果任何预热都不执行。

## 修复方案

`python/sglang/multimodal_gen/runtime/server_args.py` 的 `_adjust_warmup`:在原守卫之后增加 `elif self.warmup: self.server_warmup = self.server_warmup or self.warmup_mode == "server"` 分支——legacy `--warmup`/`--server-warmup` 开启预热且未显式指定 `--warmup-mode` 时,尊重解析出的 `warmup_mode`(如 `serve` 注入的 `"server"`)来决定 server/request 分派,不再静默停留在 request 模式;`--warmup false` 保持关闭不变。新增 `TestWarmupModeNormalization` 两个回归用例(legacy `--warmup` + 默认 server 模式 → `server_warmup=True`;`--warmup --warmup-resolutions` 死区 → server 预热执行)。

关键 diff 摘录:

```diff
             if mode_explicit or not legacy_explicit:
                 self.warmup = self.warmup_mode != "off"
                 self.server_warmup = self.warmup_mode == "server"
+            elif self.warmup:
+                self.server_warmup = self.server_warmup or self.warmup_mode == "server"
```

H100 验证:修复后 `serve --warmup` 跑 server-based 预热,基准延迟回到客户端预热基线——Z-Image-Turbo 0.74s(基线 0.65s)、FLUX.1-dev 4.80s(基线 4.68s)、Ideogram-4 5.26s(基线 5.19s)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 `_adjust_warmup` 的参数归一路径新增一个分支,显式指定 `--warmup-mode`、`--warmup false`、disagg、bare `serve` 等既有路径行为不变(正文确认相关旧测试仍通过);效果面是 diffusion serve 默认入口的预热方式由 request 模式恢复为 server 模式。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
