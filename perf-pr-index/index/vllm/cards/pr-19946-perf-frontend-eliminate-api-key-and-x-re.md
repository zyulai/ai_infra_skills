# PR #19946: [Perf][Frontend]: eliminate api_key and x_request_id headers middleware overhead

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Yazan-Sharaya |
| 合并 | 2025-06-27 |
| 分类 | Frontend / Serving (tier: major) |
| Label | documentation, frontend, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/19946 |

## 一句话结论

把 api_key / X-Request-Id 的 FastAPI 函数式 middleware 改成 pure ASGI,启用两个 middleware 的 QPS 损失从 9.8% 降到 0.7%。

## 动机

当前处理认证(`--api-key`)与 X-Request-Id headers(`--enable-request-id-headers`)的 middleware 使用 FastAPI 提供的高层抽象(把 middleware 写成可访问 `Request` 对象的函数),正文称一旦安装该 middleware 就带来不可接受的性能损失。PR 用 pure ASGI middleware 重实现同样功能,使开销几乎为零;并 Fixes #11529。

## 优化点

把两个可选 middleware 从 FastAPI 的函数式写法(依赖 `Request` 对象构造)改为 pure ASGI middleware(直接包 ASGI app、按 scope 处理),保持功能不变:对 `/v1` 开头的路径校验 `Authorization: Bearer <api-key>`(非 `/v1` 路径如 `health` 跳过校验)、开启 `--enable-request-id-headers` 时注入 32 字符的 `X-Request-Id`。同时移除文档里警告 `--enable-request-id-headers` 在高 QPS 下性能损失显著的段落,以及 `cli_args.py` 中 `parser.add_argument("--enable-request-id-headers")` 帮助文本里的 "Caution:" 段(两者因不再适用而删除)。新增 `tests/entrypoints/openai/test_optional_middleware.py` 覆盖这两个默认关闭、由 server 参数开关的 middleware。

## 关键改动

```text
vllm/entrypoints/openai/api_server.py: 核心。把 api_key / X-Request-Id 的 FastAPI 函数式 middleware 改为 pure ASGI middleware
vllm/entrypoints/openai/cli_args.py: 移除 --enable-request-id-headers help text 的 "Caution:" 性能警告段
tests/entrypoints/openai/test_optional_middleware.py: 新增,用 intfloat/multilingual-e5-small embed 模型验证无/有 api_key、缺/带 token、非 /v1 路径、X-Request-Id 注入等行为
docs/serving/openai_compatible_server.md: 删除高 QPS 下 header 性能警告段
```

关键代码(新增测试对 X-Request-Id 的断言):

```python
async def test_enable_request_id_header(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))
    assert "X-Request-Id" in response.headers
    assert len(response.headers.get("X-Request-Id", "")) == 32
```

## 性能声明

正文称测试用 `vllm bench serve` 取端到端结果,"~10% decrease in throughput is huge for the little computation that is actually done when enabling both middleware"(指旧实现)。QPS 表(Current implementation):without middleware 94 - 94 - 96 - 96 - 94,平均 94.8;`--api-key` 90 - 92 - 91 - 90 - 90,平均 90.6,差 4.4%;`--api-key` 且 `--enable-request-id-headers` 87 - 86 - 86 - 84 - 84 - 86,平均 85.5,差 9.8%。Pure ASGI:without middleware 93 - 95 - 95 - 93 - 93 - 95,平均 94.0;`--api-key` 93 - 95 - 94 - 93 - 93 - 95,平均 93.8,差 0.2%;两者 93 - 93 - 95 - 93 - 92 - 94,平均 93.3,差 0.7%。PR 正文未注明该 benchmark 的模型、硬件、精度或 workload 细节。

## 验证与回归风险

Test Plan 为新建测试文件验证 middleware 行为(正文称此前无测试)并 benchmark 前后性能;Test Result 称 "All the correctness tests pass as expected"。新增测试覆盖无 token / 缺 token(401)/ 带 token(200)/ 非 `/v1` 路径跳过校验 / X-Request-Id 存在且长度 32 等,用小 embedding 模型 `intfloat/multilingual-e5-small`(`--task embed --dtype float16 --max-model-len 512 --enforce-eager --max-num-seqs 2`)加速 CI。风险面:改动 OpenAI-compatible server 的 middleware 安装路径,影响所有带 `--api-key` 或 `--enable-request-id-headers` 的部署;文档与 CLI 帮助文本同步删除性能警告;正文未列其他 serving 路径或高并发之外的回归测试。

## 关联

PR 正文明确:Fixes #11529(文档旧注释中亦以 `pull/11529` 引用该 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
