# PR #46043: [Perf] diffusion/registry: Reduce model discovery work at startup

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-24 |
| 分类 | Other (tier: major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用。PR 正文注明:生产代码 1 个文件 +50/-52,测试 +27 行 |
| 链接 | https://github.com/sgl-project/sglang/pull/46043 |

## 一句话结论

diffusion 模型注册表启动发现提速约 43%(664ms→377ms),通过跳过无关文件与表达式子树减少 AST 解析工作量。

## 动机

多模态 diffusion 服务启动时,模型发现(registry discovery)会对每个组件文件做完整 AST 解析,并遍历每个模型 forward 实现中的所有表达式节点;而注册其实只需要类定义与赋值语句。模型库越大,启动时被浪费的解析与遍历开销越多。

## 优化点

三层剪枝,均不改变注册语义:

1. **文件级短路**:读取源码后先做 `"EntryClass" not in source` 快速字符串检查,不含 EntryClass 的辅助文件直接跳过,不做 `ast.parse`。
2. **表达式子树剪枝**:遍历从 `ast.walk` 改为手动 BFS deque;遇到 `ast.expr` 节点直接丢弃、不再展开其子节点——表达式里不可能含 class/assign 声明,thus 跳过 forward 内的张量计算表达 subtree,同时保持声明与别名的广度优先顺序(含嵌套在 `if` 等语句内的声明)。
3. **注册分支扁平化**:EntryClass 支持 `Name` / `List` / `Tuple` 三种格式的解析逻辑重写为统一的 entries 迭代,行为不变。

无新运行时 flag、无持久缓存、不导入模型;pipeline discovery 与外部包注册行为保持原样,与 #43317 的 CLI backend-detection 工作相互独立。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/registry.py: 唯一生产代码文件。
  _discover_and_register_models() 增加 EntryClass 字符串预过滤;BFS 遍历
  跳过 ast.expr 子树;扁平化 EntryClass 注册分支
python/sglang/multimodal_gen/test/unit/test_registry.py: 新增
  test_model_discovery_preserves_nested_declarations,验证跳过表达式子树后
  嵌套声明/别名/单值·list·tuple 入口格式均保留
```

关键片段(diff 摘录):

```python
nodes = deque([tree])
while nodes:
    node = nodes.popleft()
    if isinstance(node, ast.expr):
        continue
    nodes.extend(ast.iter_child_nodes(node))
```

## 性能声明

PR 正文测试条件:专用双 H200 环境(Xeon Platinum 8558P,约 2 TiB RAM,Python 3.12.15,PyTorch 2.14.1+cu130,Diffusers 0.37.0,Transformers 5.19.0),对比 main `4b273537dec5` 与本 PR `41558c352e38`:

| 测量项 | Before | After |
| --- | ---: | ---: |
| Registry discovery,20 次交错采样中位数 | 664 ms | 377 ms |
| 单 GPU 进程启动到 HTTP 就绪,3 组配对样本中位数 | 30.26 s | 29.39 s |
| Ulysses2 进程启动到 HTTP 就绪,3 组配对样本中位数 | 30.02 s | 29.61 s |

PR 明确强调:发现提速约 43%,**不是** 43% 的服务器启动提升;启动测量排除模型下载与 warmup,不含首请求 JIT 时间。110 个模型与 18 个别名的完整映射不变;12 次 Wan2.1-T2V-1.3B-Diffusers 真实服务启动,4-step/24-step 请求(512x320、17 帧、seed 42)的 24 个输出解码帧哈希前后一致。

## 验证与回归风险

- 新增测试覆盖嵌套声明、别名与单/list/tuple 入口格式,且不导入可选模型依赖;`test_registry.py`、`test_external_model_package.py`、`test_plain_component_loader.py` 共 24 个测试通过,pre-commit 通过。
- CI 状态(pack 记录):PR Test (Base) 通过,PR Test (Extra) 失败,AMD ROCm 10 失败。
- 回归面:AST 遍历剪枝若误伤非常规写法(如 EntryClass 出现在非常见语句结构中)会漏注册模型;表达 subtree 跳过依赖"表达式内无声明"这一前提。发现结果逐一映射未变 + 帧哈希一致降低了该风险。

## 关联

#43317(CLI backend-detection,PR 正文声明两者独立)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
