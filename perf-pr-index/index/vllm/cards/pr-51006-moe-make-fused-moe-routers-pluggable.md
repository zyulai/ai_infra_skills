# PR #51006: [MoE] Make fused MoE routers pluggable

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @freyfwt |
| 合并 | 2026-11-09 |
| 分类 | MoE(major) |
| Label | documentation, ready |
| Diff | +124/-1,3 files(素材未提供 diffstat,按 pack diff 计数) |
| 链接 | https://github.com/vllm-project/vllm/pull/51006 |

## 一句话结论

把 `FusedMoERouter` 接入 `PluggableLayer` 机制,OOT 插件可按具体类名替换 MoE router;路由 parity 与全模型等价性验证通过,正文明确不作 accuracy/performance claim。

## 动机

目标是让 fused MoE 的 router 可被 out-of-tree 实现整体替换,复用 `MoERunner` 与 `RoutedExperts` 已有的 `PluggableLayer` OOT 类注册机制,而不是新增 Platform hook 或再开一个 registry。未注册的 model-specific 类保持原实现,不影响现网路径。本 PR 更新自 #51005;EPLB lifecycle 相关工作留在 #49700 单独处理。

## 优化点

`FusedMoERouter` 由纯 ABC 改为 `PluggableLayer` + ABC,工厂在构造期按具体类名做 OOT dispatch:用 `FusedTopKRouter.register_oot(MyFusedTopKRouter)` 这类注册在模型构造前生效。替换契约:必须 subclass 被替换类并保留其构造参数、routing 语义、model-specific weight loader 与可选 EPLB state;对 `MoERunner`/`RoutedExperts` 的注册不会自动替换 model-specific 子类或 `SubstitutedRoutedExperts`,需单独注册;显式传给 `FusedMoEFactory` 的 router 实例直接使用。router 现在是 PyTorch module,共享的 correction-bias `Parameter` 会在 `state_dict()` 中出现别名——插件需保留共享参数对象与模型 canonical checkpoint 名,通过 gate 加载 bias 的模型要先注册 gate 再构造 MoE 层。`docs/design/plugin_system.md` 写入上述注册/替换契约。

```python
-class FusedMoERouter(ABC):
+class FusedMoERouter(PluggableLayer, ABC):
     def __init__(self, eplb_state: EplbLayerState | None = None):
+        super().__init__()
         self._routing_replay_out: torch.Tensor | None = None
```

## 关键改动

```text
vllm/model_executor/layers/fused_moe/router/fused_moe_router.py: FusedMoERouter 继承 PluggableLayer,__init__ 补 super().__init__()
docs/design/plugin_system.md: router/routed-experts/runner 的注册与替换契约文档
tests/kernels/moe/test_routing.py: 4 个 pluggable 用例(OOT dispatch parity、DeepseekV2MoE bias 加载、simulator 继承)
```

## 性能声明

无性能数字;正文明确 "No accuracy or performance claim is made"。提供的是等价性证据:CPU 上 4 个 pluggable 用例通过,plain 与 grouped_sigmoid_bias 路由的 ids/weights 与 baseline 精确一致(`assert_close` rtol=0/atol=0);真实 DeepSeek-V4-Flash 全模型对比(43 层、256 experts、MXFP8/MXFP4、Ascend NPU、TP4/EP4、eager,3 个固定 prompt 各生成 16 tokens×2 轮)generated tokens 与 selected/cumulative logprobs 完全一致,43 个 router 全部执行。控制实验:用 PR base 的 router 源码跑同一测试 4 failed,证明 OOT dispatch 是必要的;去掉 `super().__init__()` 的对照 2 failed。

## 验证与回归风险

验证范围被作者明确限定:CPU 组件级 + 该兼容 runtime(vLLM `24420b7a…`、Ascend `7684befe…`)下的 router 继承等价,不做全 head NPU、CUDA 或 graph 模式验证;另有真实 `torch_npu` 组件 probe(16 tokens、256 experts、top-k 6,commit `c60a4409…`)通过。扩展的 CPU routing/simulator 选择 35 passed / 1 failed,失败项 `test_select_experts_rejects_noncontiguous_custom_routing` 为 pre-existing(PR base 同样失败,断言未改)。回归风险:注册按具体类名匹配,子类需逐个注册;本 PR 移除了重复构造/contiguity 检查相关的部分测试(新增测试代码 346→129 行),相关既有测试体未改。

## 关联

#51005(被更新的前序 PR);#49700(EPLB lifecycle,保持分离);机制同源:MoERunner / RoutedExperts 的 `PluggableLayer` 注册。本批同分类:#65177(INT8 MoE scale 索引)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
