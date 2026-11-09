# PR #42481: [Bugfix] Fix layerwise reload alias-buffer corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42481](https://github.com/vllm-project/vllm/pull/42481) |
| 作者 | rasdani |
| 合并日期 | 2026-05-15 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
layerwise reload 把「参数别名」的非持久 buffer 当独立 buffer 拷回,覆写已加载权重(如 MambaMixer2.conv_weights);捕获时按递归参数存储指针排除别名 buffer。

## 问题现象
生产环境对 `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` 反复做 checkpoint 格式权重 reload 后出现 NaN 生成。作者在生产式 5 节点 PrimeRL run 的 out-of-tree 插桩确认了别名形态(layer_class=MambaMixer2、buffer_name=conv_weights、non_persistent=true、alias_matches=[conv1d.weight]、direct_param_names=["A","D","dt_bias"]),该 run 的 8 个 TP worker 中递归守卫共检测并跳过 320 个 `MambaMixer2.conv_weights` buffer——旧的仅检查直接参数的实现发现不了这种父 buffer/子参数别名。

## 根因
正文明确披露:layerwise reload 先记录模块的参数与 buffer,再把模块还原到 meta tensor,之后 materialize 并把处理后的 tensor 拷回原 kernel tensor 存储。这对独立 buffer 正确,但对只是模型参数 view/alias 的非持久 buffer 不安全:`MambaMixer2` 把 `conv_weights` 注册为 non-persistent buffer,而它别名子参数 `conv1d.weight`;reload 期间父模块会因直接参数(A、D、dt_bias)被 materialize/finalize,`conv_weights` 被当作独立 buffer 捕获,拷回时写入的 buffer 存储别名 `conv1d.weight`,从而覆写已加载的子卷积权重。第一道边界是元数据捕获:别名检查必须遍历 `layer.parameters(recurse=True)` 而非只看 `layer._parameters`;第二道边界在拷回:`initialize_layerwise_reload` 保存的原 kernel tensor 仍含别名 buffer 供重新注册,但捕获时被刻意排除后,临时 materialized 层中已无该 buffer,拷回路径必须跳过这种缺失情况而非盲目 `getattr(layer, name)`。

## 修复方案
- `vllm/model_executor/model_loader/reload/meta.py`:`capture_layer_to_meta` 新增 `_is_non_persistent_parameter_alias_buffer` / `_parameter_storage_ptrs`(每层预计算一次参数存储指针,跳过 `UninitializedParameter`),捕获时排除与子树任一参数共享存储的非持久 buffer;
- `vllm/model_executor/model_loader/reload/layerwise.py`:`_copy_and_restore_kernel_tensors` 拷回 buffer 前检查 `if name not in layer._buffers: continue`;
- 新增两个回归测试:同模块别名与父 buffer/子参数别名(匹配 `MambaMixer2.conv_weights -> conv1d.weight` 模式),测试结果 2 passed。

## 影响与回归风险
- 回归面:layerwise reload(`model_loader/reload`)的元数据捕获与拷回路径;行为变化为:与参数共享存储的非持久 buffer 不再被当作独立 buffer 参与恢复,保留原别名关系;正常(非别名)buffer 的 reload 不受影响;非 reload 路径无改动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
