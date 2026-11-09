# PR #23672: [Bugfix] Fix incorrect original shape in hashing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23672](https://github.com/vllm-project/vllm/pull/23672) |
| 作者 | DarkLight1337 |
| 合并日期 | 2025-08-26 |
| labels | ready, multi-modality |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
bfloat16 tensor 序列化时在 reshape 成字节流之后才取 shape,哈希丢失原始形状,形状不同、数据相同的输入哈希碰撞;改为先记录原始形状再修复。

## 问题现象
正文未描述具体现象,PR Purpose 只链接了一个代码 review 评论(vllm-project/vllm/pull/23623/files#r2301482573)。从测试用例可看出问题面:`test_hash_collision_tensor_shape` 断言形状为 (5,10,20,3) 与 (10,20,5,3) 的全零 tensor(展平后数据完全相同)必须得到不同的哈希,即"数据相同但形状不同时哈希必须不同";该测试被参数化扩到 bfloat16,说明此前 bfloat16 路径未覆盖、存在哈希碰撞问题。

## 根因
正文未披露根因(只链接 review 评论)。从 diff 描述机制(由 diff 推断):`vllm/multimodal/hasher.py` 的 `serialize_item` 处理 bfloat16 tensor 时,因 NumPy 不支持 bfloat16,先把 tensor `.contiguous()` 后 `.view((numel,)).view(torch.uint8)` 变成一维 uint8 字节数组再转 numpy;但序列化字典里的 `original_shape` 用的是 `tuple(tensor_obj.shape)`——此时 shape 已被改写成 `(numel,)`,原始多维形状(如 (5,10,20,3))没有进入哈希。于是两个形状不同、元素序列相同的 bfloat16 tensor 序列化结果完全一致,哈希发生碰撞,本应区分的缓存键被当成同一项。

## 修复方案
改 `vllm/multimodal/hasher.py` `serialize_item` 的 bfloat16 分支:在 reshape 之前先保存原始形状,序列化时使用保存值:
```python
tensor_shape = tensor_obj.shape
...
"original_shape": tuple(tensor_shape),
"data": tensor_obj.numpy(),
```
即 `original_shape` 从 reshape 后的 `(numel,)` 恢复为真正的原始多维形状。非 bfloat16 tensor 仍走 `tensor_obj.numpy()` 原路径不变。同时更新 `tests/multimodal/test_hasher.py`:将 `test_hash_collision_tensor_shape` 用 `@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])` 参数化,使形状碰撞测试覆盖 bfloat16 序列化路径。

## 影响与回归风险
- 影响窗口:正文未说明;受影响的是引入 bfloat16 字节视图序列化之后的版本(引入提交正文未给出)。
- 回归面:仅 `vllm/multimodal/hasher.py` bfloat16 tensor 的序列化/哈希路径。修复后 bfloat16 输入的哈希值会与修复前不同(形状维度重新参与哈希),依赖持久化旧哈希值的缓存需要失效重建;非 bfloat16 路径与 np.ndarray 路径不受影响。新增参数化测试可防止该路径再次回归。

## 关联
- [PR #23623 review 评论](https://github.com/vllm-project/vllm/pull/23623/files#r2301482573)(FIX 目标)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
