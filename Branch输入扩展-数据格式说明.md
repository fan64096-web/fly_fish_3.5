# DeepOHeat-V1 Branch 输入扩展 · 数据格式说明

> 阶段 1（第 2 周）交付物之一 · 2026-08-08
> 目标：让 Branch 网络接收 `功率 + 掩码 + 界面` 三通道输入，用于 3.5D 异构芯片热仿真。

---

## 一、改动总览

| 文件 | 改动 |
|---|---|
| `models.py` | `DeepOHeat_v1` 新增 `channels` 参数，Branch 第一层输入维度 = `branch_dim × channels` |
| `heat_surface.py` | 新增 `--mode` 参数（`baseline` / `3d5`），数据加载与损失按模式分支 |
| `gen_mask.py` | 新增，生成区域掩码、界面编码，并拼接出三通道训练数据 |
| `train.py` / `eval.py` / `hvp.py` / `kan.py` | **未改动** |

---

## 二、运行方式

### 1. 生成数据（只需跑一次）

```bash
python gen_mask.py
```

会在 `data/` 下生成：

| 文件 | 形状 | 说明 |
|---|---|---|
| `mask_surface.npy` | [441] | 区域掩码（21×21 摊平），取值 0/1/2/3 |
| `interface_surface.npy` | [441] | 界面编码（21×21 摊平），取值 0/1 |
| `fs_train_3d5_surface.npy` | [10000, 1323] | 训练三通道输入 |
| `fs_test_3d5_surface.npy` | [10, 1323] | 测试三通道输入 |

### 2. 训练

```bash
# 原版基线（只有功率，行为与原版完全一致）
python heat_surface.py --model_name DeepOHeat_v1 --mode baseline

# 3.5D 模式（功率+掩码+界面 三通道）
python heat_surface.py --model_name DeepOHeat_v1 --mode 3d5
```

结果目录自动带 `_mode_baseline` / `_mode_3d5` 后缀，互不覆盖。

---

## 三、三通道拼接规则（核心）

每个样本的 Branch 输入是一个长向量，由三段拼接而成：

```
f_combined = [ 功率段 | 掩码段 | 界面段 ]
  ↑             ↑        ↑
  f[0:441]     f[441:882]  f[882:1323]
```

| 段 | 长度 | 内容 |
|---|---|---|
| 功率段 | 441（=21²） | 原始功率分布，逐 (x,y) 点 |
| 掩码段 | 441 | 每个 (x,y) 点的区域 ID（0/1/2/3） |
| 界面段 | 441 | 每个 (x,y) 点是否在交界线（1/0） |

### 重要性质

- **掩码、界面对所有样本相同**（同一个芯片几何），由 `gen_mask.py` 生成后广播到每一行。
- **模型看到的输入维度**：baseline = 441，3d5 = 1323。
- **损失函数只使用功率段**：顶面功率边界条件 `bc_top` 在 3d5 模式自动只取前 441 维（通过 `power_dim` 参数控制），baseline 模式取全部输入，行为与原版逐位一致。

---

## 四、几何定义（第一阶段，简单 2×2）

芯片平面归一化坐标 [0,1]×[0,1]，切成 2×2 四块矩形：

```
        y=1
      ┌──────┬──────┐
      │ 区域2 │ 区域3 │
      ├──────┼──────┤  ← y=0.5（界面线）
      │ 区域0 │ 区域1 │
      └──────┴──────┘
   x=0      x=0.5      x=1
```

- **mask**：左下=0，右下=1，左上=2，右上=3。
- **interface**：落在 `x=0.5` 或 `y=0.5` 上的点记 1，其余记 0。

> 这是"验证流程"用的简化几何。真实 3.5D 布局（不同 chiplet 尺寸/位置）后续可在 `gen_mask.py` 里按实际坐标替换生成函数，不影响拼接与训练代码。

---

## 五、代码里做了什么（对照）

### 1. `models.py`：`DeepOHeat_v1`

```python
def __init__(..., channels=1, ...):
    self.branch = nn.MLP(branch_dim*channels, rank*field_dim, ...)
    #              ↑ 输入维度自动适配拼接后的向量长度
```

- `channels=1`：baseline，等价于原版。
- `channels=3`：3d5，Branch 第一层多出 `(1323-441)×256 = 225792` 个参数。

### 2. `heat_surface.py`：数据加载

```python
if args.mode == '3d5':
    fs_train = jnp.load('data/fs_train_3d5_surface.npy').reshape(-1, 3*21**2)
    ...
    args.channels = 3
else:  # baseline
    fs_train = jnp.load('data/fs_train_surface.npy').reshape(-1, 21**2)
    args.channels = 1
```

### 3. `heat_surface.py`：损失函数

```python
def apply_model_deepoheat_st(model, xc, yc, zc, fc, lam_b=1., power_dim=None):
    def PDE_loss(...):
        ...
        f_power = f[..., :power_dim] if power_dim is not None else f   # 只取功率段
        bc_top = jnp.mean((uz[...,-1,:] - f_power.reshape(-1,21,21,1))**2)
```

- baseline：`power_dim=None`，`f_power = f`，行为不变。
- 3d5：`power_dim=441`，只切功率段，掩码/界面不参与边界条件。

---

## 六、验证结果（本机 CPU 冒烟测试，3 epochs）

| 模式 | 参数量 | Epoch 1 loss | Epoch 3 loss | 是否正常 |
|---|---|---|---|---|
| baseline | 805,120 | 807.64 | 356.17 | ✅ |
| 3d5 | 1,030,912 | 808.74 | 367.62 | ✅ |

- 两种模式 loss 均正常下降，训练 + 评估链路完整跑通。
- baseline 参数量 = 原版 805,120，证明网络主体结构未变。

---

## 七、验收标准对照

| 验收项 | 状态 |
|---|---|
| 原版 DeepOHeat-V1 可以正常训练 | ✅ `--mode baseline` 跑通 |
| 新输入模式可以正常训练 | ✅ `--mode 3d5` 跑通 |
| 不修改 PDE loss 和网络主体结构 | ✅ 只加了 `power_dim` 切功率段的兼容逻辑，物理项未改；Branch 只是输入维度变化 |

---

## 八、下一步（阶段 2：分区子域 PDE 残差）

当前 mask / interface 已作为输入通道喂给模型，但**还没有被物理上使用**（损失仍用均匀 k）。下一阶段将在 `PDE_loss` 里：
1. 按 mask 把内部点分成不同区域。
2. 每个区域用对应热导率 `k` 计算残差。
3. （再下一阶段）在 interface 处加界面热流连续项。

届时 mask / interface 输入通道会与损失函数联动，形成完整的三步法改造。
