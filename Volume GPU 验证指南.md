# Volume GPU 验证指南（阶段 3 · 任务 2）

> 目标：在 GPU 服务器上完成 volume 版的 baseline 与 3d5 训练验证。
> 前置：本机（沙箱）已改好所有代码 + 生成小的数据文件，只差 volume 训练集三通道数据（24GB）需在服务器上生成。

---

## 一、现状（代码已改好）

| 项 | 状态 |
|---|---|
| `heat_volumetric.py` 设备号 | ✅ 已改为 `'0'`（原 `'1'`） |
| `heat_volumetric.py` `--device_name` 默认值 | ✅ 已改为 `0` |
| `heat_volumetric.py` `--mode` 参数 | ✅ baseline / 3d5 |
| `heat_volumetric.py` interface_loss | ✅ 3d5 模式启用 |
| `heat_surface.py` interface_loss | ✅ 3d5 模式启用 |
| `k_map.py` | ✅ 新增，分区 k 映射 |
| `data/mask_volume.npy` `interface_volume.npy` `fs_test_3d5_volume.npy` | ✅ 已生成 |
| `data/fs_train_3d5_volume.npy` | ❌ 24GB，需服务器生成 |

---

## 二、服务器操作步骤

### 1. 同步代码到服务器

把 `2_deepoheatv1/DeepOHeat-v1/` 整个目录传到服务器（或用 git clone 官方仓库 + 应用修复）。

### 2. 安装依赖（服务器）

```bash
# 若之前装过旧版 jax 导致段错误，先卸载
pip uninstall -y jax jaxlib

# 装 CUDA12 版 JAX + equinox + optax
pip install --upgrade "jax[cuda12]" equinox optax gputil numpy
```

> 若遇段错误，参见 `DeepOHeat-v1_朋友运行指南.md`。

### 3. 生成 volume 三通道训练集

在 `DeepOHeat-v1/` 目录下：

```bash
python gen_mask.py --volume
```

这会生成：
- `data/mask_volume.npy`（10201，区域掩码）
- `data/interface_volume.npy`（10201，界面编码）
- `data/fs_train_3d5_volume.npy`（100000×30603，约 24GB）
- `data/fs_test_3d5_volume.npy`（100×30603）

> 已存在的同名文件会被覆盖，不影响原始数据。

### 4. 跑 baseline（验证原版没坏）

```bash
python heat_volumetric.py --mode baseline
```

- 参数量应显示 805,120（原版）
- 结果目录：`results/results_volume/DeepOHeat_v1/..._mode_baseline/`
- 记录：loss 曲线、MAPE、运行时间、显存占用

### 5. 跑 3d5（分区 k + interface_loss）

```bash
python heat_volumetric.py --mode 3d5
```

- 参数量应显示 1,030,912（三通道）
- 应打印 `[3d5] 分区 k 场已构造, 形状 [1, 101, 101, 56, 1]`
- 结果目录：`..._mode_3d5/`

---

## 三、需要记录并提交的指标

每次训练完成后，`results/...` 目录下有这些文件，直接上交：

| 文件 | 内容 |
|---|---|
| `log (loss).csv` | 训练损失曲线 |
| `log (eval metrics).csv` | MAPE / rel_l2 / rmse / max_l1 / pape |
| `total runtime (sec).csv` | 运行时间 |
| `memory usage (mb).csv` | 显存占用 |
| `u_pred_heat3d.npy` | 预测温度场（可画云图） |

建议训练时截图终端输出（参数量、每 epoch loss、评估指标）。

---

## 四、预期结果

| 模式 | 参数量 | 期望 |
|---|---|---|
| baseline | 805,120 | 与原版复现结果一致（volume MAPE ~0.125%） |
| 3d5 | 1,030,912 | 分区 k + 界面约束生效，loss 正常下降 |

> 注意：3d5 的 k 值（130/150/1.5 缩放到 1.3/1.5/0.015）与原版 volume 的 k 分布（底层 20/内部 0.1）差异很大，首次运行可能 loss 数值偏高，属正常现象——3d5 是**不同的物理设定**，不能直接和 baseline 的 loss 数值对比，重点是看它能否正常收敛。

---

## 五、常见问题

| 问题 | 解决 |
|---|---|
| 段错误（Segmentation fault） | 卸载 jax/jaxlib 重装 `jax[cuda12]` |
| 找不到 `fs_train_3d5_volume.npy` | 先跑 `python gen_mask.py --volume` |
| OOM 显存不足 | 减小 `--batch`（默认 50）或 `--nc` |
| `jax.tree_map` 报错 | 确认用的是修复版代码 |
