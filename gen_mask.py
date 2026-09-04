"""
gen_mask.py
===========
DeepOHeat-V1 改造 · 掩码/界面编码生成函数库

2026-09-04 瘦身: 删除已被 make_icepak_dataset.py 取代的一次性构建脚本
(build_surface_3d5_data / build_volume_3d5_data / _write_3d5_chunked),
仅保留两个被引用的纯函数。

几何设定（AI 芯片 3.5D 布局，与 Ansys Icepak 模型 3d5_chip 一致）
--------------------------------------------------------------
横向：中列 2 Compute + 四角 4 HBM：

        y=1
      ┌───────┬───────┬───────┐
      │ HBM_3 │ Comp2 │ HBM_4 │
      ├───────┼───────┼───────┤  ← y=0.51（Compute 间间隙）
      │ HBM_1 │ Comp1 │ HBM_2 │
      └───────┴───────┴───────┘
   x=0    x=0.29  x=0.3   x=0.7 x=0.71  x=1

  - Compute（中列，2 块）：x∈[0.30,0.70]，y∈[0,0.49]∪[0.51,1.0] → mask 0
  - HBM（四角，4 块）：x∈[0,0.29]∪[0.71,1] 且 y∈[0,0.29]∪[0.71,1] → mask 1
  - 其余（间隙/暴露区）：Interposer → mask 2
  - interface：die 之间的边界线记 1
"""

import numpy as np

N_SURFACE = 21            # surface 功率图分辨率（21×21，遗留参数，保持兼容）
N_VOLUME = 101            # volume 功率图分辨率（101×101）


def make_horizontal_mask(n=N_VOLUME):
    """生成 AI 芯片横向 mask，返回 [n, n] float64 数组。

    mask 通道 = 材料/功能区域分类（0~2），与 Icepak 模型 3d5_chip 一致：
        0 = Compute Die（中列 2 块：x∈[0.30,0.70]，y∈[0,0.49]∪[0.51,1.0]）
        1 = HBM（四角 4 块：x∈[0,0.29]∪[0.71,1] 且 y∈[0,0.29]∪[0.71,1]）
        2 = Interposer（die 之间的间隙区域，材料为中介层）
    （Substrate=3, TIM=4 由纵向层定义，见 k_map.material_id）

    Icepak 实际尺寸（mm）：Compute x:3~7, y:0~4.9 & 5.1~10；
    HBM 四角 0~2.9 / 7.1~10（归一化后 0.29/0.71 边界，间隙 0.29~0.3、0.7~0.71、
    0.49~0.51 归 Interposer）。
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Compute Die：中列 x∈[0.30,0.70]，y 上下两块 [0,0.49] 和 [0.51,1.0]
    is_compute = (xx >= 0.30) & (xx <= 0.70) & ((yy <= 0.49) | (yy >= 0.51))
    # HBM：四角（x 与 y 都在 0.29 以内或 0.71 以外）
    is_hbm = (
        ((xx <= 0.29) & (yy <= 0.29)) |   # 左下 HBM_1
        ((xx >= 0.71) & (yy <= 0.29)) |   # 右下 HBM_2
        ((xx <= 0.29) & (yy >= 0.71)) |   # 左上 HBM_3
        ((xx >= 0.71) & (yy >= 0.71))     # 右上 HBM_4
    )
    # 非 Compute 非 HBM = Interposer（材料 2）
    mask = np.where(is_compute, 0, np.where(is_hbm, 1, 2)).astype(np.float64)
    return mask


def make_horizontal_interface(n=N_VOLUME):
    """生成界面编码：die 边界线记 1，其余记 0（与 Icepak 几何一致）。

    异质界面 = Compute/HBM die 之间的边界：
      x 方向：x≈0.29（HBM↔间隙）、0.30（间隙↔Compute）、0.70、0.71
      y 方向：y≈0.29、0.30、0.49（Compute 下块↔间隙）、0.51、0.70、0.71
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    interface = np.zeros((n, n), dtype=np.float64)
    # 竖线：x 方向 die 边界
    for xb in (0.29, 0.30, 0.70, 0.71):
        interface[np.isclose(xx, xb)] = 1.0
    # 横线：y 方向 die 边界
    for yb in (0.29, 0.30, 0.49, 0.51, 0.70, 0.71):
        interface[np.isclose(yy, yb)] = 1.0
    return interface
