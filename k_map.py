"""
k_map.py
========
DeepOHeat-V1 改造 · 阶段 2：mask → k 映射模块

作用
----
把「区域 ID mask」映射成「空间热导率场 k(x, y, z)」。

k 场由两部分决定：
  1. 水平方向（x, y）：区域 ID 查表（mask=0/1/2/3 → 对应 k 值）
  2. 垂直方向（z）  ：分层系数（不同 z 层乘不同系数，模拟垂直堆叠的材料差异）

数值稳定处理
------------
用户给的原始 k 值（130 / 150 / 1.5）与原版 surface 的 k=0.1 差 1000 多倍，
直接用于 PDE 残差会让数值爆炸、训练不稳。因此做统一缩放：

    实际参与训练的 k = 原始 k / K_SCALE

K_SCALE 默认 100。缩放只改变数值大小，不改变各区域 k 的相对比例，
物理意义（谁导热好、谁差）完全保留。
"""

import jax
import jax.numpy as jnp


#########################################################################
# 1. 材料 → k 映射表（AI 芯片 3.5D，行业标准值）
#########################################################################
# 材料类型 ID → 热导率 k [W/(m·K)]
#   0 = GPU Chiplet（硅）    k = 130
#   1 = HBM（硅）            k = 130
#   2 = Interposer（中介层）  k = 130
#   3 = Substrate（基板）     k = 0.5
#   4 = TIM（热界面材料）     k = 2.0
K_MATERIAL = {0: 130.0, 1: 130.0, 2: 130.0, 3: 0.5, 4: 2.0}

# 参与训练时的缩放系数（见模块顶部说明）
K_SCALE = 100.0

# 查表用数组：索引 = 材料类型 ID
K_MATERIAL_ARRAY = jnp.array([K_MATERIAL[i] for i in sorted(K_MATERIAL.keys())], dtype=jnp.float32)


#########################################################################
# 2. z 方向分层（纵向层边界，物理坐标 z）
#########################################################################
# AI 芯片 3.5D 纵向结构（自下而上 z 增）：
#   底层 z < 0.1     : Substrate（基板）
#   中层 0.1~0.35    : Interposer（中介层）
#   上层 0.35~0.4    : TIM（热界面材料）
#   顶层 z > 0.4     : die 层（GPU/HBM，横向分区）
# 说明：这些是占位边界，等真实层结构替换。
Z_SUB_TOP   = 0.10   # Substrate 顶
Z_INT_TOP   = 0.35   # Interposer 顶
Z_TIM_TOP   = 0.40   # TIM 顶


def z_layer(zz):
    """按 z 坐标返回层类型（Substrate=3, Interposer=2, TIM=4, die层=特殊）。"""
    return jnp.where(zz < Z_SUB_TOP, 3,
           jnp.where(zz < Z_INT_TOP, 2,
           jnp.where(zz < Z_TIM_TOP, 4, -1)))  # -1 = die 层，由横向决定


#########################################################################
# 3. 横向 die 分区（die 层内 GPU / HBM 布局）
#########################################################################
# AI 芯片横向布局（MI300X-inspired benchmark，等 B 组真实布局替换）：
#   4 个 Compute Die（中心 2×2）+ 4 个 HBM（四角）+ Interposer 上方边缘
#   x ∈ [0,1], y ∈ [0,1]
#   Compute 区域：中心 x∈[0.3,0.7], y∈[0.3,0.7]（2×2 四个 die）
#   HBM 区域：四角矩形
#   die 间隙（十字线区域）= Interposer 上方，横向记 2
def horizontal_region(xx, yy):
    """返回横向区域类型：0=ComputeDie, 1=HBM, 2=Interposer。

    这是 die 层的横向材料分区（0~2）；
    Substrate=3、TIM=4 由纵向层定义（见 material_id）。
    非 die 区（间隙）归为 Interposer 材料；interface 通道独立标记界面。
    xx, yy : 同形状坐标。
    """
    # Compute Die：中心区
    is_compute = (xx >= 0.3) & (xx <= 0.7) & (yy >= 0.3) & (yy <= 0.7)
    # HBM 四角
    is_hbm = (
        ((xx < 0.3) & (yy < 0.3)) |   # 左下
        ((xx > 0.7) & (yy < 0.3)) |   # 右下
        ((xx < 0.3) & (yy > 0.7)) |   # 左上
        ((xx > 0.7) & (yy > 0.7))     # 右上
    )
    return jnp.where(is_compute, 0, jnp.where(is_hbm, 1, 2)).astype(jnp.int32)


#########################################################################
# 4. 材料 ID 计算（三维：横向 die + 纵向层）
#########################################################################
def material_id(xx, yy, zz):
    """返回每个 (x,y,z) 点的材料类型 ID（0-4）。

    规则：
      - die 层（zz >= Z_TIM_TOP）：横向 GPU=0 / HBM=1
      - Interposer 层：2
      - Substrate 层：3
      - TIM 层：4
    """
    layer = z_layer(zz)                        # 3/2/4/-1
    h_region = horizontal_region(xx, yy)       # 0/1/2

    # die 层用横向分区，其他层用固定材料
    return jnp.where(layer == -1, h_region, layer)


#########################################################################
# 5. interface 类型（异质界面分类，预留多类别）
#########################################################################
# interface 通道语义：
#   0 = 非界面
#   1 = Die-TIM 界面（纵向）
#   2 = TIM-Interposer 界面（纵向）
#   3 = Interposer-Substrate 界面（纵向）
# 便于论文按界面类型分析误差。
# surface 2D 阶段：只用 0/1（1 = die 边界，横向）
def interface_type(zz):
    """按 z 坐标返回界面类型。

    zz : 任意形状数组（物理坐标 z）。
    返回与 zz 同形状的 int32 数组。
    """
    # 界面在层边界附近：取层边界两侧的薄层
    # die层底部 ≈ Z_TIM_TOP（Die-TIM 界面）
    # TIM 底部 ≈ Z_INT_TOP（TIM-Interposer 界面）
    # Interposer 底部 ≈ Z_SUB_TOP（Interposer-Substrate 界面）
    # 简化：用 z 落在边界上的薄带（±2% 厚度）标记
    eps = 0.01  # 界面薄层厚度

    is_die_tim     = (zz >= Z_TIM_TOP - eps) & (zz <= Z_TIM_TOP + eps)
    is_tim_int     = (zz >= Z_INT_TOP - eps) & (zz <= Z_INT_TOP + eps)
    is_int_sub     = (zz >= Z_SUB_TOP - eps) & (zz <= Z_SUB_TOP + eps)

    return jnp.where(is_die_tim, 1,
           jnp.where(is_tim_int, 2,
           jnp.where(is_int_sub, 3, 0))).astype(jnp.int32)


#########################################################################
# 6. 构造 k 场 k(x, y, z)
#########################################################################
def build_k_field(xc, yc, zc):
    """从坐标网格构造 k 场（三维材料查表）。

    参数
    ----
    xc : [nx, 1]   x 坐标列向量
    yc : [ny, 1]   y 坐标列向量
    zc : [nz, 1]   z 坐标列向量

    返回
    ----
    [1, nx, ny, nz, 1] 的 float32 数组，已除以 K_SCALE。
    第 0 维和最后 1 维为广播维，可与模型输出 [batch, nx, ny, nz, 1] 直接相乘。
    """
    xx, yy, zz = jnp.meshgrid(xc.ravel(), yc.ravel(), zc.ravel(), indexing='ij')

    # 三维材料查表
    mid = material_id(xx, yy, zz)              # [nx, ny, nz]
    k = K_MATERIAL_ARRAY[mid]                  # [nx, ny, nz]

    # 缩放（数值稳定）
    k = k / K_SCALE

    return k[None, ..., None]                  # [1, nx, ny, nz, 1]
