"""
gen_truth_3d5.py
================
DeepOHeat-V1 改造 · 生成 3.5D 自洽真值数据

用途
----
用有限差分（GMRES）求解 3d5 物理的热方程，为现有功率布局生成温度真值。

物理设定（与 heat_volumetric.py 的 3d5 损失一致）
------------------------------------------------
- 网格：101×101×56，z ∈ [0, 0.55]，dz = 0.01
- 内部：∇·(k∇T) = 0，k 是分区场（2×2 区域 × z 分层系数）
- 功率注入（体积热源，对应 loss 里的分层）：
    z 索引 10 (z=0.10)：+f
    z 索引 11-14 (z=0.11~0.14)：+2f
    z 索引 15 (z=0.15)：+f
- 边界条件（与 loss 一致）：
    顶面：u = 0.2 - 2·du/dz   =>  u + 2·du/dz = 0.2
    底面：u = 0.2 + 40·du/dz  =>  u - 40·du/dz = 0.2
    侧面：绝热 du/dn = 0
- 界面热流连续：界面两侧 k·du/dn 相等（离散后自然满足）

k 场来源
--------
k_map.build_k_field（含 2×2 分区 + z 分层 + /100 缩放），
与训练时模型假设的物理完全一致，保证自洽。

用法
----
python gen_truth_3d5.py --num 20
  --num 生成前 N 个测试样本的真值
输出：data/u_test_3d5.npy  [N, 101, 101, 56]
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp
from k_map import build_k_field

# ---------------- 网格参数 ----------------
# 默认低分辨率（51×51×28）验证物理；可 --res 101 用全分辨率（慢）
NX = NY = int(os.environ.get('GEN_RES', '51'))
NZ = int(0.55 * NX + 0.45)
DZ = 0.55 / (NZ - 1)          # 厚度 0.55
DX = DY = 1.0 / (NX - 1)


# ---------------- 构造 k 场（与训练一致的缩放 k） ----------------
def get_k_field_raw():
    """返回 [NX, NY, NZ] 的 k 场（与训练模型假设完全一致，含 /100 缩放）。

    训练用 k_map.build_k_field（区域3=0.015），真值生成必须用同一套 k，
    否则模型无法学会（物理不一致）。直接复用 build_k_field。
    """
    xc = jnp.linspace(0, 1, NX).reshape(-1, 1)
    yc = jnp.linspace(0, 1, NY).reshape(-1, 1)
    zc = jnp.linspace(0, 0.55, NZ).reshape(-1, 1)
    k = build_k_field(xc, yc, zc)          # [1, NX, NY, NZ, 1] 已 /100
    return np.asarray(k[0, ..., 0])        # [NX, NY, NZ]


# ---------------- 构建稀疏矩阵（变系数 k） ----------------
def build_matrix(k_field):
    """k_field: [NX, NY, NZ] 物理 k 场。返回 A 矩阵（57 万维稀疏）。"""
    n = NX * NY * NZ
    rows, cols, vals = [], [], []

    def k_at(i, j, kk):
        return k_field[i, j, kk]

    for kk in range(NZ):
        for j in range(NY):
            for i in range(NX):
                row = kk * NX * NY + j * NX + i

                if kk == 0:  # 底面：u - 40*du/dz = 0.2 => (1 + 40/dz)*u0 - (40/dz)*u1 = 0.2
                    rows.append(row); cols.append(row); vals.append(1.0 + 40.0 / DZ)
                    if kk < NZ - 1:
                        rows.append(row); cols.append(row + NX * NY); vals.append(-40.0 / DZ)
                elif kk == NZ - 1:  # 顶面：u + 2*du/dz = 0.2 => (1 + 2/dz)*u_top - (2/dz)*u_topm1 = 0.2
                    rows.append(row); cols.append(row); vals.append(1.0 + 2.0 / DZ)
                    if kk > 0:
                        rows.append(row); cols.append(row - NX * NY); vals.append(-2.0 / DZ)
                else:  # 内部：∇·(k∇T) = 0，变系数 k
                    # 用界面调和平均（harmonic average）处理变 k 的热流连续
                    diag = 0.0

                    # x 方向：k_{i+1/2}*(T_{i+1}-T_i)/dx - k_{i-1/2}*(T_i-T_{i-1})/dx
                    # 侧面绝热（Neumann）
                    if i == 0:   # 左面
                        k_face = k_at(i, j, kk)
                        diag += -k_face / DX**2
                        rows.append(row); cols.append(row + 1); vals.append(k_face / DX**2)
                    elif i == NX - 1:  # 右面
                        k_face = k_at(i, j, kk)
                        diag += -k_face / DX**2
                        rows.append(row); cols.append(row - 1); vals.append(k_face / DX**2)
                    else:
                        k_left  = 2 * k_at(i-1, j, kk) * k_at(i, j, kk) / (k_at(i-1, j, kk) + k_at(i, j, kk))
                        k_right = 2 * k_at(i, j, kk) * k_at(i+1, j, kk) / (k_at(i, j, kk) + k_at(i+1, j, kk))
                        diag += -(k_left + k_right) / DX**2
                        rows.append(row); cols.append(row - 1); vals.append(k_left / DX**2)
                        rows.append(row); cols.append(row + 1); vals.append(k_right / DX**2)

                    # y 方向
                    if j == 0:
                        k_face = k_at(i, j, kk)
                        diag += -k_face / DY**2
                        rows.append(row); cols.append(row + NX); vals.append(k_face / DY**2)
                    elif j == NY - 1:
                        k_face = k_at(i, j, kk)
                        diag += -k_face / DY**2
                        rows.append(row); cols.append(row - NX); vals.append(k_face / DY**2)
                    else:
                        k_down = 2 * k_at(i, j-1, kk) * k_at(i, j, kk) / (k_at(i, j-1, kk) + k_at(i, j, kk))
                        k_up   = 2 * k_at(i, j, kk) * k_at(i, j+1, kk) / (k_at(i, j, kk) + k_at(i, j+1, kk))
                        diag += -(k_down + k_up) / DY**2
                        rows.append(row); cols.append(row - NX); vals.append(k_down / DY**2)
                        rows.append(row); cols.append(row + NX); vals.append(k_up / DY**2)

                    # z 方向
                    if kk > 0 and kk < NZ - 1:
                        k_down = 2 * k_at(i, j, kk-1) * k_at(i, j, kk) / (k_at(i, j, kk-1) + k_at(i, j, kk))
                        k_up   = 2 * k_at(i, j, kk) * k_at(i, j, kk+1) / (k_at(i, j, kk) + k_at(i, j, kk+1))
                        diag += -(k_down + k_up) / DZ**2
                        rows.append(row); cols.append(row - NX * NY); vals.append(k_down / DZ**2)
                        rows.append(row); cols.append(row + NX * NY); vals.append(k_up / DZ**2)

                    rows.append(row); cols.append(row); vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A


# ---------------- 构建 RHS（体积功率注入） ----------------
# 功率注入的 z 位置（对应物理坐标）：
#   z = 0.10 加 f, z ∈ (0.10, 0.15) 加 2f, z = 0.15 加 f
# 不同分辨率下换算成 z 索引
Z_INJ_BOT = 0.10   # z=0.10
Z_INJ_TOP = 0.15   # z=0.15


def _z_index(z_phys):
    """物理 z 坐标 → 网格索引（最近）。"""
    return int(round(z_phys / DZ))


def build_rhs(f_map):
    """f_map: [NX, NY] 平面功率映射。返回 RHS 向量。"""
    b = np.zeros(NX * NY * NZ)
    f = np.asarray(f_map).reshape(NX, NY)

    # 体积功率注入：k·∇²T + 功率 = 0 => RHS = -功率
    k_bot = _z_index(Z_INJ_BOT)          # z=0.10
    k_top = _z_index(Z_INJ_TOP)          # z=0.15
    for kk in [k_bot, k_top]:
        sl = kk * NX * NY
        b[sl:sl + NX * NY] -= f.ravel()          # 单倍功率
    for kk in range(k_bot + 1, k_top):
        sl = kk * NX * NY
        b[sl:sl + NX * NY] -= 2 * f.ravel()      # 双倍功率（中间层）

    return b


# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=20, help='生成前 N 个测试样本真值')
    parser.add_argument('--res', type=int, default=51, help='网格分辨率 (默认 51, 可选 101)')
    args = parser.parse_args()

    global NX, NY, NZ, DZ, DX, DY
    NX = NY = args.res
    NZ = int(0.55 * NX + 0.45)
    DZ = 0.55 / (NZ - 1)
    DX = DY = 1.0 / (NX - 1)

    print(f"分辨率 {NX}x{NX}x{NZ}")
    print("构建 k 场...")
    k_field = get_k_field_raw()
    print(f"k 场形状: {k_field.shape}, 范围 [{k_field.min():.5f}, {k_field.max():.4f}]")

    print(f"构建系统矩阵 ({NX}x{NX}x{NZ})...")
    A = build_matrix(k_field).tocsc()
    print(f"矩阵规模: {A.shape}, 非零元: {A.nnz}")

    # 加载测试功率布局（101 分辨率，下采样到 NX）
    fs_test_orig = np.load('data/fs_test_volume.npy').reshape(-1, 101, 101)
    num = min(args.num, fs_test_orig.shape[0])
    print(f"生成前 {num} 个样本的真值（分辨率 {NX}x{NX}x{NZ}）...")

    # 下采样功率映射到 NX（最近邻）
    def downsample_f(f_orig):
        idx = (np.arange(NX) * 100.0 / (NX - 1)).astype(int)
        return f_orig[np.ix_(idx, idx)]

    # 关键优化：矩阵 A 对所有样本相同（只 RHS 不同）
    # 一次性 LU 分解，后续每次只做前代/回代（快很多）
    print("LU 分解（一次性，可能较慢）...")
    import time
    t0 = time.time()
    lu = spla.splu(A)
    print(f"LU 分解完成, 耗时 {time.time()-t0:.1f}s")

    u_all = np.zeros((num, NX, NY, NZ))
    for s in range(num):
        f_map = downsample_f(fs_test_orig[s])  # [NX, NY]
        b = build_rhs(f_map)

        # 用 LU 前代/回代求解（复用分解，远快于每次 spsolve）
        sol = lu.solve(b)
        # 重排为 [NX, NY, NZ]
        u = sol.reshape(NZ, NX, NY).transpose(1, 2, 0)  # [NX, NY, NZ]
        u_all[s] = u
        if (s + 1) % 5 == 0 or s == num - 1:
            print(f"  已完成 {s+1}/{num}, 耗时 {time.time()-t0:.1f}s, 温度范围 [{u.min():.3f}, {u.max():.3f}]")

    # 保存（带分辨率标识）
    # 101 分辨率用标准名 u_test_3d5.npy（heat_volumetric 3d5 模式自动读取）
    # 其他分辨率带后缀，避免覆盖
    if NX == 101:
        out = 'data/u_test_3d5.npy'
    else:
        out = f'data/u_test_3d5_{NX}x{NX}x{NZ}.npy'
    np.save(out, u_all)
    print(f"已保存 {out}, 形状 {u_all.shape}")


if __name__ == '__main__':
    main()
