"""
parse_icepak_report.py
======================
DeepOHeat-V1 3.5D 改造 · 将 Icepak Full report 导出的温度场转为 3d5 训练/验证数据

输入
----
Icepak 操作:Report → Full report → 取消勾选 "Only summary information"
          → 变量 Temperature → Write to file(如 report.02)

文件格式(每节点一行):
  Node    X              Y              Z              Value
     1  5.0000000e-03  4.9000000e-03  1.0000000e-03  4.9185968e+01
  (坐标单位 m, 温度单位 °C)

输出（data/ 目录）
-----------------
  fs_icepak_volume.npy   [N, 3*NX*NY]     三通道输入（功率|掩码|界面）
  u_icepak_volume.npy    [N, NX, NY, NZ]  归一化温度真值 u
  T_icepak_raw.npy       [N, NX, NY, NZ]  原始温度（°C）

用法
----
python parse_icepak_report.py --report C:/Users/25164/Desktop/3d5_chip/report.02 \
       --power "0.28,0.28,0.12,0.12,0.12,0.12" --nx 101 --ny 101 --nz 56
"""

import os
import re
import json
import argparse
import numpy as np

try:
    import scipy.interpolate as si
except ImportError:
    si = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ---------------------------------------------------------------------
# 几何映射（Icepak 真实几何 ↔ 训练归一化几何）
# ---------------------------------------------------------------------
X_ICEPAK_MM = 10.0
Y_ICEPAK_MM = 10.0
# z 分段线性映射：Icepak 层边界(mm) → 训练层边界(归一化)，与 k_map.py 一致
Z_ICEPAK_BOUNDS = [0.0, 0.5, 1.0, 1.2, 1.8]
Z_TRAIN_BOUNDS  = [0.0, 0.1, 0.35, 0.4, 0.55]

# die 区域（归一化 [x0,x1,y0,y1]），与 gen_mask/k_map/parse_icepak_data 一致
DIE_REGIONS = [
    (0.30, 0.70, 0.00, 0.49),   # Compute_1
    (0.30, 0.70, 0.51, 1.00),   # Compute_2
    (0.00, 0.29, 0.00, 0.29),   # HBM_1
    (0.71, 1.00, 0.00, 0.29),   # HBM_2
    (0.00, 0.29, 0.71, 1.00),   # HBM_3
    (0.71, 1.00, 0.71, 1.00),   # HBM_4
]


def z_icepak_to_train(z_mm):
    """Icepak z(mm) → 训练归一化 z 的分段线性映射。"""
    zi = np.asarray(z_mm, dtype=np.float64)
    zt = np.empty_like(zi)
    for (a0, a1, b0, b1) in zip(Z_ICEPAK_BOUNDS[:-1], Z_ICEPAK_BOUNDS[1:],
                                Z_TRAIN_BOUNDS[:-1], Z_TRAIN_BOUNDS[1:]):
        mask = (zi >= a0) & (zi <= a1)
        zt[mask] = b0 + (b1 - b0) * (zi[mask] - a0) / (a1 - a0)
    return zt


# ---------------------------------------------------------------------
# 读 Icepak report 文件（Node X Y Z Value）
# ---------------------------------------------------------------------
def read_icepak_report(path):
    """解析 Full report 输出，返回 (xyz_m [M,3] 单位m, T_c [M] 单位°C)。"""
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()

    xyz, T = [], []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 5 and parts[0].isdigit():
            try:
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
                T.append(float(parts[4]))
            except ValueError:
                continue
    xyz = np.array(xyz)
    T = np.array(T)
    print(f'[report] 解析出 {len(T)} 个节点, 温度范围 [{T.min():.2f}, {T.max():.2f}] °C')
    return xyz, T


# ---------------------------------------------------------------------
# 功率 map（平面功率密度，与 3d5 功率段格式一致）
# ---------------------------------------------------------------------
def build_power_map(powers, nx, ny):
    """按 die 区域生成 [nx, ny] 平面功率 map（归一化），含总功率校准。"""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    p = np.zeros((nx, ny))
    for (x0, x1, y0, y1), pw in zip(DIE_REGIONS, powers):
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            continue
        mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        p[mask] += pw / area

    total_w = sum(powers)
    if total_w > 0:
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        integral = float(p.sum() * dx * dy)
        if integral > 0:
            p *= total_w / integral
    return p


def normalize_power(p, mode='sum'):
    if mode == 'none':
        return p
    if mode == 'peak':
        m = float(np.abs(p).max())
        return p / m if m > 0 else p
    s = float(p.sum())
    return p / s if s > 0 else p


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Icepak Full report → 3d5 训练数据')
    ap.add_argument('--report', required=True, help='report.02 文件路径')
    ap.add_argument('--power', required=True,
                    help='6 个 die 功耗(W)："0.28,0.28,0.12,0.12,0.12,0.12"')
    ap.add_argument('--nx', type=int, default=101)
    ap.add_argument('--ny', type=int, default=101)
    ap.add_argument('--nz', type=int, default=56)
    ap.add_argument('--t_amb', type=float, default=20.0,
                    help='环境温度 °C（默认 20=293.15K，与 eval.py 还原公式兼容）')
    ap.add_argument('--t_scale', type=float, default=25.0,
                    help='温度缩放；auto=用数据最大温升')
    ap.add_argument('--power_norm', default='sum', choices=['none', 'sum', 'peak'])
    ap.add_argument('--out_tag', default='icepak')
    args = ap.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    powers = [float(v) for v in args.power.split(',')]
    assert len(powers) == 6, '--power 需要 6 个值'

    # 1) 读散点
    xyz_m, T_c = read_icepak_report(args.report)

    # 2) 坐标归一化 + z 分段映射
    xyz_tr = xyz_m.copy()
    xyz_tr[:, 0] = xyz_m[:, 0] * 1000.0 / X_ICEPAK_MM      # m → mm → 归一化
    xyz_tr[:, 1] = xyz_m[:, 1] * 1000.0 / Y_ICEPAK_MM
    xyz_tr[:, 2] = z_icepak_to_train(xyz_m[:, 2] * 1000.0)  # m → mm → 分段映射
    print(f'归一化坐标范围: x[{xyz_tr[:,0].min():.3f},{xyz_tr[:,0].max():.3f}] '
          f'y[{xyz_tr[:,1].min():.3f},{xyz_tr[:,1].max():.3f}] '
          f'z[{xyz_tr[:,2].min():.3f},{xyz_tr[:,2].max():.3f}]')

    # 3) 插值到规则网格
    if si is None:
        raise ImportError('需要 scipy: pip install scipy')
    x_g = np.linspace(0.0, 1.0, nx)
    y_g = np.linspace(0.0, 1.0, ny)
    z_g = np.linspace(0.0, 0.55, nz)
    gx, gy, gz = np.meshgrid(x_g, y_g, z_g, indexing='ij')
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    T_flat = si.griddata(xyz_tr, T_c, pts, method='linear', fill_value=np.nan)
    T_grid = T_flat.reshape(nx, ny, nz)
    if np.isnan(T_grid).any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(
            np.isnan(T_grid), return_distances=False, return_indices=True)
        T_grid = T_grid[tuple(idx)]
    print(f'插值完成, T 范围 [{T_grid.min():.2f}, {T_grid.max():.2f}] °C')

    # 4) 温度归一化 u = (T - t_amb) / t_scale
    scale = args.t_scale
    if scale == 'auto':
        scale = float((T_grid - args.t_amb).max()) or 1.0
    u_grid = (T_grid - args.t_amb) / scale
    print(f'归一化温度 u 范围 [{u_grid.min():.4f}, {u_grid.max():.4f}]')

    # 5) 三通道输入（功率 | 掩码 | 界面）
    from gen_mask import make_horizontal_mask, make_horizontal_interface
    mask = make_horizontal_mask(nx).reshape(-1)
    interface = make_horizontal_interface(nx).reshape(-1)
    pmap = build_power_map(powers, nx, ny)
    pmap = normalize_power(pmap, args.power_norm).reshape(-1)
    fs_all = np.concatenate([pmap, mask, interface])[None, :]      # [1, 3*n_feat]
    u_all = u_grid[None, ...]                                      # [1, nx, ny, nz]
    T_all = T_grid[None, ...]

    os.makedirs(DATA_DIR, exist_ok=True)
    tag = args.out_tag
    np.save(os.path.join(DATA_DIR, f'fs_{tag}_volume.npy'), fs_all)
    np.save(os.path.join(DATA_DIR, f'u_{tag}_volume.npy'), u_all)
    np.save(os.path.join(DATA_DIR, f'T_{tag}_raw.npy'), T_all)

    meta = {
        'source': args.report,
        'n_samples': 1, 'grid': [nx, ny, nz],
        'powers_w': powers, 't_amb_c': args.t_amb,
        't_scale': 'auto' if args.t_scale == 'auto' else float(args.t_scale),
        'power_norm': args.power_norm,
        'z_map_icepak_mm': Z_ICEPAK_BOUNDS, 'z_map_train': Z_TRAIN_BOUNDS,
        'note': 'u = (T - t_amb)/t_scale; 评估还原 T_C = t_amb + t_scale*u',
    }
    with open(os.path.join(DATA_DIR, f'meta_{tag}.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n完成！')
    print(f'  data/fs_{tag}_volume.npy   {fs_all.shape}  三通道输入')
    print(f'  data/u_{tag}_volume.npy    {u_all.shape}  归一化温度真值')
    print(f'  data/T_{tag}_raw.npy       {T_all.shape}  原始温度 (°C)')


if __name__ == '__main__':
    main()
