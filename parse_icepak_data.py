"""
parse_icepak_data.py
====================
DeepOHeat-V1 3.5D 改造 · 将 Ansys Icepak 导出的温度场 CSV 转为训练/验证数据

背景
----
商业软件（Ansys Icepak）真实仿真数据 → 3d5 模型验证/训练。
Icepak 模型：10mm × 10mm × 1.8mm（Substrate 0~0.5 / Interposer 0.5~1.0 /
TIM 1.0~1.2 / die 1.2~1.8），真实温度 °C、W 级功耗。
训练代码：归一化坐标 x,y∈[0,1]，z∈[0,0.55]（0~0.1 / 0.1~0.35 / 0.35~0.4 / 0.4~0.55）。

本脚本完成两处转换：
  1. z 分段线性映射：Icepak 层边界 → 训练层边界（每层内部线性，物理一一对应）
  2. 温度归一化：u = (T - t_amb) / t_scale（默认论文温标：t_amb=25°C, t_scale=25）

输出（data/ 目录）
-----------------
  fs_icepak_volume.npy   [N, 3*NX*NY]   三通道输入（功率|掩码|界面），与 3d5 模式一致
  u_icepak_volume.npy    [N, NX, NY, NZ] 归一化温度真值（u），用于评估
  T_icepak_raw.npy       [N, NX, NY, NZ] 原始温度（°C），备份/画图
  meta_icepak.json       每个样本的功耗配置与映射参数（复现用）

用法
----
# 1) 单个 CSV（一个功耗配置，N=1）
python parse_icepak_data.py --csv out/temp_0.csv --power "10,10,3,3,3,3"

# 2) 多个 CSV（N 个样本，每个 CSV 一行功耗列表，与文件顺序对应）
python parse_icepak_data.py --csv out/t0.csv,out/t1.csv --power "10,10,3,3,3,3;5,5,2,2,2,2"

# 3) 自定义分辨率 / 温度归一化
python parse_icepak_data.py --csv out/t0.csv --power "10,10,3,3,3,3" \
       --nx 101 --ny 101 --nz 56 --t_amb 25 --t_scale 25

功耗顺序约定（与 gen_mask.py / k_map.py 横向布局一致）：
  [Compute_1, Compute_2, HBM_1, HBM_2, HBM_3, HBM_4] 单位 W
"""

import os
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
# Icepak z 层边界（mm）与训练 z 层边界（归一化）一一对应，分段线性映射
Z_ICEPAK_BOUNDS = [0.0, 0.5, 1.0, 1.2, 1.8]      # Substrate/Interposer/TIM/die
Z_TRAIN_BOUNDS  = [0.0, 0.1, 0.35, 0.4, 0.55]    # 与 k_map.py 的 Z_*_TOP 一致

# 芯片平面：Icepak 10mm → 训练 [0,1]
X_ICEPAK_MM = 10.0
Y_ICEPAK_MM = 10.0

# die 区域（归一化坐标 [x0,x1,y0,y1]），与 Icepak 建模一致，也与
# gen_mask.py / k_map.py 的 horizontal_region 一致：
#   Compute_1/2: 中列 x∈[0.3,0.7]，y 上下两块
#   HBM_1~4: 四角
DIE_REGIONS = [
    (0.30, 0.70, 0.00, 0.49),   # Compute_1
    (0.30, 0.70, 0.51, 1.00),   # Compute_2
    (0.00, 0.29, 0.00, 0.29),   # HBM_1 左下
    (0.71, 1.00, 0.00, 0.29),   # HBM_2 右下
    (0.00, 0.29, 0.71, 1.00),   # HBM_3 左上
    (0.71, 1.00, 0.71, 1.00),   # HBM_4 右上
]


def z_icepak_to_train(z_icepak):
    """Icepak z(mm) → 训练归一化 z 的分段线性映射（任意形状数组）。"""
    zi = np.asarray(z_icepak, dtype=np.float64)
    zt = np.empty_like(zi)
    for (a0, a1, b0, b1) in zip(Z_ICEPAK_BOUNDS[:-1], Z_ICEPAK_BOUNDS[1:],
                                Z_TRAIN_BOUNDS[:-1], Z_TRAIN_BOUNDS[1:]):
        mask = (zi >= a0) & (zi <= a1)
        zt[mask] = b0 + (b1 - b0) * (zi[mask] - a0) / (a1 - a0)
    return zt


# ---------------------------------------------------------------------
# CSV 读取（自动探测列）
# ---------------------------------------------------------------------
def read_icepak_csv(path):
    """读 Icepak 导出的 CSV，返回 (coords [M,3], T [M])。

    自动探测列名：坐标列名含 x/y/z（忽略大小写），温度列名含 temp/temperature/T。
    若探测失败抛异常，提示用 --cols 手动指定。
    """
    import csv
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        sample = f.read(4096)
        f.seek(0)
        # 探测分隔符
        if sample.count(',') >= sample.count(';') and sample.count(',') > 0:
            delim = ','
        else:
            delim = ';'
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    header = rows[0]
    ncol = len(header)

    def find_col(keywords):
        for i, h in enumerate(header):
            hl = h.strip().lower()
            if any(k in hl for k in keywords):
                return i
        return None

    ix, iy, iz = find_col(['x']), find_col(['y']), find_col(['z'])
    # 温度列：排除坐标列后找 temp
    it = None
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if any(k in hl for k in ['temp', 'temperature']) and i not in (ix, iy, iz):
            it = i
            break
    if it is None:  # 退路：单列 'T'
        it = find_col(['t'])

    if ix is None or iy is None or iz is None or it is None:
        raise ValueError(
            f'无法自动识别坐标/温度列。表头: {header}\n'
            f'请用 --cols "x,y,z,temp" 手动指定列名。')

    data = np.array([[r[ix], r[iy], r[iz], r[it]] for r in rows[1:]
                     if len(r) == ncol and r[ix].strip()], dtype=np.float64)
    return data[:, :3], data[:, 3]


# ---------------------------------------------------------------------
# 功率 map（平面功率密度，与 3d5 功率段格式一致）
# ---------------------------------------------------------------------
def build_power_map(powers, nx, ny):
    """按 die 区域生成 [nx, ny] 平面功率 map（归一化坐标）。

    powers: [6] 每个 die 的功耗(W)。
    区域内功率密度 = P / 区域面积（归一化面积），非 die 区为 0。
    返回前做总功率校准：map 数值积分 = 实际总功耗，消除网格离散误差。
    返回 [nx, ny] float64。
    """
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    p = np.zeros((nx, ny))
    for (x0, x1, y0, y1), pw in zip(DIE_REGIONS, powers):
        area = (x1 - x0) * (y1 - y0)          # 归一化面积
        if area <= 0:
            continue
        mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        p[mask] += pw / area

    # 总功率校准：数值积分 ≈ 设定总功耗（网格离散误差校正）
    total_w = sum(powers)
    if total_w > 0:
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        integral = float(p.sum() * dx * dy)
        if integral > 0:
            p *= total_w / integral
    return p


def normalize_power(p, mode='sum'):
    """功率 map 缩放，使数值范围与训练数据接近（原版 volume 功率 ∈ [0, ~6.5]）。

    mode:
      none : 不缩放（保留 W/面积 原始值，量级大，可能训练不稳）
      sum  : 总功率归一化为 1（推荐，数值稳定）
      peak : 峰值归一化为 1
    """
    if mode == 'none':
        return p
    if mode == 'peak':
        m = float(np.abs(p).max())
        return p / m if m > 0 else p
    # sum
    s = float(p.sum())
    return p / s if s > 0 else p


# ---------------------------------------------------------------------
# 插值到规则网格
# ---------------------------------------------------------------------
def interpolate_to_grid(xyz, T, nx, ny, nz):
    """把散点 (xyz [M,3], T [M]) 插值到 [nx, ny, nz] 规则网格。

    网格：x,y ∈ [0,1] 线性；z 用分段映射后的 [0,0.55] 线性。
    返回 (u_grid [nx,ny,nz], x_g [nx], y_g [ny], z_g [nz])。
    """
    if si is None:
        raise ImportError('需要 scipy: pip install scipy')

    xyz_tr = xyz.copy()
    xyz_tr[:, 0] = xyz[:, 0] / X_ICEPAK_MM          # mm → [0,1]
    xyz_tr[:, 1] = xyz[:, 1] / Y_ICEPAK_MM
    xyz_tr[:, 2] = z_icepak_to_train(xyz[:, 2])     # z 分段映射

    x_g = np.linspace(0.0, 1.0, nx)
    y_g = np.linspace(0.0, 1.0, ny)
    z_g = np.linspace(0.0, 0.55, nz)
    gx, gy, gz = np.meshgrid(x_g, y_g, z_g, indexing='ij')

    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    u_flat = si.griddata(xyz_tr, T, pts, method='linear', fill_value=np.nan)
    u_grid = u_flat.reshape(nx, ny, nz)

    # 边界 NaN 用最近有效值填充
    if np.isnan(u_grid).any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(
            np.isnan(u_grid), return_distances=False, return_indices=True)
        u_grid = u_grid[tuple(idx)]
    return u_grid, x_g, y_g, z_g


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Icepak CSV → 3d5 训练/验证数据')
    ap.add_argument('--csv', required=True, help='CSV 路径（多个用逗号分隔）')
    ap.add_argument('--power', required=True,
                    help='功耗列表；单样本 "10,10,3,3,3,3"；多样本用 ; 分隔与 CSV 对应')
    ap.add_argument('--nx', type=int, default=101)
    ap.add_argument('--ny', type=int, default=101)
    ap.add_argument('--nz', type=int, default=56)
    ap.add_argument('--t_amb', type=float, default=20.0,
                    help='环境温度 °C。默认 20 (=293.15K)，与 eval.py 的还原公式 '
                         'T = 25*u + 293.15 完全兼容；若 Icepak 环境设 25°C 可传 25，'
                         '但 MAPE 会引入 5K 基准偏移')
    ap.add_argument('--t_scale', type=float, default=25.0,
                    help='温度缩放（与 eval.py 一致为 25）；传 auto 用数据最大温升')
    ap.add_argument('--power_norm', default='sum', choices=['none', 'sum', 'peak'],
                    help='功率 map 缩放：sum=总功率归一为1(推荐) / peak=峰值归一为1 / none=不缩放')
    ap.add_argument('--out_tag', default='icepak', help='输出文件名标识')
    args = ap.parse_args()

    csvs = [c.strip() for c in args.csv.split(',')]
    power_sets = [list(map(float, p.strip().split(','))) for p in args.power.split(';')]
    assert len(csvs) == len(power_sets), 'CSV 数量与功耗配置数量不一致'
    assert all(len(p) == 6 for p in power_sets), '每行功耗需 6 个值 (2 Compute + 4 HBM)'

    os.makedirs(DATA_DIR, exist_ok=True)
    N = len(csvs)
    nx, ny, nz = args.nx, args.ny, args.nz
    n_feat = nx * ny

    print(f'处理 {N} 个样本，网格 {nx}x{ny}x{nz}，功耗配置: {power_sets}')

    T_all = np.zeros((N, nx, ny, nz))
    u_all = np.zeros((N, nx, ny, nz))
    fs_all = np.zeros((N, 3 * n_feat))

    # 掩码/界面对所有样本相同（同一芯片几何）→ 与 gen_mask.py 布局一致
    from gen_mask import make_horizontal_mask, make_horizontal_interface
    mask = make_horizontal_mask(nx).reshape(-1)
    interface = make_horizontal_interface(nx).reshape(-1)

    for s, (csv_path, powers) in enumerate(zip(csvs, power_sets)):
        print(f'  [{s+1}/{N}] {csv_path}  功耗 {powers} W')
        xyz, T = read_icepak_csv(csv_path)
        print(f'    散点 {xyz.shape[0]} 个, T 范围 [{T.min():.2f}, {T.max():.2f}] °C')

        T_grid, _, _, _ = interpolate_to_grid(xyz, T, nx, ny, nz)
        T_all[s] = T_grid

        # 温度归一化：u = (T - t_amb) / t_scale
        scale = args.t_scale
        if isinstance(scale, str) and scale == 'auto':
            scale = float((T_grid - args.t_amb).max()) or 1.0
        u_all[s] = (T_grid - args.t_amb) / scale

        # 功率 map → 三通道拼接 [功率 | 掩码 | 界面]
        pmap = build_power_map(powers, nx, ny)
        pmap = normalize_power(pmap, args.power_norm)
        pmap = pmap.reshape(-1)
        fs_all[s, :n_feat] = pmap
        fs_all[s, n_feat:2*n_feat] = mask
        fs_all[s, 2*n_feat:] = interface

    tag = args.out_tag
    np.save(os.path.join(DATA_DIR, f'fs_{tag}_volume.npy'), fs_all)     # [N, 3*nx*ny]
    np.save(os.path.join(DATA_DIR, f'u_{tag}_volume.npy'), u_all)       # [N, nx, ny, nz]
    np.save(os.path.join(DATA_DIR, f'T_{tag}_raw.npy'), T_all)          # [N, nx, ny, nz] °C

    meta = {
        'n_samples': N, 'grid': [nx, ny, nz],
        'powers_w': power_sets, 't_amb_c': args.t_amb,
        't_scale': 'auto' if args.t_scale == 'auto' else float(args.t_scale),
        'z_map_icepak_mm': Z_ICEPAK_BOUNDS, 'z_map_train': Z_TRAIN_BOUNDS,
        'die_regions_norm': DIE_REGIONS,
        'u_true': f'data/u_{tag}_volume.npy', 'T_raw': f'data/T_{tag}_raw.npy',
        'note': 'u = (T - t_amb)/t_scale; 评估时还原 T = t_amb + t_scale*u',
    }
    with open(os.path.join(DATA_DIR, f'meta_{tag}.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n完成！输出：')
    print(f'  data/fs_{tag}_volume.npy     {fs_all.shape}  三通道输入')
    print(f'  data/u_{tag}_volume.npy      {u_all.shape}  归一化温度真值')
    print(f'  data/T_{tag}_raw.npy         {T_all.shape}  原始温度 (°C)')
    print(f'  data/meta_{tag}.json         复现参数')


if __name__ == '__main__':
    main()
