"""
parse_icepak_cfd.py
===================
DeepOHeat-V1 3.5D 改造 · 从 Icepak 导出的 CFD 文件生成 3d5 训练/验证数据

背景
----
Icepak 的 "Workflow data" 导出会生成两个文件：
  - 3d5_chip00.cfd.cas  （网格：节点坐标 + 单元定义）
  - 3d5_chip00.cfd.dat  （结果：每个 cell zone 的 T/P/U/V/W）

本脚本把它们合并，得到「坐标(x,y,z) + 温度 T」的散点场，
再插值到 3d5 训练用的规则网格 [NX, NY, NZ]（归一化坐标），
最后输出与 heat_volumetric.py --mode 3d5 兼容的 npy 数据。

输出（data/ 目录）
-----------------
  fs_icepak_volume.npy   [N, 3*NX*NY]   三通道输入（功率|掩码|界面）
  u_icepak_volume.npy    [N, NX, NY, NZ] 归一化温度真值 u
  T_icepak_raw.npy       [N, NX, NY, NZ] 原始温度（K）

用法
----
python parse_icepak_cfd.py --cas C:/Users/25164/Desktop/3d5_chip/3d5_chip00.cfd.cas \
       --dat C:/Users/25164/Desktop/3d5_chip/3d5_chip00.cfd.dat \
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


def z_icepak_to_train(z_icepak):
    zi = np.asarray(z_icepak, dtype=np.float64)
    zt = np.empty_like(zi)
    for (a0, a1, b0, b1) in zip(Z_ICEPAK_BOUNDS[:-1], Z_ICEPAK_BOUNDS[1:],
                                Z_TRAIN_BOUNDS[:-1], Z_TRAIN_BOUNDS[1:]):
        mask = (zi >= a0) & (zi <= a1)
        zt[mask] = b0 + (b1 - b0) * (zi[mask] - a0) / (a1 - a0)
    return zt


# ---------------------------------------------------------------------
# cas 网格解析（节点坐标 + 单元节点）
# ---------------------------------------------------------------------
def parse_cas(cas_path):
    """解析 Fluent cas：返回 (nodes [N,3], cell_centers [M,3], zone_cells {zone: [cell_idx...]})。

    Fluent cas 格式：
      (0 "Nodes:") 段: (10 (0 1 1 1) (x0 y0 z0 x1 y1 z1 ...))
      (0 "Cells:"/ "Cell zone ...") 段: 单元->节点连接
    简化：只提取节点坐标 + 每个 cell zone 的单元质心。
    """
    with open(cas_path, 'r', errors='ignore') as f:
        content = f.read()

    # --- 节点坐标: (10 (0 1 1 1) (x0 y0 z0 x1 ...)) ---
    m = re.search(r'\(10 \(0 1 1 1\)\s*\((.*?)\)\s*\)', content, re.S)
    if not m:
        raise ValueError('cas 中找不到节点坐标段 (10 ...)')
    coords = [float(v) for v in m.group(1).split()]
    n_nodes = len(coords) // 3
    nodes = np.array(coords).reshape(n_nodes, 3)          # [N,3] 单位 m
    print(f'[cas] 节点数 {n_nodes}')

    # --- 单元区: 每个 cell zone 的单元与节点连接 ---
    # Fluent: (0 "Cell zone 57, 1200 hexa cells:") ... (12 (zone 57 1 0 1 1200) ...)
    # 简化：用单元数据段 (13 ...) 节点连接，按 zone 分组
    # 先找所有 zone 标题行，确定每个 zone 的单元数
    zone_titles = {}
    for mz in re.finditer(r'\(0 "([^"]*zone\s+(\d+)[^"]*?(\d+)\s*(hexa|tetra|poly|cells)[^"]*)"\)', content, re.I):
        zone_titles[int(mz.group(2))] = int(mz.group(3))

    # 单元-节点连接: (13 (znode zone start end ...) (n1 n2 n3 ...))
    # hexa: 每单元 8 节点; 用 zone 顺序匹配
    # 简化做法：直接找 (13 ...) 段，按 zone 数切分
    hexa_conn = re.search(r'\(13 \(0 \d+ \d+ \d+ \d+\)\s*\((.*?)\)\s*\)', content, re.S)
    if hexa_conn:
        conn_vals = [int(v) for v in hexa_conn.group(1).split()]
        n_cells = len(conn_vals) // 8
        conn = np.array(conn_vals).reshape(n_cells, 8)    # [M, 8] 节点索引
        print(f'[cas] hexa 单元数 {n_cells}')
    else:
        # 尝试 poly 单元
        poly_conn = re.search(r'\(39 \([^)]*\)\s*\((.*?)\)\s*\)', content, re.S)
        if poly_conn:
            conn_vals = [int(v) for v in poly_conn.group(1).split()]
            print(f'[cas] poly 单元连接, 值数 {len(conn_vals)}')
            # 格式: 每单元 [nfaces, face1..] 复杂，暂不支持
            raise ValueError('poly 单元暂不支持，请用 hexa 网格')
        else:
            raise ValueError('cas 中找不到单元连接段')

    # 单元质心（求平均）
    cell_centers = nodes[conn].mean(axis=1)               # [M,3]
    return nodes, cell_centers, zone_titles, conn


# ---------------------------------------------------------------------
# cfd.dat 结果解析（每个 zone 的温度）
# ---------------------------------------------------------------------
def parse_dat(dat_path):
    """解析 cfd.dat：返回 {zone: {var: np.array}}。

    段格式: (0 "SV_T, domain 1, cell zone 57, 1200 cells:") 后跟 (300 (...) (values))
    """
    with open(dat_path, 'r', errors='ignore') as f:
        content = f.read()

    data = {}
    # 匹配段: (0 "SV_X, domain 1, cell zone N, M cells:") (300 (...) (values)) 
    for m in re.finditer(
            r'\(0 "SV_([A-Z]+), domain \d+, cell zone (\d+), (\d+) cells:"\)\s*'
            r'\(300 \([^)]*\)\s*\((.*?)\)\s*\)', content, re.S):
        var, zone, ncells, vtext = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        vals = np.array([float(v) for v in vtext.split()])
        assert len(vals) == ncells, f'zone {zone} {var}: 期望 {ncells} 得到 {len(vals)}'
        data.setdefault(zone, {})[var] = vals
    print(f'[dat] 解析出 {len(data)} 个 cell zone')
    return data


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
    ap = argparse.ArgumentParser(description='Icepak CFD 文件 → 3d5 训练数据')
    ap.add_argument('--cas', required=True, help='cfd.cas 路径')
    ap.add_argument('--dat', required=True, help='cfd.dat 路径')
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
    n_feat = nx * ny
    powers = [float(v) for v in args.power.split(',')]
    assert len(powers) == 6, '--power 需要 6 个值'

    print('解析网格 (cas)...')
    nodes, cell_centers, zone_titles, conn = parse_cas(args.cas)
    print('解析结果 (dat)...')
    zone_data = parse_dat(args.dat)

    # 收集所有固体 zone 的 质心坐标 + 温度
    xyz_list, T_list = [], []
    for zone, vars_dict in zone_data.items():
        if 'T' not in vars_dict:
            continue
        # 找到该 zone 的单元在全局单元中的索引
        # 简化：按单元顺序累积（cas 里 hexa 单元按 zone 顺序排列）
        # 这里用 zone_titles 提供单元数顺序匹配
        T_vals = vars_dict['T']
        # 单元质心：cas 中所有 hexa 单元按 zone 顺序排列，但 zone 顺序与 dat 一致
        # 用 dat 的 zone 顺序累计单元
        if zone not in zone_titles:
            # 用 T 数量推断
            n_c = len(T_vals)
        else:
            n_c = zone_titles[zone]
        xyz_list.append(cell_centers[0:n_c])  # 占位，下面修正
        T_list.append(T_vals)

    # 更稳妥：直接按 cell zone 顺序从 conn 中切分单元
    # 重新组织：dat 中 zone 顺序 = cas 中 zone 顺序（Icepak 导出保证）
    # 先按 dat 中出现的 zone 顺序确定单元切片
    ordered_zones = []
    # 重新解析 dat 保留顺序
    with open(args.dat, 'r', errors='ignore') as f:
        dat_content = f.read()
    zone_order = []
    for mz in re.finditer(r'\(0 "SV_T, domain \d+, cell zone (\d+), (\d+) cells:"\)', dat_content):
        zone_order.append((int(mz.group(1)), int(mz.group(2))))

    xyz_all, T_all = [], []
    start = 0
    for zone, ncells in zone_order:
        if zone not in zone_data or 'T' not in zone_data[zone]:
            continue
        T_all.append(zone_data[zone]['T'])
        xyz_all.append(cell_centers[start:start+ncells])
        start += ncells
    xyz = np.concatenate(xyz_all, axis=0)      # [M,3] m
    T = np.concatenate(T_all, axis=0)          # [M] K
    print(f'总散点数 {len(T)}，T 范围 [{T.min():.2f}, {T.max():.2f}] K')

    # 坐标归一化 + z 分段映射
    xyz_tr = xyz.copy()
    xyz_tr[:, 0] = xyz[:, 0] / (X_ICEPAK_MM / 1000.0)      # m → mm → 归一化
    xyz_tr[:, 1] = xyz[:, 1] / (Y_ICEPAK_MM / 1000.0)
    xyz_tr[:, 2] = z_icepak_to_train(xyz[:, 2] * 1000.0)   # m → mm → 分段映射

    # 检查坐标范围
    print(f'归一化坐标范围: x[{xyz_tr[:,0].min():.3f},{xyz_tr[:,0].max():.3f}] '
          f'y[{xyz_tr[:,1].min():.3f},{xyz_tr[:,1].max():.3f}] '
          f'z[{xyz_tr[:,2].min():.3f},{xyz_tr[:,2].max():.3f}]')

    # 插值到规则网格
    if si is None:
        raise ImportError('需要 scipy: pip install scipy')
    x_g = np.linspace(0.0, 1.0, nx)
    y_g = np.linspace(0.0, 1.0, ny)
    z_g = np.linspace(0.0, 0.55, nz)
    gx, gy, gz = np.meshgrid(x_g, y_g, z_g, indexing='ij')
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    T_flat = si.griddata(xyz_tr, T, pts, method='linear', fill_value=np.nan)
    T_grid = T_flat.reshape(nx, ny, nz)
    if np.isnan(T_grid).any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(
            np.isnan(T_grid), return_distances=False, return_indices=True)
        T_grid = T_grid[tuple(idx)]
    print(f'插值完成, T 范围 [{T_grid.min():.2f}, {T_grid.max():.2f}] K '
          f'({T_grid.min()-273.15:.1f} ~ {T_grid.max()-273.15:.1f} °C)')

    # 温度归一化 u = (T - t_amb_K) / t_scale
    t_amb_K = args.t_amb + 273.15
    scale = args.t_scale
    if scale == 'auto':
        scale = float((T_grid - t_amb_K).max()) or 1.0
    u_grid = (T_grid - t_amb_K) / scale
    print(f'归一化温度 u 范围 [{u_grid.min():.4f}, {u_grid.max():.4f}]')

    # 三通道输入
    from gen_mask import make_horizontal_mask, make_horizontal_interface
    mask = make_horizontal_mask(nx).reshape(-1)
    interface = make_horizontal_interface(nx).reshape(-1)
    pmap = build_power_map(powers, nx, ny)
    pmap = normalize_power(pmap, args.power_norm).reshape(-1)
    fs = np.concatenate([pmap, mask, interface])           # [3*n_feat]
    fs_all = fs[None, :]                                    # [1, 3*n_feat]
    u_all = u_grid[None, ...]                              # [1, nx, ny, nz]
    T_all2 = T_grid[None, ...]

    os.makedirs(DATA_DIR, exist_ok=True)
    tag = args.out_tag
    np.save(os.path.join(DATA_DIR, f'fs_{tag}_volume.npy'), fs_all)
    np.save(os.path.join(DATA_DIR, f'u_{tag}_volume.npy'), u_all)
    np.save(os.path.join(DATA_DIR, f'T_{tag}_raw.npy'), T_all2)

    meta = {
        'source': [args.cas, args.dat],
        'n_samples': 1, 'grid': [nx, ny, nz],
        'powers_w': powers, 't_amb_k': t_amb_K,
        't_scale': 'auto' if args.t_scale == 'auto' else float(args.t_scale),
        'power_norm': args.power_norm,
        'z_map_icepak_mm': Z_ICEPAK_BOUNDS, 'z_map_train': Z_TRAIN_BOUNDS,
        'note': 'u = (T - t_amb_K)/t_scale; 评估还原 T_C = t_amb + t_scale*u',
    }
    with open(os.path.join(DATA_DIR, f'meta_{tag}.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n完成！')
    print(f'  data/fs_{tag}_volume.npy   {fs_all.shape}  三通道输入')
    print(f'  data/u_{tag}_volume.npy    {u_all.shape}  归一化温度真值')
    print(f'  data/T_{tag}_raw.npy       {T_all2.shape}  原始温度 (K)')


if __name__ == '__main__':
    main()
