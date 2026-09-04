# -*- coding: utf-8 -*-
"""
make_icepak_dataset.py
======================
将 Icepak Full report 导出的多个温度场文件合并，生成 DeepOHeat-V1 3d5 改造
所需的真实仿真数据集（训练集 + 测试集，101x101x56 网格）。

输入
----
一个文本文件，每行一个样本，用空格/逗号分隔，格式：
    <report文件路径>  X1,X2,H1,H2,H3,H4           # 6 个 die 功率(W)
  （或）                   <报告文件> 0.20 0.20 0.08 0.08 0.08 0.08

输出（写到 --out_dir，默认 DeepOHeat-v1/data/）
------------------------------------------------
  fs_train_3d5_volume.npy   [Ntr, 3*101*101]  3d5 训练输入（功率|掩码|界面）
  fs_test_3d5_volume.npy    [Nte, 3*101*101]  3d5 测试输入
  u_train_3d5.npy           [Ntr, 101,101,56] 训练温度真值 u（归一化）
  u_test_3d5.npy            [Nte, 101,101,56] 测试温度真值 u（归一化）
  fs_train_volume.npy       [Ntr, 101*101]    纯功率输入（baseline 训练用）
  fs_test_volume.npy        [Nte, 101*101]    纯功率输入（baseline 测试用）
  u_train_volume.npy        [Ntr, 101,101,56] baseline 训练真值（与 u_train_3d5 相同）
  u_test_volume.npy         [Nte, 101,101,56] baseline 测试真值（与 u_test_3d5 相同）
  T_train_raw.npy / T_test_raw.npy  [N,101,101,56] 原始温度(°C)
  meta_icepak_dataset.json  功率配置、划分、归一化参数（复现用）

归一化约定
----------
  u = (T - t_amb) / t_scale ,  默认 t_amb=20°C, t_scale=25
  （与 eval.py 的还原公式 T = t_amb + 25*u 完全兼容，MAPE 无基准偏移）

用法
----
python make_icepak_dataset.py samples.txt --out_dir data
"""
import os
import io
import sys
import json
import glob
import re
import argparse
import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator, interp1d
except ImportError:
    RegularGridInterpolator = None
    interp1d = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'data')

# ---------------------------------------------------------------------
# 几何映射（与 parse_icepak_report.py / gen_mask.py / k_map.py 完全一致）
# ---------------------------------------------------------------------
X_ICEPAK_MM = 10.0
Y_ICEPAK_MM = 10.0
Z_ICEPAK_BOUNDS = [0.0, 0.5, 1.0, 1.2, 1.8]     # Substrate/Interposer/TIM/die
Z_TRAIN_BOUNDS = [0.0, 0.1, 0.35, 0.4, 0.55]    # 与 k_map.py 一致

DIE_REGIONS = [
    (0.30, 0.70, 0.00, 0.49),   # Compute_1
    (0.30, 0.70, 0.51, 1.00),   # Compute_2
    (0.00, 0.29, 0.00, 0.29),   # HBM_1
    (0.71, 1.00, 0.00, 0.29),   # HBM_2
    (0.00, 0.29, 0.71, 1.00),   # HBM_3
    (0.71, 1.00, 0.71, 1.00),   # HBM_4
]

NX, NY, NZ = 101, 101, 56       # 训练网格分辨率
T_AMB = 20.0                    # 环境温度 °C（eval.py 兼容）
T_SCALE = 25.0                  # 温度缩放（eval.py 兼容）

# 固定测试集：报告编号（见《功率清单100组.md》，含 3 极端 + 7 常规，勿改）
TEST_REPORT_NUMS = {16, 19, 22, 25, 26, 27, 59, 80, 86, 93}


def read_icepak_report(path):
    """解析 Full report：返回 (xyz_m [M,3] 单位m, T_c [M] 单位°C)。"""
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
    return xyz, T


def build_power_map(powers, nx=NX, ny=NY):
    """按 die 区域生成 [nx, ny] 平面功率 map（归一化），含总功率校准。"""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    p = np.zeros((nx, ny))
    for (x0, x1, y0, y1), pw in zip(DIE_REGIONS, powers):
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            continue
        m = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        p[m] += pw / area
    total_w = sum(powers)
    if total_w > 0:
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        integral = float(p.sum() * dx * dy)
        if integral > 0:
            p *= total_w / integral
    return p


def to_grid(xyz_m, T_c, nx=NX, ny=NY, nz=NZ):
    """稀疏规则网格散点 → [nx, ny, nz] 规则网格（快速、物理正确）。

    Icepak Full report 导出的是规律网格（x/y 均匀 21 点、z 分层 20 层）。
    这里：
      1) 只保留固体部分（z <= 1.8mm，舍弃上方空气层）；
      2) x/y 方向双线性插值到 [nx, ny]；
      3) z 方向按物理层分段（Substrate/Interposer/TIM/die），段内线性插值到目标层。
    """
    if RegularGridInterpolator is None:
        raise RuntimeError('需要 scipy')
    # 1) 只保留固体部分
    solid = xyz_m[:, 2] <= 1.8e-3
    xyz_s, T_s = xyz_m[solid], T_c[solid]

    xs = np.unique(np.round(xyz_s[:, 0], 6))
    ys = np.unique(np.round(xyz_s[:, 1], 6))
    zs = np.unique(np.round(xyz_s[:, 2], 6))
    zs_mm = zs * 1000.0

    xi = np.searchsorted(xs, np.round(xyz_s[:, 0], 6))
    yi = np.searchsorted(ys, np.round(xyz_s[:, 1], 6))
    zi = np.searchsorted(zs, np.round(xyz_s[:, 2], 6))
    v = np.zeros((len(xs), len(ys), len(zs)))
    v[xi, yi, zi] = T_s

    xg = np.linspace(0.0, 1e-2, nx)          # [m] 物理坐标
    yg = np.linspace(0.0, 1e-2, ny)
    tg = np.linspace(0.0, 0.55, nz)          # 训练 z 坐标
    XX, YY = np.meshgrid(xg, yg, indexing='ij')

    # 2) 每个源 z 层的 x/y 双线性插值 -> [NSZ, nx, ny]
    planes = np.stack([
        RegularGridInterpolator((xs, ys), v[:, :, k], method='linear',
                                bounds_error=False, fill_value=None)((XX, YY))
        for k in range(len(zs))
    ])

    # 3) z 方向分段线性映射（物理段 -> 目标段）
    out = np.empty((nx, ny, nz))
    for (a0, a1, b0, b1) in zip(Z_ICEPAK_BOUNDS[:-1], Z_ICEPAK_BOUNDS[1:],
                                Z_TRAIN_BOUNDS[:-1], Z_TRAIN_BOUNDS[1:]):
        m = (zs_mm >= a0 - 1e-9) & (zs_mm <= a1 + 1e-9)
        zseg, kseg = zs_mm[m], np.where(m)[0]
        if len(zseg) == 0:
            continue
        zt_seg = b0 + (b1 - b0) * (zseg - a0) / (a1 - a0)   # 源层在目标坐标系的位置
        f_z = interp1d(zt_seg, planes[kseg], axis=0, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
        seg_mask = (tg >= b0 - 1e-9) & (tg <= b1 + 1e-9)
        tidx = np.where(seg_mask)[0]
        vals = f_z(tg[tidx])                       # [M, nx, ny]
        out[:, :, tidx] = np.moveaxis(vals, 0, -1)
    return out


def parse_samples(path):
    """读样本清单文件：每行 <report路径> 空格/逗号分隔的6个功率。"""
    samples = []
    with open(path, 'r', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#') or ln.startswith('file'):
                continue
            parts = [p for p in ln.replace(',', ' ').split() if p.strip()]
            if len(parts) < 7:
                print(f'[跳过] 行格式不对: {ln}')
                continue
            rpath, powers = parts[0], [float(v) for v in parts[1:7]]
            if not os.path.exists(rpath):
                print(f'[跳过] 文件不存在: {rpath}')
                continue
            samples.append(dict(report=os.path.abspath(rpath),
                                powers=powers,
                                powers_s=' '.join(f'{v:g}' for v in powers)))
    return samples


def main():
    ap = argparse.ArgumentParser(description='Icepak 多 report → 3d5 数据集')
    ap.add_argument('samples', help='样本清单文件（每行: 报告路径 + 6个功率）')
    ap.add_argument('--out_dir', default=DEFAULT_OUT, help='输出目录')
    ap.add_argument('--t_amb', type=float, default=T_AMB)
    ap.add_argument('--t_scale', type=float, default=T_SCALE)
    ap.add_argument('--power_norm', default='unified', choices=['unified', 'none'],
                    help='功率 map 缩放: unified=全数据集统一缩放(峰值=5,推荐) / none=原始功率密度')
    args = ap.parse_args()

    samples = parse_samples(args.samples)
    if len(samples) < 2:
        print('有效样本不足 2 个，无法构成训练+测试集')
        sys.exit(1)
    n = len(samples)
    print(f'[i] 有效样本 {n} 个:')
    for s in samples:
        print(f'    {os.path.basename(s["report"]):16s}  功率 {s["powers_s"]}')

    os.makedirs(args.out_dir, exist_ok=True)
    nx, ny, nz = NX, NY, NZ
    n_feat = nx * ny

    # 掩码 / 界面（所有样本相同，101 分辨率）
    sys.path.insert(0, HERE)
    from gen_mask import make_horizontal_mask, make_horizontal_interface
    mask = make_horizontal_mask(nx).reshape(-1)
    interface = make_horizontal_interface(nx).reshape(-1)

    # 第一遍：只构建原始功率密度图（不做任何归一化），用于计算全数据集统一缩放系数。
    # 这样既保留样本间绝对功率差异（冷-热可区分），又把量级对齐到原版训练数据（0~8）。
    pmap_raw = [build_power_map(s['powers'], nx, ny).reshape(-1) for s in samples]
    peak_all = max(float(p.max()) for p in pmap_raw)
    if args.power_norm == 'unified':
        scale = 5.0 / peak_all          # 全数据集最大峰值对齐到 5（与原版同量级）
        print(f'[功率] 全数据集峰值 {peak_all:.4f} -> 统一缩放 x{scale:.2f} (峰值=5)')
    elif args.power_norm == 'none':
        scale = 1.0
    else:
        raise ValueError(f'不支持的功率归一化: {args.power_norm}（只支持 unified/none）')

    fs = np.zeros((n, 3 * n_feat))
    u = np.zeros((n, nx, ny, nz))
    T = np.zeros((n, nx, ny, nz))

    for i, s in enumerate(samples):
        print(f'[{i+1}/{n}] {os.path.basename(s["report"])}', end=' ')
        xyz_m, T_c = read_icepak_report(s['report'])
        T_grid = to_grid(xyz_m, T_c, nx, ny, nz)
        pmap = pmap_raw[i] * scale
        fs[i, :n_feat] = pmap
        fs[i, n_feat:2*n_feat] = mask
        fs[i, 2*n_feat:] = interface
        u[i] = (T_grid - args.t_amb) / args.t_scale
        T[i] = T_grid
        print(f'节点{xyz_m.shape[0]}  T[{T_grid.min():.2f},{T_grid.max():.2f}]°C  '
              f'功率峰值{pmap.max():.3f}')

    # 数据集划分：按报告编号固定测试集（TEST_REPORT_NUMS，见《功率清单100组.md》）。
    # 不再使用"前80%/后20%"——那样会把极端热点样本按顺序排进测试集，导致对比不公平。
    def report_num(s):
        base = os.path.basename(s['report'])
        m = re.search(r'(\d+)', base)
        return int(m.group(1)) if m else -1

    i_te = [i for i, s in enumerate(samples) if report_num(s) in TEST_REPORT_NUMS]
    i_tr = [i for i in range(n) if i not in set(i_te)]
    if not i_te:
        print('[拆分] 警告: 测试集为空！样本清单中未找到任何固定测试报告编号')
        print('       请确认清单文件名形如 report.16 / report.100 等')
        sys.exit(1)
    print(f'\n[拆分] 训练 {len(i_tr)} 个 / 测试 {len(i_te)} 个 (固定测试集)')
    print(f'[拆分] 测试集: ' + ' '.join(os.path.basename(samples[i]['report']) for i in i_te))

    def save(name, arr):
        np.save(os.path.join(args.out_dir, name), arr)
        print(f'  写出 {name:34s} {arr.shape}')

    # 3d5 三通道 + baseline 纯功率 + 真值
    save('fs_train_3d5_volume.npy', fs[i_tr])
    save('fs_test_3d5_volume.npy', fs[i_te])
    save('u_train_3d5.npy', u[i_tr])
    save('u_test_3d5.npy', u[i_te])
    save('fs_train_volume.npy', fs[i_tr, :n_feat])
    save('fs_test_volume.npy', fs[i_te, :n_feat])
    save('u_train_volume.npy', u[i_tr])
    save('u_test_volume.npy', u[i_te])
    save('T_train_raw.npy', T[i_tr])
    save('T_test_raw.npy', T[i_te])

    meta = {
        'n_samples': n, 'n_train': len(i_tr), 'n_test': len(i_te),
        'grid': [nx, ny, nz],
        't_amb_c': args.t_amb, 't_scale': args.t_scale,
        'power_norm': args.power_norm,
        'power_scale': scale, 'power_peak_all': float(peak_all),
        'train': [os.path.basename(s['report']) for s in [samples[k] for k in i_tr]],
        'test': [os.path.basename(s['report']) for s in [samples[k] for k in i_te]],
        'samples': [{'file': os.path.basename(s['report']),
                     'powers_w': s['powers']} for s in samples],
        'note': 'u=(T-20)/25; 评估还原 T_C = 20 + 25*u (eval.py 兼容)',
    }
    with open(os.path.join(args.out_dir, 'meta_icepak_dataset.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'\n完成！输出在 {args.out_dir}')
    print('  3d5 模式 训练: data/fs_train_3d5_volume.npy + data/u_train_3d5.npy')
    print('  3d5 模式 测试: data/fs_test_3d5_volume.npy  + data/u_test_3d5.npy')
    print('  baseline 训练: data/fs_train_volume.npy + data/u_train_volume.npy')
    print('  baseline 测试: data/fs_test_volume.npy  + data/u_test_volume.npy')


if __name__ == '__main__':
    main()