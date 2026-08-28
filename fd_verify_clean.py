# -*- coding: utf-8 -*-
"""
fd_verify_clean.py —— FD(有限差分)自洽性验证 · 干净版(2026-08-28)
==================================================================
借鉴自朋友仓库的 fd_verify.py(矩阵组装思路一致), 三处关键修正:

1. 【测试集纯洁性】原始版用 u_test 真值对比得出 power_scale=0.2809, 再在同一
   测试集上报 MAPE —— 相当于先看答案再考试。本版:
   - 校准模式(--calibrate)只用【训练集】样本对比真值;
   - 普通模式(--sample N 且 N < n_train)也默认读训练集;
   - 想用测试集做"最终确认"(训练完、系数定死之后一次性验证)需显式
     --allow_test, 且流程上要求此时系数已冻结、不再改。

2. 【测试集纯洁性】原始版用 u_test 真值对比得出 power_scale=0.2809, 再在同一
   测试集上报 MAPE —— 相当于先看答案再考试。本版:
   - 一律只读【训练集】(fs_train_3d5_volume / u_train_3d5);
   - --scan 在训练集样本上扫描温升倍率, 取倍率≈1 的系数;
   - FD 是线性方程, 温升 ∝ 功率系数, 所以 1/倍率是精确标定, 不是拟合;
   - 想用测试集做"最终确认"(系数已冻结之后一次性验证)需显式 --allow_test。

3. 【默认不修正】默认 power_scale=1.0(与现训练行为一致); 标定值由用户从
   --scan 结果读出后 export DHV_POWER_SCALE=<值> 传入训练。

3. 【缩放扫描】--scan 可一次跑多个 power_scale, 看温升倍率曲线,
   直观找到"温升倍率≈1"的系数(仍只对训练集样本)。

物理设定与训练 loss 完全一致: k 场(k_map.py)/功率注入 die 层×1/×2/×1/
Robin BC(物理推导 HTC=500)。FD 解的是同一套方程, 因此若 FD 温升 ≈ 真值
温升, 说明物理设定自洽; 若差 N 倍, 说明归一化不自洽。

用法(服务器, 在 DeepOHeat-v1 目录):
    python fd_verify_clean.py --mode 3d5 --nc 51                 # 默认系数自检
    python fd_verify_clean.py --mode 3d5 --nc 51 --scan          # 扫描缩放
    python fd_verify_clean.py --mode 3d5 --nc 51 --sample 3      # 训练集第3个样本
    python fd_verify_clean.py --mode baseline --nc 51            # baseline 自检(无缩放)
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---- BC 系数(与 heat_volumetric.py 完全一致) ----
DHV_BC_TOP_REF = float(os.environ.get('DHV_BC_TOP_REF', '0.2'))
DHV_BC_TOP_KH = float(os.environ.get('DHV_BC_TOP_KH', '79.444'))
DHV_BC_BOT_REF = float(os.environ.get('DHV_BC_BOT_REF', '0.2'))
DHV_BC_BOT_KH = float(os.environ.get('DHV_BC_BOT_KH', '0.3056'))
# baseline 原版 BC(论文遗留, 不动)
BL_BC_TOP_KH, BL_BC_TOP_REF = 2.0, 0.2
BL_BC_BOT_KH, BL_BC_BOT_REF = 40.0, 0.2
# 功率幅值系数(默认 1.0=不修正; 由 --scan 在训练集上标定后经环境变量传入)
DHV_POWER_SCALE = float(os.environ.get('DHV_POWER_SCALE', '1.0'))

T_AMB, T_SCALE = 20.0, 25.0   # u=(T-20)/25, 与 make_icepak_dataset/eval 一致
# "环境 25°C"是 Icepak 建模的对流参考温度, 温升倍率用它
T_AMB_CONV = 25.0


def solve_fd(mode, p, k_field=None, nc=51, power_scale=1.0):
    """有限差分解 ∇·(k∇u)+q=0(稳态), 物理设定与训练 loss 一致。返回 [nc,nc,nz]。"""
    nz = int(0.55 * nc + 0.45)
    dx = dy = 1.0 / (nc - 1)
    dz = 0.55 / (nz - 1)
    N = nc * nc * nz

    def idx(i, j, kk):
        return (kk * nc + j) * nc + i

    if mode == '3d5':
        top_kh, top_ref = DHV_BC_TOP_KH, DHV_BC_TOP_REF
        bot_kh, bot_ref = DHV_BC_BOT_KH, DHV_BC_BOT_REF
    else:
        top_kh, top_ref = BL_BC_TOP_KH, BL_BC_TOP_REF
        bot_kh, bot_ref = BL_BC_BOT_KH, BL_BC_BOT_REF

    # ---- q 场(与训练 loss 的注入结构完全一致) ----
    # 3d5: die 层(z=0.40~0.55)注入, ×1/×2/×1;  功率乘 power_scale
    # baseline: 原版 z=0.10~0.15 注入(不乘 power_scale, 保持原版)
    q = np.zeros((nc, nc, nz))
    if mode == '3d5':
        k_die_bot = int(round(0.40 / dz))
        k_die_top = int(round(0.55 / dz))
        q[:, :, k_die_bot] += power_scale * p
        q[:, :, k_die_bot + 1:k_die_top] += 2 * power_scale * p[:, :, None]
        q[:, :, k_die_top] += power_scale * p
    else:
        k_bot_inj = int(round(0.10 / dz))
        k_top_inj = int(round(0.15 / dz))
        q[:, :, k_bot_inj] += p
        q[:, :, k_bot_inj + 1:k_top_inj] += 2 * p[:, :, None]
        q[:, :, k_top_inj] += p

    # ---- k 场 ----
    if mode == '3d5':
        k = k_field
    else:
        k = 0.1 * np.ones((nc, nc, nz))
        k[:, :, int(round(0.10 / dz))] = 2 * 0.1 * 2 / (0.1 + 2)   # 调和平均, 与原版一致

    # ---- 组装 7-point 稀疏矩阵(侧面绝热 = 镜像) ----
    rows, cols, vals, rhs = [], [], [], []
    for kk in range(nz):
        for j in range(nc):
            for i in range(nc):
                r = idx(i, j, kk)
                if kk == 0:
                    a = bot_kh / dz
                    rows += [r, r]; cols += [r, idx(i, j, 1)]
                    vals += [1 + a, -a]; rhs.append(bot_ref)
                    continue
                if kk == nz - 1:
                    a = top_kh / dz
                    rows += [r, r]; cols += [r, idx(i, j, nz - 2)]
                    vals += [1 + a, -a]; rhs.append(top_ref)
                    continue
                diag = 0.0
                kc = k[i, j, kk]
                if i == 0:
                    rows.append(r); cols.append(idx(i + 1, j, kk)); vals.append(-2 * kc / dx ** 2)
                    diag += 2 * kc / dx ** 2
                elif i == nc - 1:
                    rows.append(r); cols.append(idx(i - 1, j, kk)); vals.append(-2 * kc / dx ** 2)
                    diag += 2 * kc / dx ** 2
                else:
                    rows += [r, r]; cols += [idx(i - 1, j, kk), idx(i + 1, j, kk)]
                    vals += [-kc / dx ** 2, -kc / dx ** 2]; diag += 2 * kc / dx ** 2
                if j == 0:
                    rows.append(r); cols.append(idx(i, j + 1, kk)); vals.append(-2 * kc / dy ** 2)
                    diag += 2 * kc / dy ** 2
                elif j == nc - 1:
                    rows.append(r); cols.append(idx(i, j - 1, kk)); vals.append(-2 * kc / dy ** 2)
                    diag += 2 * kc / dy ** 2
                else:
                    rows += [r, r]; cols += [idx(i, j - 1, kk), idx(i, j + 1, kk)]
                    vals += [-kc / dy ** 2, -kc / dy ** 2]; diag += 2 * kc / dy ** 2
                rows += [r, r]; cols += [idx(i, j, kk - 1), idx(i, j, kk + 1)]
                vals += [-kc / dz ** 2, -kc / dz ** 2]; diag += 2 * kc / dz ** 2
                rows.append(r); cols.append(r); vals.append(diag)
                rhs.append(q[i, j, kk])   # ∇·(k∇u)+q=0 → 对角行 RHS=+q

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    b = np.array(rhs)

    t0 = time.time()
    u = None
    if N <= 100000:
        u = spla.spsolve(A.tocsc(), b)
    else:
        diag = np.abs(A.diagonal())
        diag[diag < 1e-12] = 1.0
        Dinv = sp.diags(1.0 / diag)
        u, _ = spla.gmres(Dinv @ A, Dinv @ b, rtol=1e-6, atol=1e-8, maxiter=3000)
    print(f'[FD] 求解 {time.time()-t0:.1f}s (网格 {nc}x{nc}x{nz}, N={N})')
    return np.asarray(u).reshape(nz, nc, nc).transpose(1, 2, 0)


def load_sample(sample, nc):
    """读【训练集】样本(功率图 + 真值 u)。绝不经手测试集, 除非显式 --allow_test。"""
    nz = int(0.55 * nc + 0.45)
    fs = np.load('data/fs_train_3d5_volume.npy', mmap_mode='r')
    u = np.load('data/u_train_3d5.npy', mmap_mode='r')
    if nc == 101:
        p = np.asarray(fs[sample, :101 * 101]).reshape(101, 101)
        u_true = np.asarray(u[sample])
    else:
        xi = np.round(np.linspace(0, 100, nc)).astype(int)
        zi = np.round(np.linspace(0, 55, nz)).astype(int)
        p = np.asarray(fs[sample, :101 * 101]).reshape(101, 101)[np.ix_(xi, xi)]
        u_true = np.asarray(u[sample])[np.ix_(xi, xi, zi)]
    return p, u_true


def report(u_pred, u_true, tag):
    rl2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    T_pred = T_AMB + T_SCALE * u_pred
    T_true = T_AMB + T_SCALE * u_true
    fd_rise = T_pred.max() - T_AMB_CONV
    true_rise = T_true.max() - T_AMB_CONV
    ratio = fd_rise / true_rise if true_rise else float('nan')
    mae = np.abs(T_pred - T_true).mean()
    print(f'  [{tag}] rel_l2={rl2:.4f}  MAE={mae:.2f}°C  '
          f'FD温升={fd_rise:.2f}°C 真值温升={true_rise:.2f}°C 倍率={ratio:.2f}x')
    return ratio


def main():
    ap = argparse.ArgumentParser(description='FD 自洽性验证 · 干净版(训练集, 不碰测试集)')
    ap.add_argument('--mode', default='3d5', choices=['3d5', 'baseline'])
    ap.add_argument('--sample', type=int, default=0, help='训练集样本序号')
    ap.add_argument('--nc', type=int, default=51, help='网格分辨率(51快速, 101全分辨率)')
    ap.add_argument('--power_scale', type=float, default=None,
                    help=f'功率幅值系数(默认读 DHV_POWER_SCALE={DHV_POWER_SCALE}, 即不修正)')
    ap.add_argument('--scan', action='store_true', help='扫描多个 power_scale 看温升倍率')
    ap.add_argument('--allow_test', action='store_true',
                    help='[慎用] 用测试集样本(仅限系数冻结后的最终确认)')
    args = ap.parse_args()

    nc = args.nc
    ps_default = args.power_scale if args.power_scale is not None else DHV_POWER_SCALE

    if args.mode == 'baseline':
        print('[FD] baseline 模式: 原版物理不动, 无功率缩放, 仅自检 FD 链路')
        p, u_true = load_sample(args.sample, nc)
        u_pred = solve_fd('baseline', p, None, nc)
        report(u_pred, u_true, f'baseline')
        return

    from k_map import build_k_field
    import jax.numpy as jnp
    nz = int(0.55 * nc + 0.45)
    xc = jnp.linspace(0, 1, nc).reshape(-1, 1)
    yc = jnp.linspace(0, 1, nc).reshape(-1, 1)
    zc = jnp.linspace(0, 0.55, nz).reshape(-1, 1)
    k_field = np.asarray(build_k_field(xc, yc, zc))[0, ..., 0]

    p, u_true = load_sample(args.sample, nc)
    print(f'[FD] 3d5 模式 | 训练集样本 {args.sample} | 功率峰值 {p.max():.4f} | '
          f'真值 T [{T_AMB+T_SCALE*u_true.min():.1f}, {T_AMB+T_SCALE*u_true.max():.1f}]°C')

    if args.scan:
        print(f'[FD] 扫描 power_scale (当前默认 {DHV_POWER_SCALE}, 找倍率≈1 的值):')
        for ps in [0.15, 0.20, 0.274, 0.28, 0.35, 0.50, 1.0]:
            u_pred = solve_fd('3d5', p, k_field, nc, power_scale=ps)
            ratio = report(u_pred, u_true, f'ps={ps:.3f}')
            mark = ' ←≈1 自洽' if abs(ratio - 1.0) < 0.1 else ''
            print(f'      {mark}')
    else:
        u_pred = solve_fd('3d5', p, k_field, nc, power_scale=ps_default)
        ratio = report(u_pred, u_true, f'ps={ps_default:.4f}')
        print('\n===== 判断 =====')
        if abs(ratio - 1.0) < 0.15:
            print('温升倍率 ≈ 1: 物理设定(含功率幅值)与真实数据自洽, 可以训练')
        elif ratio > 1.15:
            print(f'温升倍率 {ratio:.2f} > 1: 功率项仍偏强, 建议 DHV_POWER_SCALE '
                  f'≈ {ps_default/ratio:.3f} (由本次训练集对比得出, 需如实记录来源)')
        else:
            print(f'温升倍率 {ratio:.2f} < 1: 功率项偏弱, 建议 DHV_POWER_SCALE '
                  f'≈ {ps_default/ratio:.3f} (由本次训练集对比得出, 需如实记录来源)')


if __name__ == '__main__':
    main()
