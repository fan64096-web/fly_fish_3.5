# -*- coding: utf-8 -*-
"""
eval_layers.py —— 逐 z 层 / 横向区域 误差定位分析
==================================================
加载训练好的模型,在固定测试集(10 组)上计算预测误差,并按
  a) z 方向材料层（Substrate/Interposer/TIM/Die）
  b) 横向区域（Die vs 间隙,由 mask 通道）
拆开,定位 3d5 预测到底错在哪几层/哪几个区域。

用法
----
python3 eval_layers.py <模型tag目录> --mode baseline|3d5

  例:
  python3 eval_layers.py results/results_volume/DeepOHeat_v1/tag_baseline_mode --mode baseline
  python3 eval_layers.py results/results_volume/DeepOHeat_v1/tag_mode_3d5_full --mode 3d5

依赖:与训练完全相同的 models.py / 数据 / 环境。
"""
import os
import sys
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag_dir', help='模型 tag 目录(含 .eqx)')
    ap.add_argument('--mode', default='3d5', choices=['baseline', '3d5'])
    ap.add_argument('--nc', type=int, default=101)
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--branch_depth', type=int, default=8)
    ap.add_argument('--branch_hidden', type=int, default=256)
    ap.add_argument('--trunk_depth', type=int, default=3)
    ap.add_argument('--trunk_hidden', type=int, default=64)
    ap.add_argument('--r', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from models import DeepOHeat_v1

    nc = args.nc
    nz = int(0.55 * nc + 0.45)
    n_feat = nc * nc

    # ---- 数据加载(与 heat_volumetric 一致) ----
    if args.mode == '3d5':
        fs_test = np.load(f'{args.data_dir}/fs_test_3d5_volume.npy', mmap_mode='r')
        channels = 3
    else:
        fs_test = np.load(f'{args.data_dir}/fs_test_volume.npy', mmap_mode='r').reshape(-1, n_feat)
        channels = 1
    u_test = np.load(f'{args.data_dir}/u_test_volume.npy')   # [N, nc, nc, nz]

    # ---- 模型重建(必须与训练超参完全一致) ----
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)
    model = DeepOHeat_v1(dim=3, branch_dim=n_feat, field_dim=1,
                         branch_depth=args.branch_depth, branch_hidden=args.branch_hidden,
                         trunk_depth=args.trunk_depth, trunk_hidden=args.trunk_hidden,
                         rank=args.r, channels=channels, key=subkey)
    eqx_file = None
    for fn in os.listdir(args.tag_dir):
        if fn.endswith('.eqx'):
            eqx_file = os.path.join(args.tag_dir, fn)
            break
    if eqx_file is None:
        print(f'[错误] {args.tag_dir} 下未找到 .eqx 文件')
        sys.exit(1)
    model = eqx.tree_deserialise_leaves(eqx_file, model)
    print(f'[加载] {eqx_file}')

    # ---- 测试网格(与 eval_heat3d 一致) ----
    x = jnp.linspace(0, 1, nc).reshape(-1, 1)
    y = jnp.linspace(0, 1, nc).reshape(-1, 1)
    z = jnp.linspace(0, 0.55, nz).reshape(-1, 1)

    N = fs_test.shape[0]
    print(f'[测试] {N} 个样本, 分辨率 {nc}x{nc}x{nz}, channels={channels}')

    u_pred_all = []
    for i in range(N):
        f = jnp.asarray(np.asarray(fs_test[i]))
        u_pred = model(((x, y, z), f))        # [nc, nc, nz, 1]
        u_pred_all.append(np.asarray(u_pred[..., 0]))
    u_pred = np.stack(u_pred_all, axis=0)     # [N, nc, nc, nz]

    u_true = np.asarray(u_test)
    # 还原真实温度(°C, 与 make_icepak_dataset 归一化兼容)
    T_true = 20.0 + 25.0 * u_true
    T_pred = 20.0 + 25.0 * u_pred
    err_abs = np.abs(T_pred - T_true)         # [N, nc, nc, nz]  °C

    # ---- ① 整体指标(与 eval.py 同口径,取均值) ----
    rl2 = np.linalg.norm(u_pred - u_true, axis=(1,2,3)) / np.linalg.norm(u_true, axis=(1,2,3))
    mape_all = np.mean(np.abs(T_pred - T_true) / T_true, axis=(1,2,3))
    print(f'\n========== 整体 ==========')
    print(f'rel_l2   mean: {rl2.mean():.4f} (std {rl2.std():.4f})')
    print(f'MAPE     mean: {mape_all.mean()*100:.2f}%  (在 T=20+25u 口径)')

    # ---- ② 逐 z 层误差(材料层) ----
    zvals = np.linspace(0, 0.55, nz)
    def zlayer(z):
        if z < 0.10: return 'Substrate'
        if z < 0.35: return 'Interposer'
        if z < 0.40: return 'TIM'
        return 'Die'
    zlab = [zlayer(z) for z in zvals]

    print(f'\n========== 逐 z 层(材料层)平均绝对误差 ==========')
    print(f'{"层":16s} {"z范围":12s} {"MAE(°C)":>10s} {"rel_err":>10s}  {"MAPE贡献":>12s}')
    for name, z0, z1 in [('Substrate', 0, 10), ('Interposer', 10, 35), ('TIM', 35, 40), ('Die', 40, 56)]:
        e = err_abs[:, :, :, z0:z1]
        t = T_true[:, :, :, z0:z1]
        mae = e.mean()
        rel = np.mean(e / t)
        mape_contrib = np.mean(e / t)
        print(f'{name:16s} z[{z0:2d}:{z1:2d}] z={zvals[z0]:.2f}~{zvals[z1-1]:.2f}  '
              f'{mae:9.3f}   {rel*100:9.2f}%  {mape_contrib*100:11.2f}%')

    # ---- ③ 横向区域(mask)误差(仅 Die 层有意义,完整看也行) ----
    if args.mode == '3d5':
        # mask 通道 = fs_test 的第 2 段
        mask = np.asarray(fs_test[0, n_feat:2*n_feat]).reshape(nc, nc)
        # 在 Die 层(全部 z>=0.40)看 Compute vs HBM vs 间隙
        die_e = err_abs[:, :, :, 40:]
        labels = [(0, 'Compute'), (1, 'HBM'), (2, '间隙/Interposer')]
        print(f'\n========== Die 层横向区域误差 ==========')
        for mid, lab in labels:
            sel = (mask == mid)[None, :, :, None]
            e = np.where(sel, die_e, 0)
            cnt = sel.sum()
            if cnt > 0:
                print(f'{lab:16s}  MAE(°C): {e.sum()/cnt:9.3f}   节点数: {cnt}')

    # ---- ④ 每样本热点区域 ----
    print(f'\n========== 每样本 Die 层(T>40°C 区域)误差 ==========')
    die_t = T_true[:, :, :, 42:56]
    for i in range(N):
        hot = die_t[i] > 40
        e = np.abs(T_pred[i, :, :, 42:56] - T_true[i, :, :, 42:56])
        if hot.sum() > 0:
            print(f'  样本{i}: 热点区MAE={e[hot].mean():.3f}°C  全部z={err_abs[i].mean():.3f}°C  Tmax={T_true[i].max():.1f}°C')

if __name__ == '__main__':
    main()