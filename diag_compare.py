# -*- coding: utf-8 -*-
"""diag_compare.py —— 第一层诊断: baseline vs 3d5 误差分布(交接文档十七节方案)
难度分桶(3极端 vs 7常规) + 摄氏度口径 + 分层(Die/TIM/Interposer/Substrate) + 热点误差
"""
import os, glob, sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
os.chdir('/root/autodl-tmp/DeepOHeat-v1')

u_test = np.load('data/u_test_volume.npy')           # [10,101,101,56]
N, nc, _, nz = u_test.shape
T_true = 20.0 + 25.0 * u_test                        # 摄氏度

# 测试集样本顺序 = report.16,19,22,25,26,27,59,80,86,93 (固定测试集, 升序)
TEST_REPORTS = [16,19,22,25,26,27,59,80,86,93]
EXTREME = {80, 86, 93}                               # 3 极端热点
ext_idx = [i for i, r in enumerate(TEST_REPORTS) if r in EXTREME]
reg_idx = [i for i, r in enumerate(TEST_REPORTS) if r not in EXTREME]

# 分层(z 训练坐标): Substrate 0:10 / Interposer 10:35 / TIM 35:40 / Die 40:56
LAYERS = [('Substrate', 0, 10), ('Interposer', 10, 35), ('TIM', 35, 40), ('Die', 40, 56)]

def kelvin_mape(up, ut):
    a, b = 25*up + 293.15, 25*ut + 293.15            # eval.py 同口径
    return np.mean(np.abs((a-b)/b)) * 100

def celsius_mape(up, ut):
    a, b = 20+25*up, 20+25*ut
    return np.mean(np.abs(a-b)/b) * 100

def mae_c(up, ut):
    return np.mean(np.abs((up-ut)*25))

TAGS = []
for pat, label in [('nf16_nc101*mode_baseline_valsel_seed42', 'baseline'),
                   ('nf16_nc101*schedcosine_lrmin0.02_seed42', '3d5-B(fvm)'),
                   ('nf16_nc101*efcontinuous_seed42', '3d5-A(cont)'),
                   ('nf16_nc101*efcontinuous*muon2*', '3d5-C(muon2)')]:
    dirs = glob.glob(f'results/results_volume/DeepOHeat_v1/{pat}')
    if dirs and os.path.exists(os.path.join(dirs[0], 'u_pred_heat3d.npy')):
        TAGS.append((label, dirs[0]))

print('=' * 100)
print(f'{"指标":34s}' + ''.join(f'{lab:>16s}' for lab, _ in TAGS))
print('=' * 100)

preds = {}
for lab, d in TAGS:
    up = np.load(os.path.join(d, 'u_pred_heat3d.npy'))
    if up.ndim == 5:
        up = up[..., 0]
    preds[lab] = up

def row(name, fn):
    vals = []
    for lab, _ in TAGS:
        vals.append(fn(preds[lab]))
    best = min(range(len(vals)), key=lambda i: vals[i])
    s = f'{name:34s}'
    for i, v in enumerate(vals):
        mark = ' *' if i == best else '  '
        s += f'{v:14.4f}%{mark}' if '%' in name or 'MAPE' in name else f'{v:14.4f}{mark}'
    print(s)

row('开尔文MAPE%(eval口径)', lambda up: kelvin_mape(up, u_test))
row('摄氏MAPE%', lambda up: celsius_mape(up, u_test))
row('MAE(°C)', lambda up: mae_c(up, u_test))
row('rel_l2%(均值)', lambda up: np.mean(np.linalg.norm(up.reshape(10,-1)-u_test.reshape(10,-1),axis=1)/np.linalg.norm(u_test.reshape(10,-1),axis=1))*100)
print('-' * 100)

# 难度分桶
def bucket_mape(fn, idx):
    return lambda up: np.mean([fn(up[i:i+1], u_test[i:i+1]) for i in idx])
row('极端桶(80/86/93) 开尔文MAPE%', bucket_mape(kelvin_mape, ext_idx))
row('常规桶(7组) 开尔文MAPE%', bucket_mape(kelvin_mape, reg_idx))
row('极端桶 摄氏MAPE%', bucket_mape(celsius_mape, ext_idx))
row('常规桶 摄氏MAPE%', bucket_mape(celsius_mape, reg_idx))
print('-' * 100)

# 分层 MAE
for lname, z0, z1 in LAYERS:
    row(f'{lname} MAE(°C)', lambda up, z0=z0, z1=z1: np.mean(np.abs((up-u_test)[:,:,:,z0:z1])*25))
print('-' * 100)

# 热点: Die层 T>40°C 区域
def hot_mae(up):
    die_t = T_true[:, :, :, 40:]
    e = np.abs((up - u_test)[:, :, :, 40:] * 25)
    m = die_t > 40
    return np.mean([e[i][m[i]].mean() for i in range(N)])
def tmax_err(up):
    return np.mean([abs(up[i].max()-u_test[i].max())*25 for i in range(N)])
row('热点区(T>40°C) MAE(°C)', hot_mae)
row('Tmax误差(°C,逐样本均值)', tmax_err)

print('=' * 100)
print('* = 该指标最优. 极端桶 = report.80/86/93 (3组极限热点)')
