# -*- coding: utf-8 -*-
"""diag_hot.py —— 热点专项: 逐样本 top-1% 最热节点误差 + 极端样本明细"""
import os, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
os.chdir('/root/autodl-tmp/DeepOHeat-v1')

u_test = np.load('data/u_test_volume.npy')
N = u_test.shape[0]
T_true = 20.0 + 25.0 * u_test
TEST_REPORTS = [16,19,22,25,26,27,59,80,86,93]
EXTREME = {80, 86, 93}

TAGS = []
for pat, label in [('nf16_nc101*mode_baseline_valsel_seed42', 'baseline'),
                   ('nf16_nc101*schedcosine_lrmin0.02_seed42', '3d5-B(fvm)'),
                   ('nf16_nc101*efcontinuous_seed42', '3d5-A(cont)'),
                   ('nf16_nc101*efcontinuous*muon2*', '3d5-C(muon2)')]:
    dirs = glob.glob(f'results/results_volume/DeepOHeat_v1/{pat}')
    if dirs:
        TAGS.append((label, dirs[0]))

preds = {}
for lab, d in TAGS:
    up = np.load(os.path.join(d, 'u_pred_heat3d.npy'))
    if up.ndim == 5: up = up[..., 0]
    preds[lab] = up

print('=' * 105)
hdr = f'{"样本":12s}{"Tmax真值":>9s}{"极端":>5s}'
for lab, _ in TAGS:
    hdr += f'{lab+"/热点MAE":>18s}'
print(hdr + f'{"最优":>12s}')
print('=' * 105)

hot_all = {lab: [] for lab, _ in TAGS}
tmax_err = {lab: [] for lab, _ in TAGS}
for i in range(N):
    die = T_true[i, :, :, 40:]
    thr = np.percentile(die, 99)                 # 该样本最热的 1% die 节点
    m = die >= thr
    line = f'report.{TEST_REPORTS[i]:<5d}{T_true[i].max():>8.1f}C{"极端" if TEST_REPORTS[i] in EXTREME else "":>4s}'
    errs = {}
    for lab, _ in TAGS:
        e = np.abs((preds[lab][i] - u_test[i])[:, :, 40:] * 25)
        v = e[m].mean()
        errs[lab] = v
        hot_all[lab].append(v)
        tmax_err[lab].append(abs(preds[lab][i].max() - u_test[i].max()) * 25)
        line += f'{v:18.3f}'
    best = min(errs, key=errs.get)
    line += f'{best:>12s}'
    print(line)

print('-' * 105)
print(f'{"全测试集均值":17s}', end='')
for lab, _ in TAGS:
    print(f'{np.mean(hot_all[lab]):18.3f}', end='')
print()
ext = [i for i, r in enumerate(TEST_REPORTS) if r in EXTREME]
reg = [i for i, r in enumerate(TEST_REPORTS) if r not in EXTREME]
print(f'{"极端桶均值(80/86/93)":22s}', end='')
for lab, _ in TAGS:
    print(f'{np.mean([hot_all[lab][i] for i in ext]):18.3f}', end='')
print()
print(f'{"常规桶均值(7组)":22s}', end='')
for lab, _ in TAGS:
    print(f'{np.mean([hot_all[lab][i] for i in reg]):18.3f}', end='')
print()
print(f'{"Tmax误差均值(°C)":22s}', end='')
for lab, _ in TAGS:
    print(f'{np.mean(tmax_err[lab]):18.3f}', end='')
print()
