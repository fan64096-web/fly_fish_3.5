#!/usr/bin/env bash
# 精简版: 每配置一次. 3.5d热点强化(seed42)=1.57%已完成, 这里只跑3.5d主配置和Muon2消融
cd /root/autodl-tmp/DeepOHeat-v1

echo "===== 3.5d主配置: fvm能量(对照, 其余同热点强化配置) ====="
env DHV_POWER_SCALE=0.33 DHV_ENERGY=1 DHV_ENERGY_FORM=fvm \
    DHV_SCHED=cosine DHV_GRAD_CLIP=0 \
    /root/miniconda3/bin/python3 -u heat_volumetric.py --mode 3d5 \
    --batch 16 --epochs 20000 --log_epoch 5000 --seed 42

echo "===== Muon2消融: 3.5d热点强化 + Muon2 ====="
env DHV_POWER_SCALE=0.33 DHV_ENERGY=1 DHV_ENERGY_FORM=continuous \
    DHV_SCHED=cosine DHV_GRAD_CLIP=0 DHV_MUON2=1 \
    /root/miniconda3/bin/python3 -u heat_volumetric.py --mode 3d5 \
    --batch 16 --epochs 20000 --log_epoch 5000 --seed 42

echo "ARM_BC_DONE"
