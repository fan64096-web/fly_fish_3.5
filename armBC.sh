#!/usr/bin/env bash
# 精简版: 每臂一次. 臂A(seed42)=1.57%已完成, 这里只跑臂B和臂C
cd /root/autodl-tmp/DeepOHeat-v1

echo "===== 臂B: fvm能量(对照, 其余同臂A) ====="
env DHV_POWER_SCALE=0.33 DHV_ENERGY=1 DHV_ENERGY_FORM=fvm \
    DHV_SCHED=cosine DHV_GRAD_CLIP=0 \
    /root/miniconda3/bin/python3 -u heat_volumetric.py --mode 3d5 \
    --batch 16 --epochs 20000 --log_epoch 5000 --seed 42

echo "===== 臂C: 臂A + Muon2 ====="
env DHV_POWER_SCALE=0.33 DHV_ENERGY=1 DHV_ENERGY_FORM=continuous \
    DHV_SCHED=cosine DHV_GRAD_CLIP=0 DHV_MUON2=1 \
    /root/miniconda3/bin/python3 -u heat_volumetric.py --mode 3d5 \
    --batch 16 --epochs 20000 --log_epoch 5000 --seed 42

echo "ARM_BC_DONE"
