#!/usr/bin/env bash
# armA: continuous能量 + adam + cosine+floor + 界面开 —— 对齐同事 v3/v6 配置锚点
cd /root/autodl-tmp/DeepOHeat-v1
for s in 42 43 44; do
  env DHV_POWER_SCALE=0.33 DHV_ENERGY=1 DHV_ENERGY_FORM=continuous \
      DHV_SCHED=cosine DHV_GRAD_CLIP=0 \
      /root/miniconda3/bin/python3 heat_volumetric.py --mode 3d5 \
      --batch 16 --epochs 20000 --log_epoch 1000 --seed $s
done
echo "ARM_A_ALL_DONE"
