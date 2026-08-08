#!/usr/bin/env bash
# =============================================================
# Volume GPU 小规模验证一键脚本
# DeepOHeat-V1 3.5D 改造 · 阶段 3 任务 2
# =============================================================
# 用法：
#   把本文件放到 DeepOHeat-v1/ 目录下，然后：
#   bash run_volume_validation.sh
#
# 它会依次：
#   1. 检查磁盘空间（需 ≥25GB）
#   2. 生成 volume 三通道训练数据（gen_mask.py --volume，约 24GB）
#   3. 小规模跑 baseline（200 epochs）
#   4. 小规模跑 3d5（200 epochs）
#   5. 打印结果文件位置
#
# 说明：
#   - 默认 --epochs 200 只是验证代码链路，不是正式训练
#   - 若 OOM，脚本会自动用 --batch 16 重试一次
#   - 正式训练请去掉 --epochs 参数
# =============================================================
set -e
cd "$(dirname "$0")"

echo "======================================"
echo "[1/4] 检查磁盘空间..."
echo "======================================"
AVAIL=$(df -P . | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL / 1024 / 1024))
echo "当前可用磁盘: ${AVAIL_GB}GB (需要 ≥25GB)"
if [ "$AVAIL_GB" -lt 25 ]; then
    echo "❌ 磁盘空间不足！请清理后重试。"
    exit 1
fi
echo "✅ 磁盘空间充足"

echo ""
echo "======================================"
echo "[2/4] 生成 volume 三通道训练数据..."
echo "======================================"
if [ -f data/fs_train_3d5_volume.npy ]; then
    echo "已存在 data/fs_train_3d5_volume.npy，跳过生成"
else
    python gen_mask.py --volume
    echo "✅ 数据生成完成"
fi

echo ""
echo "======================================"
echo "[3/4] 小规模跑 baseline (200 epochs)..."
echo "======================================"
python heat_volumetric.py --mode baseline --epochs 200 --log_epoch 20 \
    || { echo "⚠️ baseline 失败，尝试 batch 16"; \
         python heat_volumetric.py --mode baseline --epochs 200 --log_epoch 20 --batch 16; }

echo ""
echo "======================================"
echo "[4/4] 小规模跑 3d5 (200 epochs)..."
echo "======================================"
python heat_volumetric.py --mode 3d5 --epochs 200 --log_epoch 20 \
    || { echo "⚠️ 3d5 失败，尝试 batch 16"; \
         python heat_volumetric.py --mode 3d5 --epochs 200 --log_epoch 20 --batch 16; }

echo ""
echo "======================================"
echo "✅ 全部完成！结果文件位置："
echo "======================================"
echo "baseline:"
ls -d results/results_volume/DeepOHeat_v1/*_mode_baseline/ 2>/dev/null || echo "  (未找到)"
echo "3d5:"
ls -d results/results_volume/DeepOHeat_v1/*_mode_3d5/ 2>/dev/null || echo "  (未找到)"
echo ""
echo "每个目录下的指标文件："
echo "  log (loss).csv          - 训练损失曲线"
echo "  log (eval metrics).csv  - MAPE / rel_l2 / rmse"
echo "  total runtime (sec).csv - 训练时间"
echo "  memory usage (mb).csv   - GPU 显存"
echo ""
echo "提示：3d5 的 MAPE 偏高是正常的（k 设定与数据物理不一致），"
echo "      重点看 loss 能否下降、能否正常训练。"
