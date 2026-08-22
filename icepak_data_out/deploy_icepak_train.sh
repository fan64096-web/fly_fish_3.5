#!/usr/bin/env bash
# =====================================================================
# deploy_icepak_train.sh —— Icepak 真实数据 · 服务器一键训练与对比
#
# 训练并对比 3 个模型（同一份 10 组 Icepak 真实数据）：
#   1) baseline        论文原版算法（纯功率输入）
#   2) 3d5             改造算法（三通道 + 分区k PDE + 界面热流连续）
#   3) 3d5-ni          消融：3d5 但关闭界面热流连续
#
# 数据流
#   reports/ 里的 10 个 report.* + 功率表
#     -> make_icepak_dataset.py 转换
#     -> DeepOHeat-v1/data/  (8 训练 + 2 测试, 101x101x56)
#     -> 依次训练三组
#
# 用法
#   把本文件所在目录整个上传到服务器（放在 DeepOHeat-v1 旁边或里面任一位置），
#   然后：  bash deploy_icepak_train.sh
#   可用环境变量覆盖： EPOCHS（默认3000） LOG_EVERY（默认200）
#
# 结果目录： DeepOHeat-v1/results/results_volume/DeepOHeat_v1/tag_*/
# =====================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="$SCRIPT_DIR/reports"
SAMPLES_TXT="$SCRIPT_DIR/icepak_samples.txt"

# ---- 1. 向上定位 DeepOHeat-v1 根目录（含 models.py 的目录）----
CODE_DIR=""
d="$SCRIPT_DIR"
for _ in 1 2 3 4 5 6; do
    if [ -f "$d/models.py" ]; then CODE_DIR="$d"; break; fi
    d="$(dirname "$d")"
done
if [ -z "$CODE_DIR" ]; then
    echo "[deploy] 未找到含 models.py 的 DeepOHeat-v1 目录"
    echo "[deploy] 请把本部署文件夹放在 DeepOHeat-v1 目录内或其上级某处"
    exit 1
fi
DATA_DIR="$CODE_DIR/data"
echo "[deploy] 代码目录: $CODE_DIR"
echo "[deploy] 数据目录: $DATA_DIR"

# ---- 2. 检查报告是否齐全（report.01 ~ report.100）----
echo "[deploy] 检查报告 (01~100)..."
missing=0
cnt=0
for i in $(seq -w 1 99); do
    [ -f "$REPORTS_DIR/report.$i" ] || { echo "  缺 report.$i"; missing=1; }
    cnt=$((cnt+1))
done
[ -f "$REPORTS_DIR/report.100" ] || { echo "  缺 report.100"; missing=1; }
cnt=$((cnt+1))
if [ "$missing" = "1" ]; then
    echo "[deploy] 100 个报告不齐全（应有 $cnt 个），请放到 $REPORTS_DIR"
    exit 1
fi
echo "[deploy] $cnt 个报告齐全。"

# ---- 3. 复制转换脚本到代码目录（保证能 import gen_mask 等）----
cp "$SCRIPT_DIR/make_icepak_dataset.py" "$CODE_DIR/" || { echo "复制脚本失败"; exit 1; }

# ---- 4. 准备样本清单（服务器路径版）----
if [ -f "$SAMPLES_TXT" ]; then
    sed "s|C:/Users/25164/Desktop/3d5_chip/数据集/|$REPORTS_DIR/|g" "$SAMPLES_TXT" \
        > "$SCRIPT_DIR/icepak_samples_server.txt"
    SAMPLES_SRV="$SCRIPT_DIR/icepak_samples_server.txt"
    echo "[deploy] 样本清单: $SAMPLES_SRV"
else
    # 自动生成：从 reports/ 目录自动构建
    SAMPLES_SRV="$SCRIPT_DIR/icepak_samples_server.txt"
    : > "$SAMPLES_SRV"
    cat > "$SAMPLES_SRV" <<'TABLE'
# 自动生成的样本清单（需要与 reports/ 目录中的报告文件对应）
TABLE
    # 10 组功率（与用户确认过的方案一致）
    pw=( "0.20 0.20 0.08 0.08 0.08 0.08"
         "0.28 0.28 0.12 0.12 0.12 0.12"
         "0.24 0.32 0.10 0.14 0.10 0.14"
         "0.32 0.24 0.14 0.10 0.14 0.10"
         "0.35 0.35 0.15 0.15 0.15 0.15"
         "0.18 0.30 0.06 0.12 0.18 0.08"
         "0.30 0.18 0.12 0.06 0.08 0.18"
         "0.40 0.25 0.08 0.16 0.12 0.06"
         "0.22 0.38 0.16 0.08 0.06 0.14"
         "0.38 0.20 0.06 0.14 0.18 0.10" )
    for i in 0 1 2 3 4 5 6 7 8 9; do
        n=$((i+1))
        printf "%s/report.%02d  ${pw[$i]}\n" "$REPORTS_DIR" "$n" >> "$SAMPLES_SRV"
    done
    echo "[deploy] 样本清单已自动生成: $SAMPLES_SRV"
fi

# ---- 5. 备份原始数据 ----
mkdir -p "$DATA_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$CODE_DIR/data_backup_orig_$STAMP"
mkdir -p "$BACKUP_DIR"
for f in fs_train_3d5_volume.npy fs_test_3d5_volume.npy \
         fs_train_volume.npy      fs_test_volume.npy \
         u_train_3d5.npy          u_test_3d5.npy \
         u_train_volume.npy       u_test_volume.npy; do
    if [ -f "$DATA_DIR/$f" ]; then cp "$DATA_DIR/$f" "$BACKUP_DIR/" && echo "  备份 $f"; fi
done
echo "[deploy] 原始数据已备份: $BACKUP_DIR"

# ---- 6. 生成 Icepak 真实数据集（在代码目录运行，确保 import 正常）----
cd "$CODE_DIR"
echo "[deploy] 生成数据集 (100 样本 -> 90 训练 + 10 测试, 101x101x56)..."
python3 "$CODE_DIR/make_icepak_dataset.py" "$SAMPLES_SRV" --out_dir "$DATA_DIR" || { echo "数据生成失败"; exit 1; }

# ---- 7. 训练三组 ----
EPOCHS="${EPOCHS:-3000}"
LOG_EVERY="${LOG_EVERY:-200}"
BATCH="${BATCH:-8}"
NC="${NC:-101}"

# 清理本次要用的结果目录，防止旧结果（尤其 3d5 全/无界面 曾共用目录）污染本次归档。
rm -rf "$CODE_DIR/results/results_volume/DeepOHeat_v1"/nf${BATCH}_nc${NC}_*_mode_*
rm -rf "$CODE_DIR/results/results_volume/DeepOHeat_v1"/tag_baseline_mode \
       "$CODE_DIR/results/results_volume/DeepOHeat_v1"/tag_mode_3d5_full \
       "$CODE_DIR/results/results_volume/DeepOHeat_v1"/tag_mode_3d5_no_interface
echo "[deploy] 已清理旧结果目录"

run_one() { # mode tag extra_env
    local mode="$1" tag="$2" envs="${3:-}"
    echo ""
    echo "======================================================"
    echo "[deploy] 训练: $tag (mode=$mode, epochs=$EPOCHS)"
    echo "======================================================"
    RESULT="$CODE_DIR/results/results_volume/DeepOHeat_v1/tag_$tag"
    mkdir -p "$RESULT"
    env $envs python3 heat_volumetric.py --mode "$mode" --model_name DeepOHeat_v1 \
         --batch "$BATCH" --epochs "$EPOCHS" --log_epoch "$LOG_EVERY" \
         2>&1 | tee "$RESULT/train_console.log" | tail -60
    # 归档真实结果：heat_volumetric 把 csv/npy/eqx 写到 nf<batch>_nc<NC>_..._mode_<mode> 目录。
    # 3d5 完整版与 3d5 无界面版 mode 相同、会写进同一目录互相覆盖，
    # 因此每组训练完立即把真实结果复制进自己的 tag_* 目录。
    real_dir="$CODE_DIR/results/results_volume/DeepOHeat_v1/nf${BATCH}_nc${NC}_*_mode_${mode}"
    for d in $real_dir; do
        if [ -d "$d" ]; then
            cp -u "$d"/* "$RESULT"/ 2>/dev/null || true
            echo "[deploy] 结果归档: $(basename "$d") -> $RESULT"
            break
        fi
    done
    echo "[deploy] $tag 完成: $RESULT"
}

run_one baseline baseline_mode        ""
run_one 3d5      mode_3d5_full        ""
run_one 3d5      mode_3d5_no_interface "DHV_NO_INTERFACE=1"

# ---- 8. 汇总对比 ----
echo ""
echo "======================================================"
echo "[deploy] 三组对比摘要：loss 末段 / rel_l2 / MAPE"
echo "======================================================"
python3 - "$CODE_DIR" <<'PYEOF'
import os, sys
code_dir = sys.argv[1]
base = os.path.join(code_dir, "results", "results_volume", "DeepOHeat_v1")
def summary(tag, label):
    d = os.path.join(base, f"tag_{tag}")
    loss_f = os.path.join(d, "log (loss).csv")
    ev_f = os.path.join(d, "log (eval metrics).csv")
    print(f"\n--- {label} ---")
    if os.path.exists(loss_f):
        rows = [l.strip() for l in open(loss_f) if l.strip()]
        print("  loss 末3:", rows[-3:] if rows else "空")
    if os.path.exists(ev_f):
        kv = {}
        for l in open(ev_f):
            if ':' in l:
                k, v = l.strip().split(':', 1)
                kv[k.strip()] = v.strip()
        print("  rel_l2_mean:", kv.get('rel_l2_mean', 'N/A'),
              "| mape_mean:", kv.get('mape_mean', 'N/A'),
              "| pape_mean:", kv.get('pape_mean', 'N/A'))
summary("baseline_mode", "baseline(原版)")
summary("mode_3d5_full", "3d5(分区k+界面)")
summary("mode_3d5_no_interface", "3d5-无界面(消融)")
PYEOF

echo ""
echo "[deploy] 全部完成。详细结果: $CODE_DIR/results/results_volume/DeepOHeat_v1/tag_*/"