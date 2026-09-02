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

# ---- 7. 训练三组 × 多 seed（固定协议）----
EPOCHS="${EPOCHS:-3000}"
LOG_EVERY="${LOG_EVERY:-200}"
BATCH="${BATCH:-8}"
NC="${NC:-101}"
# 固定协议多种子：空格分隔的 seed 列表，每组模型各跑一遍，最后报 mean±std。
#   固定协议 = 各组同轮数/同batch/同验证集划分规则(同seed同划分)/同评估频率，
#   唯一自由变量是物理假设；种子重复实验量化"彩票效应"。
SEEDS="${SEEDS:-42 2026 7}"
DHV_EVAL_EVERY="${DHV_EVAL_EVERY:-500}"

echo "[deploy] 固定协议: epochs=$EPOCHS batch=$BATCH seeds=($SEEDS) eval_every=$DHV_EVAL_EVERY"

# 清理本次要用的结果目录（含旧的无 seed 目录与旧 tag 目录），防止污染。
rm -rf "$CODE_DIR/results/results_volume/DeepOHeat_v1"/nf${BATCH}_nc${NC}_*_mode_*
rm -rf "$CODE_DIR/results/results_volume/DeepOHeat_v1"/tag_seed*_baseline_mode \
       "$CODE_DIR/results/results_volume/DeepOHeat_v1"/tag_seed*_mode_3d5_full \
       "$CODE_DIR/results/results_volume/DeepOHeat_v1"/tag_seed*_mode_3d5_no_interface
echo "[deploy] 已清理旧结果目录"

run_one() { # mode tag seed extra_env
    local mode="$1" tag="$2" seed="$3" envs="${4:-}"
    echo ""
    echo "======================================================"
    echo "[deploy] 训练: $tag (mode=$mode, epochs=$EPOCHS, seed=$seed)"
    echo "======================================================"
    RESULT="$CODE_DIR/results/results_volume/DeepOHeat_v1/tag_seed${seed}_${tag}"
    mkdir -p "$RESULT"
    env $envs DHV_EVAL_EVERY="$DHV_EVAL_EVERY" python3 heat_volumetric.py --mode "$mode" --model_name DeepOHeat_v1 \
         --batch "$BATCH" --epochs "$EPOCHS" --log_epoch "$LOG_EVERY" --seed "$seed" \
         2>&1 | tee "$RESULT/train_console.log" | tail -60
    # 归档: 结果目录现在带 _valsel_seed<seed> 后缀, 精确定位本组目录
    real_dir=$(ls -d "$CODE_DIR/results/results_volume/DeepOHeat_v1"/nf${BATCH}_nc${NC}_*_mode_${mode}_valsel_seed${seed} 2>/dev/null | head -1)
    if [ -n "$real_dir" ] && [ -d "$real_dir" ]; then
        cp -u "$real_dir"/* "$RESULT"/ 2>/dev/null || true
        echo "[deploy] 结果归档: $(basename "$real_dir") -> $RESULT"
    else
        echo "[deploy] ⚠️ 未找到预期结果目录 nf*_mode_${mode}_valsel_seed${seed}"
    fi
    echo "[deploy] $tag(seed=$seed) 完成: $RESULT"
}

for seed in $SEEDS; do
    run_one baseline baseline_mode         "$seed" ""
    # A2: 3d5 两组默认 DHV_FVM_IFACE=1（FVM 调和平均处理界面，替代显式界面项）。
    #   no_interface 组仍保留 DHV_NO_INTERFACE=1 做"无界面"消融（但 PDE 已是 FVM）。
    run_one 3d5      mode_3d5_full         "$seed" "DHV_FVM_IFACE=1"
    run_one 3d5      mode_3d5_no_interface "$seed" "DHV_FVM_IFACE=1 DHV_NO_INTERFACE=1"
done

# ---- 8. 汇总对比（多 seed: mean±std）----
echo ""
echo "======================================================"
echo "[deploy] 固定协议对比摘要 (seeds: $SEEDS)"
echo "======================================================"
python3 - "$CODE_DIR" "$SEEDS" <<'PYEOF'
import os, sys, re
import statistics
code_dir, seeds = sys.argv[1], sys.argv[2].split()
base = os.path.join(code_dir, "results", "results_volume", "DeepOHeat_v1")
def metrics_for(tag, s):
    ev_f = os.path.join(base, f"tag_seed{s}_{tag}", "log (eval metrics).csv")
    if not os.path.exists(ev_f):
        return None
    kv = {}
    for l in open(ev_f):
        if ':' in l:
            k, v = l.strip().split(':', 1)
            kv[k.strip()] = v.strip()
    return kv
def best_val(tag, s):
    vf = os.path.join(base, f"tag_seed{s}_{tag}", "log (val mape).csv")
    if not os.path.exists(vf):
        return None
    rows = [l.strip() for l in open(vf) if l.strip()]
    if not rows:
        return None
    vals = [float(r.split(',')[1]) for r in rows]
    return min(vals)
groups = [("baseline_mode", "baseline(原版)"),
          ("mode_3d5_full", "3d5(分区k+界面)"),
          ("mode_3d5_no_interface", "3d5-无界面(消融)")]
table = {}
for tag, label in groups:
    rel2s, mapes, vmapes = [], [], []
    for s in seeds:
        kv = metrics_for(tag, s)
        if kv:
            rel2s.append(float(kv['rel_l2_mean']))
            mapes.append(float(kv['mape_mean']))
        bv = best_val(tag, s)
        if bv is not None:
            vmapes.append(bv)
    table[label] = (rel2s, mapes, vmapes)
def fmt(xs, scale=1.0, pct=False):
    if not xs:
        return "N/A"
    m = statistics.mean(xs)*scale
    sd = statistics.stdev(xs)*scale if len(xs) > 1 else 0.0
    return f"{m:.4f}±{sd:.4f}" + ("%" if pct else "")
for label, (rel2s, mapes, vmapes) in table.items():
    tag_of = {l: t for t, l in groups}[label]
    print(f"\n--- {label} (n={len(mapes)} seeds) ---")
    for s in seeds:
        kv = metrics_for(tag_of, s)
        if kv:
            print(f"  seed {s}: rel_l2={float(kv['rel_l2_mean']):.4f}  mape={float(kv['mape_mean'])*100:.2f}%")
    print(f"  测试 rel_l2  mean±std: {fmt(rel2s)}")
    print(f"  测试 MAPE    mean±std: {fmt(mapes, 100, True)}")
    print(f"  验证最优MAPE mean±std: {fmt(vmapes, 100, True)}  (早停选优依据)")
PYEOF

echo ""
echo "[deploy] 全部完成。详细结果: $CODE_DIR/results/results_volume/DeepOHeat_v1/tag_seed*/"