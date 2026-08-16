"""
gen_mask.py
===========
DeepOHeat-V1 改造 · 阶段 1：生成 3.5D 区域掩码（mask）与界面编码（interface）数据

用途
----
为 Branch 网络生成两个新输入通道：
  1. mask       : 区域掩码，告诉模型每个 (x, y) 位置属于哪个区域（chiplet）
  2. interface  : 界面编码，告诉模型每个 (x, y) 位置是否在区域交界处

几何设定（AI 芯片 3.5D 布局，与 Ansys Icepak 模型 3d5_chip 完全一致）
---------------------------------------------------------------------------
横向：中列 2 Compute + 四角 4 HBM（Icepak 实际建模布局）：

        y=1
      ┌───────┬───────┬───────┐
      │ HBM_3 │ Comp2 │ HBM_4 │
      ├───────┼───────┼───────┤  ← y=0.51（Compute 间间隙）
      │ HBM_1 │ Comp1 │ HBM_2 │
      └───────┴───────┴───────┘
   x=0    x=0.29  x=0.3   x=0.7 x=0.71  x=1

  - Compute（中列，2 块）：x∈[0.30,0.70]，y∈[0,0.49]∪[0.51,1.0] → mask 0
  - HBM（四角，4 块）：x∈[0,0.29]∪[0.71,1] 且 y∈[0,0.29]∪[0.71,1] → mask 1
  - 其余（间隙/暴露区）：Interposer → mask 2
  - interface：die 之间的边界线（x≈0.29/0.30/0.70/0.71, y≈0.29/0.30/0.49/0.51/0.70/0.71）记 1

产物（写入 data/ 目录）
-----------------------
  mask_surface.npy        [441]   float64  区域掩码（21×21 摊平）
  interface_surface.npy   [441]   float64  界面编码（21×21 摊平）
  fs_train_3d5_surface.npy [10000, 1323]   训练输入（功率|掩码|界面 三通道拼接）
  fs_test_3d5_surface.npy  [10, 1323]      测试输入（同上）

三通道拼接规则（每样本）
-----------------------
  原始功率向量  p : [441]
  区域掩码      m : [441]  （所有样本相同，广播）
  界面编码      i : [441]  （所有样本相同，广播）
  拼接结果      f : [1323] = [p, m, i]

运行
----
  python gen_mask.py
"""

import os
import numpy as np

# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
N_SURFACE = 21            # surface 功率图分辨率（21×21）
N_VOLUME = 101            # volume 功率图分辨率（101×101）
REGION_IDS = (0, 1, 2, 3)  # 四个区域的 ID

# ---------------------------------------------------------------------
# 掩码 / 界面生成
# ---------------------------------------------------------------------
def make_horizontal_mask(n=N_SURFACE):
    """生成 AI 芯片横向 mask，返回 [n, n] float64 数组。

    mask 通道 = 材料/功能区域分类（0~2），与 Icepak 模型 3d5_chip 一致：
        0 = Compute Die（中列 2 块：x∈[0.30,0.70]，y∈[0,0.49]∪[0.51,1.0]）
        1 = HBM（四角 4 块：x∈[0,0.29]∪[0.71,1] 且 y∈[0,0.29]∪[0.71,1]）
        2 = Interposer（die 之间的间隙区域，材料为中介层）
    （Substrate=3, TIM=4 由纵向层定义，见 k_map.material_id）

    Icepak 实际尺寸（mm）：Compute x:3~7, y:0~4.9 & 5.1~10；
    HBM 四角 0~2.9 / 7.1~10（归一化后 0.29/0.71 边界，间隙 0.29~0.3、0.7~0.71、
    0.49~0.51 归 Interposer）。
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Compute Die：中列 x∈[0.30,0.70]，y 上下两块 [0,0.49] 和 [0.51,1.0]
    is_compute = (xx >= 0.30) & (xx <= 0.70) & ((yy <= 0.49) | (yy >= 0.51))
    # HBM：四角（x 与 y 都在 0.29 以内或 0.71 以外）
    is_hbm = (
        ((xx <= 0.29) & (yy <= 0.29)) |   # 左下 HBM_1
        ((xx >= 0.71) & (yy <= 0.29)) |   # 右下 HBM_2
        ((xx <= 0.29) & (yy >= 0.71)) |   # 左上 HBM_3
        ((xx >= 0.71) & (yy >= 0.71))     # 右上 HBM_4
    )
    # 非 Compute 非 HBM = Interposer（材料 2）
    mask = np.where(is_compute, 0, np.where(is_hbm, 1, 2)).astype(np.float64)
    return mask


def make_horizontal_interface(n=N_SURFACE):
    """生成界面编码：die 边界线记 1，其余记 0（与 Icepak 几何一致）。

    异质界面 = Compute/HBM die 之间的边界：
      x 方向：x≈0.29（HBM↔间隙）、0.30（间隙↔Compute）、0.70、0.71
      y 方向：y≈0.29、0.30、0.49（Compute 下块↔间隙）、0.51、0.70、0.71
    返回 [n, n] float64 数组。
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    interface = np.zeros((n, n), dtype=np.float64)
    # 竖线：x 方向 die 边界
    for xb in (0.29, 0.30, 0.70, 0.71):
        interface[np.isclose(xx, xb)] = 1.0
    # 横线：y 方向 die 边界
    for yb in (0.29, 0.30, 0.49, 0.51, 0.70, 0.71):
        interface[np.isclose(yy, yb)] = 1.0
    return interface


# ---------------------------------------------------------------------
# 主流程：生成掩码/界面，并拼接出三通道训练/测试数据
# ---------------------------------------------------------------------
def build_surface_3d5_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) 生成掩码与界面，并单独保存（便于人工检查）
    mask = make_horizontal_mask(N_SURFACE).reshape(-1)          # [441]
    interface = make_horizontal_interface(N_SURFACE).reshape(-1)  # [441]
    np.save(os.path.join(DATA_DIR, 'mask_surface.npy'), mask)
    np.save(os.path.join(DATA_DIR, 'interface_surface.npy'), interface)
    print(f'[gen_mask] mask_surface.npy      形状 {mask.shape}, 取值 {np.unique(mask)}')
    print(f'[gen_mask] interface_surface.npy 形状 {interface.shape}, 取值 {np.unique(interface)}')

    # 2) 加载原始功率数据（[N, n*n]）
    fs_train = np.load(os.path.join(DATA_DIR, 'fs_train_surface.npy')).reshape(-1, N_SURFACE ** 2)
    fs_test = np.load(os.path.join(DATA_DIR, 'fs_test_surface.npy')).reshape(-1, N_SURFACE ** 2)
    print(f'[gen_mask] 原始功率 fs_train {fs_train.shape}, fs_test {fs_test.shape}')

    # 3) 掩码/界面对所有样本相同 → 广播到与功率相同行数
    m_train = np.broadcast_to(mask, fs_train.shape)        # [N_train, 441]
    i_train = np.broadcast_to(interface, fs_train.shape)
    m_test = np.broadcast_to(mask, fs_test.shape)          # [N_test, 441]
    i_test = np.broadcast_to(interface, fs_test.shape)

    # 4) 三通道拼接：第 1 块 = 功率，第 2 块 = 掩码，第 3 块 = 界面
    fs_train_3d5 = np.concatenate([fs_train, m_train, i_train], axis=-1)   # [N_train, 1323]
    fs_test_3d5 = np.concatenate([fs_test, m_test, i_test], axis=-1)       # [N_test, 1323]

    # 5) 保存拼接后的数据
    np.save(os.path.join(DATA_DIR, 'fs_train_3d5_surface.npy'), fs_train_3d5)
    np.save(os.path.join(DATA_DIR, 'fs_test_3d5_surface.npy'), fs_test_3d5)
    print(f'[gen_mask] fs_train_3d5_surface.npy 形状 {fs_train_3d5.shape}')
    print(f'[gen_mask] fs_test_3d5_surface.npy  形状 {fs_test_3d5.shape}')

    # 6) 自检：随机挑几个位置，打印功率/掩码/界面 三段，确认拼接正确
    row = 0
    print(f'\n[gen_mask] 自检样本#0 前 6 个位置的拼接内容:')
    for k in range(6):
        p = fs_train_3d5[row, k]
        m = fs_train_3d5[row, N_SURFACE ** 2 + k]
        i = fs_train_3d5[row, 2 * N_SURFACE ** 2 + k]
        print(f'  位置{k:2d}: 功率={p:8.4f}  掩码={int(m)}  界面={int(i)}')


# ---------------------------------------------------------------------
# Volume 版：生成 3.5D 三通道数据（101×101）
#
# 内存优化说明：
#   volume 训练集拼接后是 100000×30603 float64 ≈ 24GB。
#   若先用 np.concatenate 拼出完整数组再写盘，峰值内存 ≈ 8GB(原始)+24GB(拼接) ≈ 32GB+，
#   32GB 内存的服务器会 OOM。因此这里用「memmap 源 + 逐块拼接写入」：
#     - 原始功率用 memmap 打开（不占内存）
#     - 目标文件用 memmap 预分配
#     - 每块 CHUNK 行：读功率 → 拼掩码/界面 → 写入目标
#   峰值内存 ≈ CHUNK×30603×8 ≈ 0.4GB，完全可控。
# ---------------------------------------------------------------------
CHUNK = 5000  # 每块处理的行数（控制峰值内存；5000×30603×8B ≈ 1.2GB）


def build_volume_3d5_data(res=None):
    """生成 volume 三通道数据，支持任意分辨率（默认 101，可 --res 51 下采样）。

    原始 fs_train_volume.npy 是 101×101；res=51 时下采样功率到 51×51，
    mask/interface 也用 51 分辨率，输出 fs_train_3d5_volume_51.npy。
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 目标分辨率（默认 101 = 原始）
    if res is None:
        res = N_VOLUME
    suffix = '' if res == N_VOLUME else f'_{res}x{res}'

    # 1) 生成掩码与界面（目标分辨率）
    mask = make_horizontal_mask(res).reshape(-1)          # [res²]
    interface = make_horizontal_interface(res).reshape(-1)  # [res²]
    np.save(os.path.join(DATA_DIR, f'mask_volume{suffix}.npy'), mask)
    np.save(os.path.join(DATA_DIR, f'interface_volume{suffix}.npy'), interface)
    print(f'[gen_mask] mask_volume{suffix}.npy      形状 {mask.shape}, 取值 {np.unique(mask)}')
    print(f'[gen_mask] interface_volume{suffix}.npy 形状 {interface.shape}, 取值 {np.unique(interface)}')

    # 2) 用 memmap 打开原始功率（101×101），如需下采样则逐块处理
    fs_train_orig = np.load(os.path.join(DATA_DIR, 'fs_train_volume.npy'), mmap_mode='r')
    fs_test_orig = np.load(os.path.join(DATA_DIR, 'fs_test_volume.npy'), mmap_mode='r')
    n_train, n_test = fs_train_orig.shape[0], fs_test_orig.shape[0]

    # 下采样索引（最近邻，从 101 → res）
    if res == N_VOLUME:
        idx = None  # 不用下采样
    else:
        idx = (np.arange(res) * 100.0 / (res - 1)).astype(int)

    def _read_plane(fs, i):
        """读第 i 个样本的功率平面，下采样到 res×res。"""
        plane = np.asarray(fs[i]).reshape(N_VOLUME, N_VOLUME)  # 101×101
        if idx is not None:
            plane = plane[np.ix_(idx, idx)]  # res×res
        return plane.reshape(-1)  # res²

    # 3) 逐块写入训练集（内存友好）
    fname_train = f'fs_train_3d5_volume{suffix}.npy'
    n_feat = res ** 2
    print(f'[gen_mask] 写 {fname_train} ([{n_train}, {3*n_feat}], 约 {n_train*3*n_feat*8/1e9:.1f}GB)...')
    out_tr = np.lib.format.open_memmap(os.path.join(DATA_DIR, fname_train), mode='w+',
                                       dtype=np.float64, shape=(n_train, 3 * n_feat))
    for s in range(0, n_train, CHUNK):
        e = min(s + CHUNK, n_train)
        block = np.zeros((e - s, 3 * n_feat))
        for j, i in enumerate(range(s, e)):
            block[j, :n_feat] = _read_plane(fs_train_orig, i)   # 功率
            block[j, n_feat:2*n_feat] = mask                     # 掩码
            block[j, 2*n_feat:] = interface                       # 界面
        out_tr[s:e] = block
    out_tr.flush()
    del out_tr

    # 4) 写入测试集
    fname_test = f'fs_test_3d5_volume{suffix}.npy'
    print(f'[gen_mask] 写 {fname_test} ([{n_test}, {3*n_feat}])...')
    out_te = np.lib.format.open_memmap(os.path.join(DATA_DIR, fname_test), mode='w+',
                                       dtype=np.float64, shape=(n_test, 3 * n_feat))
    for s in range(0, n_test, CHUNK):
        e = min(s + CHUNK, n_test)
        block = np.zeros((e - s, 3 * n_feat))
        for j, i in enumerate(range(s, e)):
            block[j, :n_feat] = _read_plane(fs_test_orig, i)
            block[j, n_feat:2*n_feat] = mask
            block[j, 2*n_feat:] = interface
        out_te[s:e] = block
    out_te.flush()
    del out_te

    print(f'[gen_mask] 完成: {fname_train} [{n_train}, {3*n_feat}], {fname_test} [{n_test}, {3*n_feat}]')


def _write_3d5_chunked(fname, fs_memmap, mask, interface, n_rows):
    """从 memmap 源 fs_memmap 逐块读功率、拼掩码/界面、写入目标 memmap。

    目标文件形状 [n_rows, 3*N]：第 0~N 列功率，N~2N 列掩码，2N~3N 列界面。
    """
    n_feat = mask.shape[0]                       # N = 10201
    path = os.path.join(DATA_DIR, fname)
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(n_rows, 3 * n_feat))
    for s in range(0, n_rows, CHUNK):
        e = min(s + CHUNK, n_rows)
        power = np.asarray(fs_memmap[s:e])        # [chunk, N] 读进内存
        out[s:e, :n_feat] = power                 # 功率
        out[s:e, n_feat:2 * n_feat] = mask[None, :]      # 掩码（广播）
        out[s:e, 2 * n_feat:] = interface[None, :]       # 界面（广播）
    out.flush()
    del out


if __name__ == '__main__':
    # 默认生成 surface；传 --volume 生成 volume；--res 指定分辨率（默认 101）
    import sys
    res = None
    if '--res' in sys.argv:
        i = sys.argv.index('--res')
        res = int(sys.argv[i + 1])
    if '--volume' in sys.argv:
        build_volume_3d5_data(res=res)
    else:
        build_surface_3d5_data()
