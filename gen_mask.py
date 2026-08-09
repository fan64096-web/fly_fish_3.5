"""
gen_mask.py
===========
DeepOHeat-V1 改造 · 阶段 1：生成 3.5D 区域掩码（mask）与界面编码（interface）数据

用途
----
为 Branch 网络生成两个新输入通道：
  1. mask       : 区域掩码，告诉模型每个 (x, y) 位置属于哪个区域（chiplet）
  2. interface  : 界面编码，告诉模型每个 (x, y) 位置是否在区域交界处

几何设定（AI 芯片 3.5D 布局，贴合产品实际）
-------------------------------------------
横向：2 GPU + 4 HBM + Interposer 边缘（对应 k_map.horizontal_region）

        y=1
      ┌───────────────┐
      │ HBM │ GPU │ HBM│
      │-----+-----+----│
      │ HBM │ GPU │ HBM│
      └───────────────┘
   x=0             x=1

  - mask      : 每个 (x,y) 的横向材料类型（0=GPU, 1=HBM, 2=Interposer边缘）
  - interface : GPU/HBM 边界 + 芯片边缘（Interposer 边缘）记 1

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

    mask 通道 = 材料/功能区域分类（0~4），与 k_map 一致：
        0 = Compute Die（中心 2×2）
        1 = HBM（四角）
        2 = Interposer（die 之间的间隙区域，材料为中介层）
        （Substrate=3, TIM=4 由纵向层定义，见 k_map.material_id）

    无独立"间隙"区域；间隙归为 Interposer 材料（ID=2）。
    interface 通道单独标记界面，不作为区域 ID。
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    is_compute = (xx >= 0.3) & (xx <= 0.7) & (yy >= 0.3) & (yy <= 0.7)
    is_hbm = (
        ((xx < 0.3) & (yy < 0.3)) |
        ((xx > 0.7) & (yy < 0.3)) |
        ((xx < 0.3) & (yy > 0.7)) |
        ((xx > 0.7) & (yy > 0.7))
    )
    # 非 Compute 非 HBM = Interposer（材料 2）
    mask = np.where(is_compute, 0, np.where(is_hbm, 1, 2)).astype(np.float64)
    return mask


def make_horizontal_interface(n=N_SURFACE):
    """生成界面编码：die 间隙（Interposer 上方）记 1，其余记 0。

    异质界面 = Compute/HBM die 与 die 之间的间隙（Interposer 暴露区）。
    返回 [n, n] float64 数组。
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    interface = np.zeros((n, n), dtype=np.float64)
    # die 间隙十字线：x≈0.3, 0.7 和 y≈0.3, 0.7
    interface[np.isclose(xx, 0.3)] = 1.0
    interface[np.isclose(xx, 0.7)] = 1.0
    interface[np.isclose(yy, 0.3)] = 1.0
    interface[np.isclose(yy, 0.7)] = 1.0
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


def build_volume_3d5_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) 生成掩码与界面（复用 2×2 几何，分辨率 101）
    mask = make_horizontal_mask(N_VOLUME).reshape(-1)          # [10201]
    interface = make_horizontal_interface(N_VOLUME).reshape(-1)  # [10201]
    np.save(os.path.join(DATA_DIR, 'mask_volume.npy'), mask)
    np.save(os.path.join(DATA_DIR, 'interface_volume.npy'), interface)
    print(f'[gen_mask] mask_volume.npy      形状 {mask.shape}, 取值 {np.unique(mask)}')
    print(f'[gen_mask] interface_volume.npy 形状 {interface.shape}, 取值 {np.unique(interface)}')

    # 2) 用 memmap 打开原始功率（不占内存）
    fs_train = np.load(os.path.join(DATA_DIR, 'fs_train_volume.npy'), mmap_mode='r').reshape(-1, N_VOLUME ** 2)
    fs_test = np.load(os.path.join(DATA_DIR, 'fs_test_volume.npy'), mmap_mode='r').reshape(-1, N_VOLUME ** 2)
    n_train, n_test = fs_train.shape[0], fs_test.shape[0]
    print(f'[gen_mask] 原始功率 fs_train {fs_train.shape}, fs_test {fs_test.shape}')

    # 3) 逐块拼接写入训练集
    print('[gen_mask] 写 volume 训练集（约 24GB，逐块写入，峰值内存低）...')
    _write_3d5_chunked('fs_train_3d5_volume.npy', fs_train, mask, interface, n_train)

    # 4) 逐块拼接写入测试集
    print('[gen_mask] 写 volume 测试集...')
    _write_3d5_chunked('fs_test_3d5_volume.npy', fs_test, mask, interface, n_test)

    print(f'[gen_mask] 完成: fs_train_3d5_volume.npy [{n_train}, {3*N_VOLUME**2}], '
          f'fs_test_3d5_volume.npy [{n_test}, {3*N_VOLUME**2}]')


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
    # 默认生成 surface；传 --volume 生成 volume
    import sys
    if '--volume' in sys.argv:
        build_volume_3d5_data()
    else:
        build_surface_3d5_data()
