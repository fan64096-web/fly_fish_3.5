import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# 消融开关：DHV_NO_INTERFACE=1 时禁用 interface_loss（3d5 无界面热流连续对比组）
DHV_NO_INTERFACE = os.environ.get('DHV_NO_INTERFACE', '0') == '1'
# 界面项权重：默认 1.0；调大=界面约束更强，调小=更弱（可用来探索合适强度）
DHV_IFACE_LAM = float(os.environ.get('DHV_IFACE_LAM', '1.0'))
# 3d5 模式 BC：纯物理设定（Icepak 建模参数推导，不碰训练数据）
#   Icepak 边界条件（Icepak建模交接-2026-08-16.md）：
#     Substrate 底面 & die 表面: HTC=500 W/(m²·K), 环境 25°C；侧面弱对流
#   物理 Robin: u ± α·uz = u_amb,  α = k/(HTC·Lz)
#     Lz = 1.8mm/0.55 = 3.273e-3 m(每训练z单位)
#     top (die顶面, k_die=130):   α_top = 130/(500·3.273e-3) = 79.44, u_amb=0.2
#     bottom (Substrate底面, k_sub=0.5): α_bot = 0.5/(500·3.273e-3) = 0.3056, u_amb=0.2
#   注意: 参考温度 u_amb=0.2 与 baseline 原版相同(都是25°C)；真正错的是系数。
DHV_BC_TOP_REF   = float(os.environ.get('DHV_BC_TOP_REF', '0.2'))
DHV_BC_TOP_KH    = float(os.environ.get('DHV_BC_TOP_KH', '79.444'))
DHV_BC_BOT_REF   = float(os.environ.get('DHV_BC_BOT_REF', '0.2'))
DHV_BC_BOT_KH    = float(os.environ.get('DHV_BC_BOT_KH', '0.3056'))
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import argparse
import optax
from functools import partial
from models import DeepOHeat_ST, DeepOHeat_v1
from k_map import build_k_field, K_SCALE
from hvp import hvp_fwdfwd
from train import train_loop, update
from eval import eval_heat3d

@jax.jit
def create_mesh(xi_batch, yi_batch, zi_batch):
    return jnp.meshgrid(xi_batch.ravel(), yi_batch.ravel(), zi_batch.ravel(), indexing='ij')


#########################################################################
# Loss function
#########################################################################
@eqx.filter_jit
def apply_model_deepoheat_st(model, xc, yc, zc, fc, lam_b=1., k_field=None, power_dim=None, nc=101):

    def PDE_loss(model, x, y, z, f):
        # compute u
        u = model(((x, y, z), f))

        # tangent vector dx/dx dy/dy dz/dz
        v_x = jnp.ones(x.shape)
        v_y = jnp.ones(y.shape)
        v_z = jnp.ones(z.shape)

        # 1st, 2nd derivatives of u
        ux, uxx = hvp_fwdfwd(lambda x: model(((x, y, z), f)), (x,), (v_x,), True)
        uy, uyy = hvp_fwdfwd(lambda y: model(((x, y, z), f)), (y,), (v_y,), True)
        uz, uzz = hvp_fwdfwd(lambda z: model(((x, y, z), f)), (z,), (v_z,), True)

        # 网格参数（nc 驱动，支持任意分辨率）
        nz = int(0.55 * nc + 0.45)           # z 层数
        dz = 0.55 / (nz - 1)                 # z 步长
        # 功率注入的 z 索引（baseline 原版位置，勿改：z=0.10 / 0.11~0.14 / 0.15）
        #   注意：此位置是论文原版对"合成数据"的设定（热源在中介层带）。
        #   真实 Icepak 数据的热源在 die 层（训练坐标 z=0.40~0.55）。
        #   3d5 分支已改为真实 die 层注入；baseline 为对照基准保持原版不变。
        k_bot_inj = int(round(0.10 / dz))    # z=0.10
        k_top_inj = int(round(0.15 / dz))    # z=0.15

        # PDE residual（volume 版原版按 z 分层注入功率与 k，3d5 模式在此基础上乘分区 k 场）
        laplacian = (uxx + uyy + uzz)
        laplacian_bottom_power = laplacian[:,:,:,k_bot_inj:k_bot_inj+1,:]  # z=0.10
        laplacian_interior_power = laplacian[:,:,:,k_bot_inj+1:k_top_inj,:]  # z=0.11~0.14
        laplacian_top_power = laplacian[:,:,:,k_top_inj:k_top_inj+1,:]      # z=0.15

        # harmonic average
        k_bottom = 2 * 0.1 * 2 / (0.1 + 2)
        k_interior = 0.1
        k_top = 0.1

        # 3.5D 模式：f 是三通道拼接 [功率|掩码|界面]，功率注入只用功率段
        # baseline 模式：power_dim=None，f 即纯功率，行为与原版一致
        f_power = f[..., :power_dim] if power_dim is not None else f

        # 低 k 层残差归一化开关：DHV_K_NORMALIZE=1(默认) 时，3d5 无源区残差除以 k。
        #   稳态无源热传导 k∇²u=0 (k≠0) ⇔ ∇²u=0, 数学等价, 不损失物理。
        #   作用: 修复"低 k 层梯度消失" —— TIM(k=0.0015)/基板(k=0.0004) 乘 k 后
        #   残差被压小 3~6 个量级，训练完全看不见这些层；除 k 后各层 ∇² 等权。
        #   （实测 TIM 层 |∇²u|≈19 全场最大却贡献≈0.001, 正因被 k 吞掉。）
        #   有源 die 层(k=0.1)不产生梯度消失, 维持 k∇²u+q 物理形式。
        DHV_K_NORM = os.environ.get('DHV_K_NORMALIZE', '1') == '1'
        if k_field is not None:
            # ============ 3d5 模式：功率注入真实 die 层 ============
            # 训练坐标 z 的物理层映射（与 make_icepak_dataset.py / k_map.py 一致）：
            #   z<0.10 Substrate / 0.10~0.35 Interposer / 0.35~0.40 TIM / 0.40~0.55 die
            # 真实热源在 die（封装顶层），原代码注入 z=0.10~0.15（Interposer）是错位，
            # 改为注入 z=0.40~0.55（die 层）。baseline 分支保持原版位置不动。
            k_die_bot = int(round(0.40 / dz))     # z=0.40 die 底
            k_die_top = int(round(0.55 / dz))     # z=0.55 die 顶（=上层边界）
            # die 下方（z<0.40：Substrate/Interposer/TIM）无功率：
            #   开启归一化 => 只用 ∇²u（无源区 k∇²u=0 ⇔ ∇²u=0）; 关闭 => k_field·∇²u
            if DHV_K_NORM:
                below = laplacian[:,:,:,0:k_die_bot,:]
            else:
                below = k_field[:,:,:,0:k_die_bot,:] * laplacian[:,:,:,0:k_die_bot,:]
            # die 底 1 层（z=0.40）×1 功率
            bot_pow = k_field[:,:,:,k_die_bot:k_die_bot+1,:] * laplacian[:,:,:,k_die_bot:k_die_bot+1,:] \
                      + f_power.reshape(-1, nc, nc, 1, 1)
            # die 中部（z=0.41~0.54）×2 功率
            int_pow = k_field[:,:,:,k_die_bot+1:k_die_top,:] * laplacian[:,:,:,k_die_bot+1:k_die_top,:] \
                      + 2 * f_power.reshape(-1, nc, nc, 1, 1)
            # die 顶 1 层（z=0.55）×1 功率
            top_pow = k_field[:,:,:,k_die_top:k_die_top+1,:] * laplacian[:,:,:,k_die_top:k_die_top+1,:] \
                      + f_power.reshape(-1, nc, nc, 1, 1)
            # 拼接（z 从大到小，与 baseline 分支组织方式一致）
            pde_res = jnp.concatenate([top_pow, int_pow, bot_pow, below], axis=3)
        else:
            # baseline 模式：原版分段（各 z 段固定 k）
            pde_res = jnp.concatenate([0.1*laplacian[:,:,:,k_top_inj+1:,:],
                                        k_top*laplacian_top_power+f_power.reshape(-1, nc, nc, 1, 1),
                                        k_interior*laplacian_interior_power+2*f_power.reshape(-1, nc, nc, 1, 1),
                                        k_bottom*laplacian_bottom_power+f_power.reshape(-1, nc, nc, 1, 1),
                                        0.1*laplacian[:,:,:,0:k_bot_inj,:]
                ],axis=3)
        pde_res = jnp.mean(pde_res**2)


        # top surface (Robin: u ± α·uz = u_amb，符号按外法向)
        #   baseline: 论文原版(0.2/系数2)不动
        #   3d5: 纯物理设定（Icepak HTC=500/环境25°C 推导, 不碰训练数据）
        #     top    (die顶面, k=130):    u + 79.44·uz = 0.2
        #     bottom (Substrate底面,k=0.5): u − 0.306·uz = 0.2
        #   原版系数 top=2/bottom=40 与物理推导方向相反、量级差几十倍
        #   （原版 top 系数过小→顶面几乎不约束; bottom 40 过大→残差≈1.5e4 巨大）
        if k_field is not None:
            bc_top = jnp.mean((u[:,:,:,-1,:] - DHV_BC_TOP_REF + DHV_BC_TOP_KH*uz[:,:,:,-1,:])**2)
            bc_bottom = jnp.mean((u[:,:,:,0,:] - DHV_BC_BOT_REF - DHV_BC_BOT_KH*uz[:,:,:,0,:])**2)
        else:
            bc_top = jnp.mean((u[:,:,:,-1,:] - 0.2 + 2*uz[:,:,:,-1,:])**2)
            bc_bottom = jnp.mean((u[:,:,:,0,:] - 0.2 - 40*uz[:,:,:,0,:])**2)
        # other_surfaces（x/y 侧面，真实数据近似绝热 —— baseline 与 3d5 均相同）
        bc_other = jnp.mean((uy[:,:,0,:,:])**2) + jnp.mean((uy[:,:,-1,:,:])**2) + jnp.mean((ux[:,0,:,:,:])**2) + jnp.mean((ux[:,-1,:,:,:])**2)

        # 界面热流连续：界面两侧 k·∂u/∂n 相等
        #   - 仅 3d5 模式启用
        #   - 消融时设 DHV_NO_INTERFACE=1 可关闭（对比"无界面热流连续"）
        # 异质界面在 z 向层间（3.5D 纵向结构，与 k_map / make_icepak_dataset 一致）：
        #   z=0.10  Substrate|Interposer   k: 0.5→130   真实数据热流基本连续 ✅
        #   z=0.35  Interposer|TIM         k: 130→2     真实数据热流基本连续 ✅
        #   （z=0.40 TIM|die 不做界面约束：die 是功率注入层+接触热阻，
        #     真实数据在那里热流明显不连续（Δk·∂u/∂z 为其他界面 3.7 倍），
        #     强制连续会把 die 层温度"焊死"、使 die 预测误差变差）
        # 界面两类网格点（下侧 zi-1 / 上侧 zi）的热流连续 → k·uz 相等
        #   ⚠️ BUG 修复(2026-08-22)：k_field 已被 K_SCALE=1300 缩放(硅130→0.1)，
        #   直接用缩放 k 算界面项会被压低 K_SCALE²≈169万倍，实测 interface_loss≈4.6e-7
        #   而 PDE 残差≈6.27，界面项占比≈0.000007%，等于没参与训练（3d5-full ≈ no-interface
        #   的原因就在这，不是"界面约束有害"）。这里乘回 K_SCALE 用真实物理 k 计算，
        #   界面热流连续才有真正的约束力。强度由 DHV_IFACE_LAM 控制。
        if k_field is not None and not DHV_NO_INTERFACE:
            z_ifaces = (0.10, 0.35)            # 训练坐标系 z 界面位置（mm）
            interface_loss = 0.0
            for z_iface in z_ifaces:
                zi = int(round(z_iface / dz))      # 界面上侧网格索引
                k_below = k_field[:, :, :, zi-1, :] * K_SCALE   # 下侧真实材料 k
                k_above = k_field[:, :, :, zi, :]   * K_SCALE   # 上侧真实材料 k
                uz_below = uz[:, :, :, zi-1, :]
                uz_above = uz[:, :, :, zi, :]
                interface_loss += jnp.mean((k_above*uz_above - k_below*uz_below)**2)
            interface_loss = interface_loss / len(z_ifaces) * DHV_IFACE_LAM   # 平均到每个界面
        else:
            interface_loss = 0.0


        return pde_res + lam_b*(bc_top + bc_bottom + bc_other) + interface_loss


    # isolate loss func from redundant arguments
    loss_fn = lambda model: PDE_loss(model, xc, yc, zc, fc)
                       

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


#########################################################################
# Train generator
#########################################################################
# 注意：这个函数不能用 jax.jit，否则 JAX 会把整个 memmap 数组（24GB）
# 捕获成编译常量，导致 OOM。改用 numpy 随机采样，只把 batch 数据转成 jax 数组。
# 调用处也必须是普通 lambda（不能包 jax.jit）。
def deepoheat_st_train_generator(fs, batch, nc, key):
    """返回坐标网格与 batch 个功率映射（jax 数组）。

    fs 为 numpy/memmap 数组（CPU），这里用 numpy 采样避免 JAX 常量捕获。
    """
    nx = nc
    ny = nc
    nz = int(0.55*nc + 0.45)

    # 用 numpy 随机采样（fs 是 memmap，读入 batch 行不占太多内存）
    # key 是 jax PRNG key（[2] uint32 数组），转成 numpy 标量作为 numpy 采样种子
    seed = int(np.asarray(key[0]))
    idx = np.random.default_rng(seed).choice(fs.shape[0], size=batch, replace=False)
    fc = np.asarray(fs[idx, :])            # [batch, dim]，转成 numpy 数组
    fc = jnp.asarray(fc)                   # 再转 jax 数组

    xc = jnp.linspace(0, 1, nx).reshape(-1,1)
    yc = jnp.linspace(0, 1, ny).reshape(-1,1)
    zc = jnp.linspace(0, 0.55, nz).reshape(-1,1)

    return xc, yc, zc, fc


#########################################################################
# Test generator
#########################################################################
@partial(jax.jit, static_argnums=(2,))
def deepoheat_st_test_generator(fs, u, nc=101):
    nz = int(0.55 * nc + 0.45)
    x = jnp.linspace(0, 1, nc).reshape(-1,1)
    y = jnp.linspace(0, 1, nc).reshape(-1,1)
    z = jnp.linspace(0, 0.55, nz).reshape(-1,1)
    return x, y, z, fs, u


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')
    parser.add_argument('--model_name', type=str, default='DeepOHeat_v1', choices=['DeepOHeat_ST', 'DeepOHeat_v1'], help='model name (DeepOHeat_ST; DeepOHeat_v1)')
    parser.add_argument('--device_name', type=int, default=0, choices=[0, 1], help='GPU device')
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', '3d5'],
                        help='input mode: baseline=原版; 3d5=分区k PDE')

    # training data settings
    parser.add_argument('--nc', type=int, default=101, help='the number of input points for each axis')
    parser.add_argument('--batch', type=int, default=50, help='the number of train functions')
    
    # training settings
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100000, help='training epochs')
    parser.add_argument('--log_epoch', type=int, default=100, help='log the loss every chosen epochs')

    # model settings
    parser.add_argument('--dim', type=int, default=3, help='the input size')
    parser.add_argument('--branch_dim', type=int, default=101**2, help='the number of sensors for indentifying an input function')
    parser.add_argument('--field_dim', type=int, default=1, help='the dimension of the output field')
    parser.add_argument('--branch_depth', type=int, default=8, help='the number of hidden layers, including the output layer')
    parser.add_argument('--branch_hidden', type=int, default=256, help='the size of each hidden layer')
    parser.add_argument('--trunk_depth', type=int, default=3, help='the number of hidden layers, including the output layer')
    parser.add_argument('--trunk_hidden', type=int, default=64, help='the size of each hidden layer')
    parser.add_argument('--r', type=int, default=128, help='rank*field_dim equals the output size')
    
    args = parser.parse_args()


    # 数据加载：按 mode 选择 Branch 输入通道
    # 注意：必须用 np.load(mmap_mode='r') 而不是 jnp.load！
    #   jnp.load 返回 JAX 数组，np.asarray(fs[idx,:]) 会触发全量 GPU 传输导致 OOM。
    #   np.load(mmap_mode='r') 返回 numpy memmap，索引只读 batch 行，不碰 GPU。
    # 分辨率相关文件名后缀（nc 决定）：nc=101 用默认，nc=51 用 _51x51
    _res = args.nc
    _suf = '' if _res == 101 else f'_{_res}x{_res}'
    args.branch_dim = _res ** 2  # branch_dim 跟随分辨率

    if args.mode == '3d5':
        # 3.5D 模式：功率+掩码+界面 三通道（数据由 gen_mask.py 的 volume 版生成）
        # nc=101: fs_train_3d5_volume.npy; nc=51: fs_train_3d5_volume_51x51.npy
        fs_train = np.load(f'data/fs_train_3d5_volume{_suf}.npy', mmap_mode='r')
        fs_test = np.load(f'data/fs_test_3d5_volume{_suf}.npy', mmap_mode='r')
        args.channels = 3
        if args.model_name != 'DeepOHeat_v1':
            print(f'[mode=3d5] 仅支持 DeepOHeat_v1，已强制切换（原为 {args.model_name}）')
            args.model_name = 'DeepOHeat_v1'
    else:
        # baseline 模式：仅功率（原版），也用 memmap 降低内存
        # 注意：fs_train_volume.npy 原始形状是 [N, 101, 101]，必须 reshape 成 [N, 101**2]
        fs_train = np.load('data/fs_train_volume.npy', mmap_mode='r').reshape(-1, 101**2)
        fs_test = np.load('data/fs_test_volume.npy', mmap_mode='r').reshape(-1, 101**2)
        args.channels = 1
    # 评估真值：统一用 Icepak 真实温度场（baseline 与 3d5 一致，保证公平对比）
    #   注意：旧逻辑曾让 3d5 优先用"自洽真值" data/u_test_3d5.npy（gen_truth_3d5.py 生成，
    #   即 3d5 分区 k 假设下 PDE 的解，给 3d5 一个"自己假设下的标准答案"）。但自洽真值
    #   是"模型应学到的物理"，不是"Icepak 真实温度"；用它评估会与 baseline（用 Icepak
    #   真值）不可比。Icepak 真实数据具备后，统一用它评估，删除自洽真值分支（2026-08-25）。
    u_test = np.load('data/u_test_volume.npy')
    print(f'评估真值: data/u_test_volume.npy (Icepak 真实温度场, 形状 {u_test.shape})')

    # result dir
    root_dir = os.path.join(os.getcwd(), 'results', 'results_volume', args.model_name)
    result_dir = os.path.join(root_dir, 'nf'+str(args.batch)+'_nc'+str(args.nc) + '_branch_' + str(args.branch_depth) +
                              '_'+str(args.branch_hidden)+'_trunk_' + str(args.trunk_depth) +
                              '_'+str(args.trunk_hidden)+'_r'+ str(args.r) + '_mode_' + str(args.mode))
    
    # make dir
    os.makedirs(result_dir, exist_ok=True)
    
    # logs
    if os.path.exists(os.path.join(result_dir, 'log (loss).csv')):
        os.remove(os.path.join(result_dir, 'log (loss).csv'))

    if os.path.exists(os.path.join(result_dir, 'log (eval metrics).csv')):
        os.remove(os.path.join(result_dir, 'log (eval metrics).csv'))
        
    if os.path.exists(os.path.join(result_dir, 'log (physics_loss).csv')):
        os.remove(os.path.join(result_dir, 'log (physics_loss).csv'))
    
    if os.path.exists(os.path.join(result_dir, 'total parameters.csv')):
        os.remove(os.path.join(result_dir, 'total parameters.csv'))
    
    if os.path.exists(os.path.join(result_dir, 'total runtime (sec).csv')):
        os.remove(os.path.join(result_dir, 'total runtime (sec).csv'))

    if os.path.exists(os.path.join(result_dir, 'memory usage (mb).csv')):
        os.remove(os.path.join(result_dir, 'memory usage (mb).csv'))


    # update function
    update_fn = update

    # define the optimizer
    schedule = optax.exponential_decay(args.lr,1000,0.9)
    optimizer = optax.adam(schedule)

    # random key
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)

    # init model
    if args.model_name == 'DeepOHeat_ST':
        model = eqx.filter_jit(DeepOHeat_ST(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim, 
                                                           branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth, 
                                                           trunk_hidden=args.trunk_hidden, rank=args.r, key=subkey))
    else:
        model = eqx.filter_jit(DeepOHeat_v1(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim,
                                                        branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth,
                                                        trunk_hidden=args.trunk_hidden, rank=args.r, channels=args.channels, key=subkey))
    
    # Filter the model to get only the trainable parameters
    params = eqx.filter(model, eqx.is_array)
    # Count the total number of parameters by summing the size of each array
    num_params = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, params)))
    print(f'Total number of parameters: {num_params}')
    
    # init state
    key, subkey = jax.random.split(key)
    opt_state = optimizer.init(params)

    # train/test generator
 
    train_generator = lambda key: deepoheat_st_train_generator(fs_train, args.batch, args.nc, key)
    test_generator = jax.jit(lambda fs, u: deepoheat_st_test_generator(fs, u, args.nc))
    if args.mode == '3d5':
        # 3d5 模式：PDE 残差使用分区 k 场
        nx, nz = args.nc, int(0.55*args.nc + 0.45)
        _xc = jnp.linspace(0, 1, nx).reshape(-1, 1)
        _yc = jnp.linspace(0, 1, nx).reshape(-1, 1)
        _zc = jnp.linspace(0, 0.55, nz).reshape(-1, 1)
        k_field = build_k_field(_xc, _yc, _zc)  # [1, nx, nx, nz, 1]
        print(f'[3d5] 分区 k 场已构造, 形状 {k_field.shape}, 值域 [{float(k_field.min()):.5f}, {float(k_field.max()):.4f}]')
        loss_fn = lambda model, xc, yc, zc, fc: apply_model_deepoheat_st(model, xc, yc, zc, fc, k_field=k_field, power_dim=args.nc**2, nc=args.nc)
    else:
        # baseline 模式：与原版完全一致
        loss_fn = lambda model, xc, yc, zc, fc: apply_model_deepoheat_st(model, xc, yc, zc, fc, nc=args.nc)
  

    # train the model
    model, optimizer, opt_state, runtime = train_loop(model, optimizer, opt_state, update_fn, train_generator, loss_fn, args.epochs, args.log_epoch, result_dir, args.device_name, subkey)
    
    # save the model
    eqx.tree_serialise_leaves(os.path.join(result_dir,args.model_name+'_trained_model.eqx'),model)
    
    
    
    
    # eval the trained model
    rel_l2_mean, rel_l2_std, rmse_mean, rmse_std, max_l1_mean, max_l1_std, mape_mean, mape_std, pape_mean, pape_std = eval_heat3d(model,test_generator,fs_test,u_test,result_dir)
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)')
    print(f'rel_l2 --> mean: {rel_l2_mean:.8f} (std: {rel_l2_std: 8f})')
    print(f'rmse --> mean: {rmse_mean:.8f} (std: {rmse_std: 8f})')
    print(f'max_l1 --> mean: {max_l1_mean:.8f} (std: {max_l1_std: 8f})')
    print(f'mape --> mean: {mape_mean:.8f} (std: {mape_std: 8f})')
    print(f'pape --> mean: {pape_mean:.8f} (std: {pape_std: 8f})')
    
    
    
    # # save runtime and eval metrics
    runtime = np.array([runtime])
    num_params = np.array([num_params])
    np.savetxt(os.path.join(result_dir, 'total runtime (sec).csv'), runtime, delimiter=',')
    np.savetxt(os.path.join(result_dir, 'total parameters.csv'), num_params, delimiter=',')
    with open(os.path.join(result_dir, 'log (eval metrics).csv'), 'a') as f:
        f.write(f'rel_l2_mean: {rel_l2_mean}\n')
        f.write(f'rel_l2_std: {rel_l2_std}\n')
        f.write(f'rmse_mean: {rmse_mean}\n')
        f.write(f'rmse_std: {rmse_std}\n')
        f.write(f'max_l1_mean: {max_l1_mean}\n')
        f.write(f'max_l1_std: {max_l1_std}\n')
        f.write(f'mape_mean: {mape_mean}\n')
        f.write(f'mape_std: {mape_std}\n')
        f.write(f'pape_mean: {pape_mean}\n')
        f.write(f'pape_std: {pape_std}\n')