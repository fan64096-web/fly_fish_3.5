import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp
import equinox as eqx
from models import DeepOHeat_v1

# ---------------- 可配置项 ----------------
MODEL_PATH = "results/results_surface/DeepOHeat_v1/nf50_nc21_branch_8_256_trunk_3_64_r128/DeepOHeat_v1_trained_model.eqx"
NX, NY, NZ = 101, 101, 51
DX = 1.0 / (NX - 1)
DY = 1.0 / (NY - 1)
DZ = 0.5 / (NZ - 1)          # 厚度 0.5
K = 0.1                       # 热导率
# ---------------- 构建模型 ----------------
def load_model(path):
    model = eqx.filter_jit(DeepOHeat_v1(
        dim=3, branch_dim=21**2, field_dim=1,
        branch_depth=8, branch_hidden=256,
        trunk_depth=3, trunk_hidden=64,
        rank=128, key=jax.random.PRNGKey(42)
    ))
    model = eqx.tree_deserialise_leaves(path, model)
    return model

# ---------------- 构建稀疏矩阵 ----------------
def build_matrix():
    n = NX * NY * NZ
    rows, cols, vals = [], [], []
    for k in range(NZ):
        z = k * DZ
        for j in range(NY):
            for i in range(NX):
                row = k * NX * NY + j * NX + i
                if k == 0:  # 底面：对流 BC  u - 0.2 - 0.2*du/dz = 0  =>  (1 + 0.2/dz)*u0 - (0.2/dz)*u1 = 0.2
                    rows.append(row); cols.append(row); vals.append(1.0 + 0.2/DZ)
                    if k < NZ-1:
                        rows.append(row); cols.append(row + NX*NY); vals.append(-0.2/DZ)
                elif k == NZ-1:  # 顶面：Neumann 功率映射  du/dz = f  =>  u_{top} - u_{top-1} = f*DZ（RHS 处理）
                    rows.append(row); cols.append(row); vals.append(1.0)
                    if k > 0:
                        rows.append(row); cols.append(row - NX*NY); vals.append(-1.0)
                else:  # 内部点：∇·(k∇T) = 0, 侧面绝热
                    diag = 0.0
                    # x 方向 Neumann
                    if i == 0:
                        diag += -1.0/DX**2
                        rows.append(row); cols.append(row+1); vals.append(1.0/DX**2)
                    elif i == NX-1:
                        diag += -1.0/DX**2
                        rows.append(row); cols.append(row-1); vals.append(1.0/DX**2)
                    else:
                        diag += -2.0/DX**2
                        rows.append(row); cols.append(row-1); vals.append(1.0/DX**2)
                        rows.append(row); cols.append(row+1); vals.append(1.0/DX**2)
                    # y 方向 Neumann
                    if j == 0:
                        diag += -1.0/DY**2
                        rows.append(row); cols.append(row+NX); vals.append(1.0/DY**2)
                    elif j == NY-1:
                        diag += -1.0/DY**2
                        rows.append(row); cols.append(row-NX); vals.append(1.0/DY**2)
                    else:
                        diag += -2.0/DY**2
                        rows.append(row); cols.append(row-NX); vals.append(1.0/DY**2)
                        rows.append(row); cols.append(row+NX); vals.append(1.0/DY**2)
                    # z 方向
                    diag += -2.0/DZ**2
                    if k > 0:
                        rows.append(row); cols.append(row-NX*NY); vals.append(1.0/DZ**2)
                    if k < NZ-1:
                        rows.append(row); cols.append(row+NX*NY); vals.append(1.0/DZ**2)
                    rows.append(row); cols.append(row); vals.append(diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A

# ---------------- 构建 RHS ----------------
def build_rhs(q_v):
    """q_v: (NX, NY) 顶面功率映射（0~1 量级）"""
    b = np.zeros(NX * NY * NZ)
    # 顶面 Neumann: du/dz = f  =>  (u_top - u_topm1)/DZ = f  =>  u_top - u_topm1 = f*DZ
    top_slice = (NZ-1) * NX * NY
    b[top_slice:top_slice + NX*NY] = q_v.ravel() * DZ
    # 底面对流：1.0*u0 - 0.2/DZ*u1 = 0.2
    bot_slice = 0
    b[bot_slice:bot_slice + NX*NY] = 0.2
    return b

# ---------------- 模型初始猜测 ----------------
def surrogate_init(model, q_v):
    x_eval = jnp.linspace(0, 1, NX).reshape(-1, 1)
    y_eval = jnp.linspace(0, 1, NY).reshape(-1, 1)
    z_eval = jnp.linspace(0, 0.5, NZ).reshape(-1, 1)
    f = jnp.asarray(q_v.ravel()[:441]).reshape(1, -1)  # 模型输入 21^2=441
    T = model(((x_eval, y_eval, z_eval), f))  # (1, NX, NY, NZ, 1)
    T = np.asarray(T[0, ..., 0])  # (NX, NY, NZ)
    # 展平为 (n,)：索引顺序与矩阵一致（k*NX*NY + j*NX + i）
    return T.transpose(2, 0, 1).ravel()  # (NZ, NX, NY) -> (n,)

# ---------------- 残差度量 ----------------
def relative_residual(A, b, x):
    r = b - A @ x
    return np.linalg.norm(r) / np.linalg.norm(b)

# ---------------- 主流程 ----------------
def main():
    print("加载 surface 模型...")
    model = load_model(MODEL_PATH)
    print("模型加载完成")

    print("构建系统矩阵 (101x101x51)...")
    A = build_matrix()
    print(f"矩阵规模: {A.shape}, 非零元: {A.nnz}")

    # 用测试集第一个功率映射
    fs_test = np.load('data/fs_test_surface.npy').reshape(-1, 21**2)
    q_v = fs_test[0].reshape(21, 21)
    # 插值到 101x101（简单最近邻放大）
    q_v_big = np.zeros((NX, NY))
    for i in range(NX):
        for j in range(NY):
            q_v_big[i, j] = q_v[i * 21 // NX, j * 21 // NY]
    print(f"功率映射: {q_v_big.shape}, 范围 [{q_v_big.min():.3f}, {q_v_big.max():.3f}]")

    b = build_rhs(q_v_big)

    # 方式1：纯 GMRES（零初始）
    print("\n===== 方式1: 纯 GMRES（零初始猜测）=====")
    x0 = None
    sol1, info1 = spla.gmres(A, b, x0=x0, rtol=1e-6, maxiter=5000)
    r1 = relative_residual(A, b, sol1)
    print(f"GMRES 结束, info={info1}, 相对残差={r1:.2e}")

    # 方式2：模型初始猜测 + GMRES 细化
    print("\n===== 方式2: 算子模型初始猜测 + GMRES 细化 =====")
    x_surr = surrogate_init(model, q_v)  # 模型初始猜测用 21x21 原始功率映射
    r_surr = relative_residual(A, b, x_surr)
    print(f"模型初始解相对残差: {r_surr:.2e}")
    sol2, info2 = spla.gmres(A, b, x0=x_surr, rtol=1e-6, maxiter=5000)
    r2 = relative_residual(A, b, sol2)
    print(f"GMRES 结束, info={info2}, 细化后相对残差={r2:.2e}")

    print("\n===== 总结 =====")
    print(f"纯 GMRES 相对残差:       {r1:.2e}")
    print(f"模型初始解相对残差:       {r_surr:.2e}")
    print(f"模型+GMRES 细化相对残差:  {r2:.2e}")

if __name__ == "__main__":
    main()
