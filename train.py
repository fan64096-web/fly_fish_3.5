import jax
import equinox as eqx
import time
import os
import GPUtil

def get_gpu_memory(device_name):
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[device_name].memoryUsed 
    return None


# Define your update function
@eqx.filter_jit
def update(grads, optimizer, opt_state, model):
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state

# Define your training loop
def train_loop(model, optimizer, opt_state, update_fn, train_generator, loss_fn, num_epochs, log_epoch, result_dir, device_name, key):
    # key, subkey = jax.random.split(key)
    # inputs = train_generator(subkey)
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        inputs = train_generator(subkey)
        

        loss, grads = loss_fn(model, *inputs)
        model, opt_state = update_fn(grads, optimizer, opt_state, model)

        if epoch == 1:
            gpu_memory = get_gpu_memory(device_name)
            with open(os.path.join(result_dir, 'memory usage (mb).csv'), 'a') as f:
                f.write(f'{gpu_memory}\n')
            start = time.time()

        if epoch % log_epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss}")
            with open(os.path.join(result_dir, 'log (loss).csv'), 'a') as f:
                f.write(f'{loss}\n')
                
            
            
    runtime = time.time() - start

    return model, optimizer, opt_state, runtime


# ---------------------------------------------------------------------------
# 固定协议训练循环：验证集早停选优（2026-08-28 新增）
# ---------------------------------------------------------------------------
# 背景：纯 PDE 自监督训练下，训练 loss 与测试 MAPE 严重脱节，最后一轮权重
#   是"任意的"（同一模型 10000 轮能跑出 2.6%~6.7% 的漂移）。解决：
#   每 eval_every 轮在【验证集】(训练集内划出、不参与训练、与测试集无关)
#   上评一次 MAPE，保留历史最优权重，训练结束用最优权重做最终评估。
# 公平性：baseline 与 3d5 用完全相同的协议（同验证集、同评估频率、同选优
#   标准），唯一区别仍是物理假设。验证集来自训练集划分，测试集只在最后碰。
def train_loop_valsel(model, optimizer, opt_state, update_fn, train_generator,
                      loss_fn, num_epochs, log_epoch, result_dir, device_name, key,
                      val_eval_fn=None, eval_every=200):
    """带验证集选优的训练循环。

    val_eval_fn: callable(model) -> float，验证集 MAPE（越小越好）。
                 传 None 则等价普通训练（返回末轮权重）。
    eval_every:  每 N 轮评一次验证集。
    返回: (model, optimizer, opt_state, runtime, best_epoch, best_mape, final_model)
          model 已回填为验证 MAPE 最低时刻的权重快照；
          final_model = 末轮权重快照（供 model soup 用，P2-8；valsel 关闭时为 None）。
    """
    best_mape = float('inf')
    best_epoch = -1

    def snapshot(m):
        # 完整模型深拷贝(数组叶子复制, 静态叶子共享)——结束时直接返回最优时刻的
        # 整个模型。不用 tree_at 做结构级替换: equinox 要求 where 回调只依赖
        # PyTree 结构、不得依赖叶子值, 传入 tree_leaves(...) 的写法在运行时会
        # 触发 ValueError(`where` must use just the PyTree structure)。
        return jax.tree_util.tree_map(lambda x: x, m)

    best_model = snapshot(model)

    start = None
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        inputs = train_generator(subkey)
        loss, grads = loss_fn(model, *inputs)
        model, opt_state = update_fn(grads, optimizer, opt_state, model)

        if epoch == 1:
            start = time.time()

        if epoch % log_epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss}")
            with open(os.path.join(result_dir, 'log (loss).csv'), 'a') as f:
                f.write(f'{loss}\n')

        # 验证集评估 + 最优权重保留
        if val_eval_fn is not None and (epoch + 1) % eval_every == 0:
            val_mape = float(val_eval_fn(model))
            with open(os.path.join(result_dir, 'log (val mape).csv'), 'a') as f:
                f.write(f'{epoch+1},{val_mape}\n')
            mark = ''
            # 用 <= 而非 <: both 模式下"双指标同时到达前沿"的点得分恒为 2.0,
            # 若用 < 则首个评估点(2.0)之后永不更新, 最优权重冻结在epoch500(严重bug)。
            # <= 语义 = 每个"同时触及双指标前沿"的更晚检查点都覆盖前一快照,
            # 最终保留训练中最后一个双指标前沿点(且其后的发散点得分>2不会被选)。
            if val_mape <= best_mape:
                best_mape = val_mape
                best_epoch = epoch + 1
                best_model = snapshot(model)
                mark = '  <-- best'
            # 注意: 打印的是 val_eval_fn 的返回值——both 模式下是"选优得分"
            # (峰值/历史最优+平均/历史最优, 发散插曲时会很大, 无害, 不会被选中),
            # 单指标模式下才是该指标本身。真实指标轨迹见 _valeval/val_detail.csv。
            print(f"  [VAL @{epoch+1}] sel_score={val_mape:.6f}{mark}")

    runtime = time.time() - start if start is not None else 0.0

    # 末轮权重快照（soup 候选；在 best 回填前抓取）
    final_model = snapshot(model) if val_eval_fn is not None else None

    # 直接返回最优时刻的完整模型快照
    if val_eval_fn is not None and best_epoch > 0:
        model = best_model

    return model, optimizer, opt_state, runtime, best_epoch, best_mape, final_model