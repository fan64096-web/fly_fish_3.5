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
    返回: (model, optimizer, opt_state, runtime, best_epoch, best_mape)
          model 已回填为验证 MAPE 最低时刻的权重快照。
    """
    best_mape = float('inf')
    best_epoch = -1

    def snapshot(m):
        # 只复制数组叶子（可训练参数），非数组叶子不碰
        return jax.tree_util.tree_map(lambda x: x, eqx.filter(m, eqx.is_array))

    best_params = snapshot(model)
    leaves_treedef = None

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
            if val_mape < best_mape:
                best_mape = val_mape
                best_epoch = epoch + 1
                best_params = snapshot(model)
                mark = '  <-- best'
            print(f"  [VAL @{epoch+1}] val_mape={val_mape:.6f}{mark}")

    runtime = time.time() - start if start is not None else 0.0

    # 用最优权重回填 model（保持 pytree 结构，只换数组叶子）
    if val_eval_fn is not None and best_epoch > 0:
        model = eqx.tree_at(lambda m: tuple(jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))),
                            model, tuple(jax.tree_util.tree_leaves(best_params)))

    return model, optimizer, opt_state, runtime, best_epoch, best_mape