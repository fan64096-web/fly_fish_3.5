import jax
import jax.numpy as jnp
from jax import vmap

@jax.jit
def rel_l2(u, u_pred):
    u_norm = jnp.linalg.norm(u.reshape(-1,1))
    diff_norm = jnp.linalg.norm(u.reshape(-1,1)-u_pred.reshape(-1,1))
    return diff_norm / u_norm

@jax.jit
def rmse(u, u_pred):
    return jnp.sqrt(jnp.mean((u_pred.reshape(-1,1)-u.reshape(-1,1))**2))


@jax.jit
def mape(u, u_pred):
    return jnp.mean(jnp.abs((u.reshape(-1,1)-u_pred.reshape(-1,1))/u.reshape(-1,1)))

@jax.jit
def pape(u, u_pred):
    return jnp.max(jnp.abs((u.reshape(-1,1)-u_pred.reshape(-1,1))/u.reshape(-1,1)))

@jax.jit
def max_l1(u, u_pred):
    u_max = jnp.max(u)
    u_pred_max = jnp.max(u_pred)
    return jnp.abs(u_max - u_pred_max)


def eval_heat3d(model, test_generator, fs, u, result_dir):
    x, y, z, f, u = test_generator(fs, u)
    u_pred = model(((x,y,z),f))
    jnp.save(result_dir+'/u_pred_heat3d.npy', u_pred)
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u,u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u,u_pred)
    max_l1_u = vmap(max_l1, in_axes=(0, 0))(u,u_pred)
    
    u = 25*u + 293.15
    u_pred  = 25*u_pred + 293.15
    mape_u = vmap(mape, in_axes=(0, 0))(u,u_pred)
    pape_u = vmap(pape, in_axes=(0, 0))(u,u_pred)
    
    return (jnp.mean(rel_l2_u), jnp.std(rel_l2_u), 
            jnp.mean(rmse_u), jnp.std(rmse_u), 
            jnp.mean(max_l1_u), jnp.std(max_l1_u),
            jnp.mean(mape_u), jnp.std(mape_u),
            jnp.mean(pape_u), jnp.std(pape_u))
    
