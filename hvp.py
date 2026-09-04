import jax

# forward over forward
# (2026-09-04 瘦身: 删除零引用的 hvp_revrev/hvp_revfwd;
#  hvp_fwdrev 仅遗留 surface 流程使用, 保留)
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jax.jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# reverse over forward
def hvp_fwdrev(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.vjp(f, primals)[1](tangents[0])[0]
    primals_out, tangents_out = jax.jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out
