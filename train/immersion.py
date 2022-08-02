import jax
from jax import vmap, random, numpy as jnp, lax, jacfwd
from functools import partial



class Immersion(object):
    def __init__(self, decoder):
        self.decoder = decoder
        self.lower_bound, self.upper_bound = -4.0, 4.0

    @partial(jax.jit, static_argnums=(0,))
    def metric_tensor(self, z):
        N,d = z.shape
        Jg = (vmap(jacfwd(self.decoder))(z)).reshape(N,-1,d)
        M = vmap(jnp.dot)(jnp.transpose(Jg, (0,2,1)), Jg)
        return M

    @partial(jax.jit, static_argnums=(0,3,6), static_argnames=('num_steps',))
    def random_walk(self, prng, z, scale=1.0, step_size=0.01, num_steps=10, boundary=False):
        def body(carry, _):
            z_, key = carry
            key, subkey = random.split(key)
            noise = random.normal(subkey, z_.shape)
            M = self.metric_tensor(z_)*scale
            L,U = vmap(jnp.linalg.eigh)(M)
            coef = vmap(jnp.dot)(U, 1./jnp.sqrt(L) *noise)
            zt = z_ + step_size * coef

            if boundary:
                zt = jnp.where(zt < self.upper_bound, zt, self.upper_bound)
                zt = jnp.where(zt > self.lower_bound, zt, self.lower_bound)
            carry = (zt, key)
            return carry, zt
        prefix, dim_z = z.shape[:-1], z.shape[-1]
        z0 = jnp.reshape(z, (-1, dim_z))
        _, zts = lax.scan(body, (z0, prng), jnp.arange(num_steps))
        zts = jnp.reshape(zts, (num_steps, *prefix, dim_z))
        return zts
    
    @partial(jax.jit, static_argnums=(0,))
    def jac_hess(self, z):
        d = z.shape[-1]
        Jg = jnp.reshape(jacfwd(self.decoder)(z),(-1,d))
        Hg = jnp.reshape(jacfwd(jacfwd(self.decoder))(z), (-1,d*d))
        return Jg, Hg

    @partial(jax.jit, static_argnums=(0,))
    def brownian(self, rng, z, t, S = None):
        if S is None: S = jnp.eye(z.shape[-1])

        t = jnp.asarray(t, dtype=z.dtype)
        prefix = jnp.broadcast_shapes(t.shape, z.shape[:-1])
        noise = jax.random.multivariate_normal(rng, mean=jnp.zeros((z.shape[-1],)),
                                                    cov=S, shape=prefix)
        zt = z + noise * jnp.expand_dims(t, axis=(1,))
        return zt