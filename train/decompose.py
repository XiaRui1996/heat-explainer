from flax.training.train_state import TrainState
from flax.training import checkpoints
from pathlib import Path
import numpy as np
from functools import partial
from absl import logging
from jax import random, vmap, jacfwd, hessian, numpy as jnp
import jax
import optax

from heat_explainer.train.predict import restore_checkpoints as predict_restore
from heat_explainer.train.vae import restore_checkpoints as vae_restore
from heat_explainer.train.immersion import Immersion
from heat_explainer.trainutils.utils import to_heatmap
from heat_explainer.datautils.dataloader import SyntheticDataset

KEY = random.PRNGKey(14)


@jax.jit
def hess_operator(Jg, Hg, gradf, Hf):
    Ginv = jnp.linalg.inv(Jg.T.dot(Jg)) #d,d
    JgGinv = Jg.dot(Ginv) #D,d
    HJGinvf = jnp.dot(Hg.T, JgGinv.dot(gradf.T))  #d*d,o
    res = Hf - jnp.reshape(HJGinvf.T, Hf.shape) #o,d,d
    res = jnp.transpose(res, (1,2,0)) #d,d,o
    result = vmap(jnp.dot)(JgGinv, JgGinv.dot(res)) #D,o
    return result

@jax.jit
def grad_hess(vae, heatkernel, z):
    def zmaptoy(z):
        x = vae.apply_fn({'params':vae.params}, z, KEY, mode='decode')
        y = heatkernel.apply_fn({'params':heatkernel.params}, x)
        return y
    gradf = jnp.squeeze(jacfwd(zmaptoy)(z)) #o,d
    Hf = jnp.squeeze(hessian(zmaptoy)(z)) #o,d,d
    return gradf, Hf

@jax.jit
def grad_hess_synthetic(decoder, heatkernel, z):
    def zmaptoy(z):
        x = decoder(z)
        y = heatkernel.apply_fn({'params':heatkernel.params}, x)
        return y
    gradf = jnp.squeeze(jacfwd(zmaptoy)(z)) #o,d
    Hf = jnp.squeeze(hessian(zmaptoy)(z)) #o,d,d
    return gradf, Hf

@partial(jax.jit, static_argnums=(1,2,3))
def restore_heatkernel_firststep(state, step, ckptdir, learner='adamw'):
    if learner == 'sgd':
        learning_rate_fn = optax.cosine_decay_schedule(
                            init_value = 0.0001,
                            decay_steps= 20 * 60000//32)
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=0.9,
            nesterov=True,
        )
        state = TrainState.create(apply_fn=state.apply_fn,params=state.params,tx=tx)

    return checkpoints.restore_checkpoint(ckpt_dir=ckptdir,
                                          target=state, step=step)

@partial(jax.jit, static_argnums=(1,2))
def restore_heatkernel(state, step, ckptdir):
    return checkpoints.restore_checkpoint(ckpt_dir=ckptdir,
                                          target=state, step=step)

@jax.jit
def vae_predict(state, x, rng):
    recon_x, mean_z, _ = state.apply_fn({'params':state.params}, x, rng, training=False)
    return recon_x, mean_z

@jax.jit
def kernel_predict(state, x):
    return state.apply_fn({'params': state.params}, x)

def laplacian_postprocess(results, config, compare):
    t = results.shape[0]
    W, C = config.image_size, config.channels

    decomps = []
    for r in compare:
        decomp_r = results[..., r].reshape(t,W,W,C).reshape(-1,W,C)
        decomps.append(to_heatmap(decomp_r, percentile=100).reshape(t,W,W,3))
    decomps = np.stack(decomps) 

    return decomps #r, t, w, w, 3


def laplacian_decompose(test_batch, test_label, config, workdir):
    savedir = Path(workdir) / 'results'
    savedir.mkdir(parents=True, exist_ok=True)

    ckptdir = Path(workdir)  / 'heatkernel'
    vaedir = Path(workdir) / 'vae'
    predictordir = Path(workdir) / 'predictor'
    latest = int((checkpoints.latest_checkpoint(ckpt_dir=ckptdir)).split('_')[-1])
    logging.info(f"Latest heat step: {latest}")

    key = random.PRNGKey(14)

    if config.tag == 'synthetic':
        s = SyntheticDataset(config.image_size, 6, 60000, config.datadir, '/train', generate=False)
        decoder = s.decode_jax
    else:
        vae = vae_restore(config, vaedir)
        logging.info(f"Restored vae and heatkernel")
        @jax.jit
        def decoder(z): return vae.apply_fn({'params':vae.params}, z, key, mode='decode')
    heatkernel = predict_restore(config, predictordir)
    
    manifold = Immersion(decoder)

    delta = config.stepsize * config.stride
    D, d, o = config.image_size ** 2 * config.channels, config.vae_d, config.num_classes

    decomp_all = []
    for j in range(test_batch.shape[0]):
        results, result = jnp.empty((0, D, o)), jnp.zeros((D, o), dtype=np.float32)
        prediction = []

        x = test_batch[j:j+1]
        if config.tag == 'synthetic':
            z = s.encode_jax(x)
        else:
            recon_x, z = vae_predict(vae, x, key)
        Jg, Hg = manifold.jac_hess(z)
        for step in range(latest + 1):
            if step == 1:
                heatkernel = restore_heatkernel_firststep(heatkernel, step, ckptdir, learner=config.heat_learner)
            elif step > 0: heatkernel = restore_heatkernel(heatkernel, step, ckptdir)
            if config.tag == 'synthetic':
                gradf, Hf = grad_hess_synthetic(decoder, heatkernel, z)
            else:
                gradf, Hf = grad_hess(vae, heatkernel, z)
            gradf, Hf = gradf.reshape(o,d), Hf.reshape(o,d,d)
            contrib = hess_operator(Jg, Hg, gradf, Hf)
            result += delta * contrib.reshape(D,o)
            if step % config.decomp_stride == 0:
                if step > 0:
                    results = jnp.concatenate([results, result[jnp.newaxis,:]])
                if step == 0:
                    print(test_label[j], kernel_predict(heatkernel, x))
                prediction.append(kernel_predict(heatkernel, x))
                if not config.decomp_accum:
                    result = jnp.zeros((D, o),dtype=np.float32)
        
        logging.info(f"Decomposing finished for sample {j}")
        np.save(savedir / f'decomp_{j}.npy', results)
        np.save(savedir / f'pred_{j}.npy', np.array(prediction))

        compare = config.compare[test_label[j]] if config.compare is not None else np.arange(o)
        decomp = laplacian_postprocess(results, config, compare)
        decomp_all.append(decomp)

    return np.stack(decomp_all) # Batch, Outputs, Time, W, W, 3
