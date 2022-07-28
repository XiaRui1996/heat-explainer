import ml_collections as mlc
from ml_collections import config_flags
from flax.training.train_state import TrainState
from flax.training import checkpoints
from pathlib import Path
import numpy as np
from functools import partial
from absl import logging
from absl import app
from absl import flags
from jax import lax, random, vmap, jacfwd, hessian, numpy as jnp
import jax
import optax
import torch
import time

from heat_explainer.train.predict import restore_checkpoints as predict_restore
from heat_explainer.train.vae import restore_checkpoints as vae_restore
from heat_explainer.trainutils.utils import save_image, warmup_cos_decay_lr_schedule_fn, to_heatmap
from heat_explainer.models.base import l2_loss
from heat_explainer.datautils.dataloader import get_dataset
from heat_explainer.saliency.method import vanilla_smooth_gradient, integrated_gradient_blur
from heat_explainer.datautils.dataloader import SyntheticDataset



class immersion(object):
    def __init__(self, decoder):
        self.decoder = decoder

    @partial(jax.jit, static_argnums=(0,))
    def metric_tensor(self, z):
        N,d = z.shape
        Jg = (vmap(jacfwd(self.decoder))(z)).reshape(N,-1,d)
        M = vmap(jnp.dot)(jnp.transpose(Jg, (0,2,1)), Jg)
        return M

    @partial(jax.jit, static_argnums=(0,3), static_argnames=('num_steps',))
    def random_walk(self, prng, z, scale=1.0, step_size=0.01, num_steps=10):
        def body(carry, _):
            z_, key = carry
            key, subkey = random.split(key)
            noise = random.normal(subkey, z_.shape)
            M = self.metric_tensor(z_)*scale
            L,U = vmap(jnp.linalg.eigh)(M)
            coef = vmap(jnp.dot)(U, 1./jnp.sqrt(L) *noise)
            zt = z_ + step_size * coef
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

def classification_loss(output, label):
    label = jax.nn.softmax(label)
    output = jax.nn.log_softmax(output)
    return -jnp.mean(jnp.sum(output * label, axis=-1))


@jax.jit
def train_step(state, predictor, x_start, x_step):
    def loss_fn(params):
        output = state.apply_fn({'params': params}, x_start)
        label = predictor.apply_fn({'params': predictor.params}, x_step)
        loss = classification_loss(output, label)
        return loss, (label, output)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    loss = aux[0]
    label, output = aux[1]
    return state, loss, label, output

def solve_heat_kernel(config: mlc.ConfigDict, workdir: str):
    vaedir = Path(workdir) / 'vae'
    predictordir = Path(workdir) / 'predictor'
    ckptdir = Path(workdir)  / 'heatkernel2'
    imagedir = Path(workdir)  / 'heatkernel2' / 'images'
    ckptdir.mkdir(parents=True, exist_ok=True)
    imagedir.mkdir(parents=True, exist_ok=True)

    key = random.PRNGKey(14)

    encode_z = np.load(vaedir / 'encode_z.npy')
    n, d, b, N = *encode_z.shape, config.metric_batch, config.N_train
    times = config.numsteps // config.stride
    Z_start = encode_z[random.randint(key, (N,), 0, n)]

    predictor = predict_restore(config, predictordir)
    if config.tag == 'synthetic':
        s = SyntheticDataset(config.image_size, 6, 60000, config.datadir, '/train', generate=False)
        decoder = s.decode_jax
    else:
        vae = vae_restore(config, vaedir)
        @jax.jit
        def decoder(z): return vae.apply_fn({'params':vae.params}, z, key, mode='decode')
    manifold = immersion(decoder)

    scale = -np.inf
    for j in range(0, n, b):
        _, logdet = jnp.linalg.slogdet(manifold.metric_tensor(encode_z[j:j+b]))
        scale = max(scale, logdet.max())
    scale = np.exp(-scale / config.vae_d)
    logging.info("Rescaling embedded manifold by {:.3f}".format(scale))


    logging.info("Generating random walk samples")
    key, subkey = random.split(key)
    z_sample = encode_z[np.random.choice(n, config.generate_rw_size)]
    z_steps = manifold.random_walk(subkey, z_sample, 
                                   step_size = config.stepsize, 
                                   num_steps = config.numsteps,
                                   scale = scale)[::config.generate_rw_stride]
    z_steps = jnp.transpose(z_steps,(1,0,2)).reshape(-1, d)
    x_steps = decoder(z_steps)
    save_image(x_steps, imagedir / 'generate_random_walk.png', 
               nrow=config.numsteps // config.generate_rw_stride)

    Z_prev = Z_start
    heatkernel = predict_restore(config, predictordir)
    for step in range(1, times+1):
        base_learning_rate = config.learning_rate * config.batch_size / 256.
        steps_per_epoch = N // b
        learning_rate_fn = warmup_cos_decay_lr_schedule_fn(base_learning_rate, 
                                                       config.heat_epochs, 1, 
                                                       steps_per_epoch)
        tx = optax.adamw(learning_rate=learning_rate_fn, weight_decay=config.weight_decay)
        heatkernel = TrainState.create(apply_fn=heatkernel.apply_fn, 
                                       params=heatkernel.params, tx=tx)

        Z_step = []
        for i, batch_idx in enumerate(range(0, N, b)):
            key, subkey = random.split(key)
            indices = np.arange(N)[batch_idx : batch_idx+b]
            z_step = manifold.random_walk(subkey, Z_prev[indices, :],
                                          scale = scale,
                                          step_size = config.stepsize,
                                          num_steps = config.stride)[-1]
            z_start = Z_start[indices, :]
            Z_step.append(z_step)

            x_step, x_start = decoder(z_step), decoder(z_start)
            heatkernel, loss, label, output = train_step(heatkernel, predictor, x_start, x_step)
            if i % config.log_every_steps == 0:
                logging.info('Train Epoch: {} [\t{}/{} ({:.0f}%)] \tMSE: {:.9f}'.format(
                                0, i * b, N,
                                100. * i * b / N, loss))

        Z_step = jnp.concatenate(Z_step)
        for epoch in range(1, config.heat_epochs):
            permutation = np.random.permutation(N)
            for i,batch_idx in enumerate(range(0,N,b)):
                indices = permutation[batch_idx : batch_idx + b]
                z_step, z_start = Z_step[indices,:], Z_start[indices,:]
                x_step, x_start = decoder(z_step), decoder(z_start)
                heatkernel, loss, label, output = train_step(heatkernel, predictor, x_start, x_step)

                if i % config.log_every_steps == 0:
                    logging.info('Train Epoch: {} [\t{}/{} ({:.0f}%)] \tMSE: {:.9f}'.format(
                                epoch, i * b, N,
                                100. * i * b / N, loss))
        checkpoints.save_checkpoint(ckptdir, 
                                    target=heatkernel, 
                                    step=step, 
                                    prefix='checkpoint_', keep=times, overwrite=True)
        Z_prev = Z_step



@jax.jit
def hess_operator(Jg, Hg, gradf, Hf):
    Ginv = jnp.linalg.inv(Jg.T.dot(Jg)) #d,d
    JgGinv = Jg.dot(Ginv) #D,d
    HJGinvf = jnp.dot(Hg.T, JgGinv.dot(gradf.T))  #d*d,o
    res = Hf - jnp.reshape(HJGinvf.T, Hf.shape) #o,d,d
    res = jnp.transpose(res, (1,2,0)) #d,d,o
    result = vmap(jnp.dot)(JgGinv, JgGinv.dot(res)) #D,o
    return result

KEY = random.PRNGKey(14)
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
    savedir = Path(workdir) / 'results5'
    savedir.mkdir(parents=True, exist_ok=True)

    ckptdir = Path(workdir)  / 'heatkernel2'
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
    
    manifold = immersion(decoder)

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

    
def test(config, workdir):

    _, testset = get_dataset(config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=True, num_workers=8, drop_last=True)
    test_batch, test_label = next(iter(testloader))
    test_batch = config.transform((test_batch.detach().numpy()).astype(jnp.float32))
    test_label = config.transform_target(test_label.detach().numpy())

    savedir = Path(workdir) / 'results4'
    
    decomp_all = laplacian_decompose(test_batch, test_label, config, workdir)
    b, r, t = decomp_all.shape[:-3]
    for j in range(b):
        original = test_batch[j]
        if config.channels==1: original = jnp.tile(original, (1,1,3))
        if r>1: 
            padding = jnp.ones((r-1, config.image_size, config.image_size, 3))
            original = jnp.concatenate([original[jnp.newaxis, ...], padding])
        save = jnp.concatenate([original[:, jnp.newaxis, ...], decomp_all[j]], axis=1)

        save_image(save.reshape(-1, config.image_size, config.image_size, 3), 
                   savedir / f'heat_decomp_visualize_{j}_label_{test_label[j]}.png', nrow = t+1)


def test_compare(config, workdir):

    _, testset = get_dataset(config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=True, num_workers=8, drop_last=True)
    test_batch, test_label = next(iter(testloader))
    test_batch = config.transform((test_batch.detach().numpy()).astype(jnp.float32))
    test_label = config.transform_target(test_label.detach().numpy())

    predictor = predict_restore(config, Path(workdir) / 'predictor')
    savedir = Path(workdir) / 'results5'
    
    decomp_all = laplacian_decompose(test_batch, test_label, config, workdir)
    logging.info("Vanilla and smooth gradient at different scales")
    grad_sg = vanilla_smooth_gradient(predictor, test_batch, test_label, config)
    logging.info("Integrated gradient and different scales of blur")
    ig_blur = integrated_gradient_blur(predictor, test_batch, test_label, config)
    print(decomp_all.shape, grad_sg.shape, ig_blur.shape)

    for j in range(decomp_all.shape[0]):
        which = 0 if config.compare is not None else test_label[j] if config.classification else 0
        original, laplace, sg, blur = test_batch[j], decomp_all[j,which,:grad_sg.shape[1]-1,...], grad_sg[j], ig_blur[j]
        if config.channels == 1: original = jnp.tile(original, (1,1,3))
        laplace = jnp.concatenate([original[jnp.newaxis,...], laplace])

        save = jnp.concatenate([laplace, sg, blur], axis=1)
        save_image(save, 
                   savedir / f'heat_sg_blur_compare_{j}.png', nrow = decomp_all.shape[2]+1)

        
def main(argv):
    
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    #solve_heat_kernel(FLAGS.config, FLAGS.workdir)
    test(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('workdir', None, 'Directory to store model data.')
    config_flags.DEFINE_config_file(
      'config', None, 'File path to the training hyperparameter configuration.',
      lock_config=True)
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)       
            

