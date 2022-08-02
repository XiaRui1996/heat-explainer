from this import d
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
from heat_explainer.train.immersion import Immersion
from heat_explainer.trainutils.utils import save_image, warmup_cos_decay_lr_schedule_fn, to_heatmap
from heat_explainer.models.base import l2_loss, classification_loss
from heat_explainer.models.regressor import CondImageRegressor
from heat_explainer.datautils.dataloader import get_dataset
from heat_explainer.datautils.dataloader import SyntheticDataset, SyntheticYFDataset
from heat_explainer.train.decompose import laplacian_decompose



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
    ckptdir = Path(workdir)  / 'heatkernel'
    imagedir = Path(workdir)  / 'heatkernel' / 'images'
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
    manifold = Immersion(decoder)

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


def brownian(rng, x, t):
    t = jnp.asarray(t, dtype=x.dtype)
    prefix = jnp.broadcast_shapes(t.shape, x.shape[:-3])
    noise = jax.random.normal(rng, prefix + x.shape[-3:])
    xt = x + noise * jnp.expand_dims(t, axis=(-3, -2, -1))
    return xt

def solve_heat_euclidean(config: mlc.ConfigDict, workdir: str):
    predictordir = Path(workdir) / 'predictor'
    ckptdir = Path(workdir)  / 'heatkernel'
    imagedir = Path(workdir)  / 'heatkernel' / 'images'
    ckptdir.mkdir(parents=True, exist_ok=True)
    imagedir.mkdir(parents=True, exist_ok=True)

    key = random.PRNGKey(14)

    trainset, testset = get_dataset(config)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.metric_batch, shuffle=True, num_workers=8, drop_last=True)

    b, N = config.metric_batch, config.N_train
    times = config.numsteps // config.stride

    predictor = predict_restore(config, predictordir)
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

        for epoch in range(config.heat_epochs):
            for i, (x_start,_) in enumerate(trainloader):
                key, subkey = random.split(key)
                x_start = (x_start.detach().numpy()).astype(jnp.float32)
                x_start = config.transform(x_start)
                
                t = np.ones(size=(b,)) * config.stepsize * step
                x_step = brownian(subkey, x_start, t)
                heatkernel, loss, label, output = train_step(heatkernel, predictor, x_start, x_step)

                if i % config.log_every_steps == 0:
                    logging.info('Train Epoch: {} [\t{}/{} ({:.0f}%)] \tMSE: {:.9f}'.format(
                                epoch, i * b, N,
                                100. * i * b / N, loss))

        checkpoints.save_checkpoint(ckptdir, 
                                    target=heatkernel, 
                                    step=step, 
                                    prefix='checkpoint_', keep=times, overwrite=True)
    
def test(config, workdir):

    _, testset = get_dataset(config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=True, num_workers=8, drop_last=True)
    test_batch, test_label = next(iter(testloader))
    test_batch = config.transform((test_batch.detach().numpy()).astype(jnp.float32))
    test_label = config.transform_target(test_label.detach().numpy())

    savedir = Path(workdir) / 'results'
    
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
            

