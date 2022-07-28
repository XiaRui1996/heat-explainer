import torch
import optax
import jax
from jax import random, device_put, numpy as jnp
from functools import partial
from typing import Any
from pathlib import Path
from flax import linen as nn
from flax.training import checkpoints
from flax.training.train_state import TrainState
import numpy as np
from absl import app
from absl import flags
from absl import logging
import jax
import ml_collections as mlc
from ml_collections import config_flags

from heat_explainer.models.generative import VAEflax, VAElinear, kl_divergence, binary_cross_entropy
from heat_explainer.datautils.dataloader import get_dataset
from heat_explainer.trainutils.utils import save_image


MODEL = {'synthetic': partial(VAEflax, H=32, inner=4, layer_count=5, channels=3, first=True),
         'mnist': partial(VAElinear, H=128, act=nn.softplus)}




def vae_loss(x, x_recon, z_mean, z_logvar):
    print(x.shape, x_recon.shape)
    #print(binary_cross_entropy(x_recon, x).shape)
    err = jnp.sum(jnp.mean((x_recon-x)**2,axis=0)) #binary_cross_entropy(x_recon, x).mean()
    kld = kl_divergence(z_mean, z_logvar).mean()
    return err, kld

@partial(jax.jit, static_argnums=(3,4))
def train_step(state, x, rng, anneal_fn, learning_rate_fn):
    step = state.step
    anneal = anneal_fn(step)
    def loss_fn(params):
        x_recon, mean, logvar = state.apply_fn({'params':params}, x, rng)
        err, kld = vae_loss(x, x_recon, mean, logvar)
        loss = err + 1.0 * kld
        return loss, (err, kld)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    loss_val = aux[0]
    err, kld = aux[1]
    metrics = {'loss': loss_val, 'err': err, 'kld': kld}
    metrics['anneal'] = anneal
    metrics['lr'] = learning_rate_fn(step)

    return state, metrics

@jax.jit
def eval(state, x, rng):
    x_recon, mean, logvar = state.apply_fn({'params':state.params}, x, rng, training=False)
    err, kld = vae_loss(x, x_recon, mean, logvar)
    return {'err': err, 'kld': kld}

@jax.jit
def encoder(state, batch, rng):
    return state.apply_fn({'params':state.params}, batch, rng, mode='encode')

def create_train_state(rng, config: mlc.ConfigDict, 
                       model, n_train=10000, return_lr=False):
    rng, key = random.split(rng)
    input_shape = (1, config.image_size, config.image_size, config.channels)
    init_data = jnp.ones(input_shape, dtype=model.dtype)
    steps_per_epoch = n_train // config.batch_size
    base_learning_rate = config.learning_rate * config.batch_size / 256.

    cosine_fn = optax.cosine_decay_schedule(
                            init_value = base_learning_rate,
                            decay_steps= config.vae_epochs * steps_per_epoch)
    tx = optax.adamw(learning_rate=cosine_fn,weight_decay=config.weight_decay)
    @jax.jit
    def init(*args):
        return model.init(*args)
    state = TrainState.create(apply_fn=model.apply,
                              params=init(key, init_data, rng)['params'],
                              tx=tx)
    if return_lr: return state, cosine_fn
    else: return state

def restore_checkpoints(config, modeldir):
    rng = random.PRNGKey(14)
    model = MODEL[config.tag](d = config.vae_d, input_shape = (config.image_size, config.image_size, config.channels))
    state = create_train_state(rng, config, model)
    state = checkpoints.restore_checkpoint(ckpt_dir=modeldir, target=state)
    return state


def train_VAE_flax(config: mlc.ConfigDict,
                   workdir: str) -> TrainState:

    ckptdir = Path(workdir) / 'vae'
    ckptdir.mkdir(parents=True, exist_ok=True)

    trainset, testset = get_dataset(config)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.vae_batch, shuffle=True, num_workers=8, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.vae_batch, shuffle=True, num_workers=8, drop_last=True)
    
    rng = random.PRNGKey(14)

    model = MODEL[config.tag](d = config.vae_d, input_shape = (config.image_size, config.image_size, config.channels))

    state, learning_rate_fn = create_train_state(rng, config, model, len(trainset), return_lr=True)
    steps_per_epoch = len(trainset) // config.batch_size
    kl_anneal_fn = optax.linear_schedule(init_value=0.5,end_value=1., 
                                         transition_steps=(config.vae_epochs+config.vae_constant_epochs)*steps_per_epoch,
                                         transition_begin=config.vae_constant_epochs*steps_per_epoch)

    ELBO = np.zeros((config.vae_epochs, 1))

    for epoch in range(config.vae_epochs):
        # Initialize the losses
        train_loss = 0
        train_loss_num = 0
        
        key = random.PRNGKey(epoch + 42)
        # Train for all the batches
        for batch_idx, (x_batch,_) in enumerate(trainloader):
            key, subkey = random.split(key)
            x_batch = device_put((x_batch.detach().numpy()).astype(jnp.float32))
            x_batch = config.transform(x_batch)
            state, metrics  = train_step(state, x_batch, subkey, kl_anneal_fn, learning_rate_fn)
            if batch_idx % config.log_every_steps == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} \tREC err: {:.2f} \tKLD: {:.2f}, \tCoeff: {:.3f}'.format(
                        epoch, batch_idx * len(x_batch), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), metrics['loss'], metrics['err'], metrics['kld'], metrics['anneal']))
            
            train_loss += metrics['loss']
            train_loss_num += 1
        
        test_err, test_kld = 0,0
        for test_batch,_ in testloader:
            key, subkey = random.split(key)
            test_batch = device_put((test_batch.detach().numpy()).astype(jnp.float32))
            test_batch = config.transform(test_batch)

            loss_metrics = eval(state, test_batch, subkey)
            test_err += loss_metrics['err']
            test_kld += loss_metrics['kld']
        
        ELBO[epoch] = train_loss / train_loss_num
        logging.info("[Epoch: {}/{}] [objective: {:.3f}, test err: {:.3f} test kld: {:.3f}]".format(
            epoch, config.vae_epochs, ELBO[epoch, 0], test_err, test_kld))
        checkpoints.save_checkpoint(ckptdir, target=state, step=epoch, prefix='checkpoint_', keep=10, overwrite=True)
    

    logging.info("Start encoding all training data")
    trainloader_nosf = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                    shuffle=False, num_workers=8, drop_last=False)
    encode_z = []
    for i, (x_batch, _) in enumerate(trainloader_nosf):
        x_batch = device_put((x_batch.detach().numpy()).astype(jnp.float32))
        x_batch = config.transform(x_batch)

        z_batch, _  = encoder(state, x_batch, key)
        if i==0:
            rec_batch, _, _ = state.apply_fn({'params':state.params}, x_batch, key, training=False)
            batch_save = jnp.stack([x_batch, rec_batch], axis=1).reshape(-1, *x_batch.shape[1:])
            print(batch_save.shape)
            save_image(batch_save, ckptdir / 'reconstruct_batch.png', nrow=2)
        encode_z.append(z_batch)
    encode_z = jnp.concatenate(encode_z)
    np.save(ckptdir / 'encode_z.npy', encode_z)
    logging.info("Saved encoded latent to {:s}".format(str(ckptdir / 'encode_z.npy')))

    return state, encode_z


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  state, encode_z = train_VAE_flax(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  FLAGS = flags.FLAGS

  flags.DEFINE_string('workdir', None, 'Directory to store model data.')
  config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
