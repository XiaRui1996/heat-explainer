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
from ml_collections import config_flags

from heat_explainer.models.base import l2_loss, mae_loss, cross_entropy_loss, accuracy, ResNet18
from heat_explainer.trainutils.utils import warmup_cos_decay_lr_schedule_fn
from heat_explainer.datautils.dataloader import get_dataset

LOSS = {'synthetic':{'mse':l2_loss,'mae':mae_loss}, 
        'mnist':{'ce':cross_entropy_loss, 'acc':accuracy}}
MODEL = {'mnist': ResNet18(act=nn.relu, num_classes=10, num_filters=32),
         'synthetic': ResNet18(act=nn.tanh, num_classes=1, num_filters=32)}



def compute_metrics(logits, labels, methods):
    metrics = {}
    for name in methods:
        metrics[name] = methods[name](logits, labels)
    return metrics

def process(x, transform):
    x = device_put((x.detach().numpy()).astype(np.float32))
    x = transform(x) 
    return x



@partial(jax.jit, static_argnums=(2,3))
def train_step(state, batch, learning_rate_fn, tag):
    loss_methods = LOSS[tag]
    loss_fun = loss_methods[list(loss_methods.keys())[0]]

    def loss_fn(params):
        outputs = state.apply_fn({'params': params},
                                batch['image'])
        loss = loss_fun(outputs, batch['label'])
        return loss, outputs

    step = state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    loss, outputs = aux
    metrics = compute_metrics(outputs, batch['label'], loss_methods)
    metrics['learning_rate'] = lr
    metrics['loss'] = loss

    state = state.apply_gradients(grads=grads)
    return state, metrics

@partial(jax.jit, static_argnums=(2,))
def eval(state, batch, tag):
    loss_methods = LOSS[tag]
    outputs = state.apply_fn({'params': state.params}, batch['image'])
    metrics = compute_metrics(outputs, batch['label'], loss_methods)
    return metrics

def create_state(config, rng, model, n_train=10000, return_lr=False):
    base_learning_rate = config.learning_rate * config.batch_size / 256.
    steps_per_epoch = n_train // config.batch_size
    learning_rate_fn = warmup_cos_decay_lr_schedule_fn(base_learning_rate, 
                                                       config.num_epochs, 
                                                       config.warmup_epochs, 
                                                       steps_per_epoch)
    tx = optax.adamw(learning_rate=learning_rate_fn, weight_decay=config.weight_decay)
    input_shape = (1, config.image_size, config.image_size, config.channels)
    @jax.jit
    def init(*args):
        return model.init(*args)

    params = init({'params':rng}, jnp.ones(input_shape, model.dtype))['params']
    state = TrainState.create(apply_fn=model.apply,params=params,tx=tx)
    if return_lr: return state, learning_rate_fn
    else: return state


def restore_checkpoints(config,  modeldir: str, step: Any = None) -> TrainState:
    rng = random.PRNGKey(0)
    model = MODEL[config.tag]
    state = create_state(config, rng, model)
    state = checkpoints.restore_checkpoint(ckpt_dir=modeldir, target=state, step=step)
    return state


def train_and_evaluate(config, workdir) -> TrainState:
    ckptdir = Path(workdir) / 'predictor'
    ckptdir.mkdir(parents=True, exist_ok=True)

    trainset, testset = get_dataset(config)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)

    rng = random.PRNGKey(0)
    model = MODEL[config.tag]
    state, learning_rate_fn = create_state(config, rng, model, n_train=len(trainset), return_lr=True)

    loss_methods = LOSS[config.tag]
    metric_names = list(loss_methods.keys())

    for epoch in range(config.num_epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = process(data, config.transform), process(target, config.transform_target)
            batch = {'image':data, 'label':target}
            state, metrics = train_step(state, batch, learning_rate_fn, config.tag)
            
            if batch_idx % config.log_every_steps == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t{:s}: {:.6f} \t{:s}:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), 
                    metric_names[0], metrics[metric_names[0]], 
                    metric_names[1], metrics[metric_names[1]]))

        test_obj = 0
        for batch_id, (test_batch,test_target) in enumerate(testloader):
            test_batch, test_target = process(test_batch, config.transform), \
                                      process(test_target, config.transform_target)
            batch = {'image':test_batch, 'label':test_target}
            test_metrics = eval(state, batch, config.tag)
            test_obj += test_metrics[metric_names[0]]

        test_obj /= batch_id

        logging.info("[Epoch: {}/{}] [Test Loss: {:.3f}, LR: {:.6f}]".format(epoch, config.num_epochs, test_obj, metrics['learning_rate']))
        checkpoints.save_checkpoint(ckptdir, target=state, step=epoch, prefix='checkpoint_', keep=5, overwrite=True)   
    return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  train_and_evaluate(FLAGS.config_file, FLAGS.workdir)


if __name__ == '__main__':
  

  FLAGS = flags.FLAGS

  flags.DEFINE_string('workdir', None, 'Directory to store model data.')
  config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)
  flags.mark_flags_as_required(['config_file', 'workdir'])
  app.run(main)
