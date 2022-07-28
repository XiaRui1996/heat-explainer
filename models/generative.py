import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import flax.linen as fnn
from jax import jvp, vjp, random, vmap, numpy as jnp
import jax
from flax.training.train_state import TrainState
from flax.training import checkpoints
from jax import device_put
import optax
import numpy as np
from typing import Any, Callable
from functools import partial


ModuleDef = Any


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

EPS = 1e-8
@jax.vmap
def binary_cross_entropy(probs, labels):
    return - jnp.sum(labels * jnp.log(probs + EPS) + (1 - labels) * jnp.log(1 - probs + EPS))


def reparameterize(rng, mu, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, std.shape)
    return eps*std + mu

class VAEflax(fnn.Module):
  H: int
  d: int
  input_shape: tuple
  inner: int
  layer_count: int
  channels: int
  first: bool
  dtype: Any = jnp.float32

  def setup(self):
    
    if self.first: 
        self.firstconv = fnn.Conv(features=self.H, 
                                      kernel_size=(2, 2), 
                                      strides=1, 
                                      name="conv%d" % (0))
    self.convlayers = [fnn.Conv(features=self.H*(2**i), 
                                      kernel_size=(4, 4), 
                                      strides=2, 
                                      padding=((1,1),(1,1)),
                                      name="conv%d" % (i + 1)) for i in range(self.layer_count)]

    inputs = self.H * (2**(self.layer_count-1))

    self.d_max = inputs
    self.fc1 = fnn.Dense(features=self.d, name='fc1')
    self.fc2 = fnn.Dense(features=self.d, name='fc2')
    self.d1 = fnn.Dense(features=inputs * self.inner * self.inner, name='d1')

    self.dconvlayers = [fnn.ConvTranspose(features=self.H*(2**(self.layer_count-i-1)), 
                                      kernel_size=(3,3) if i==3 and self.inner==3 else (4,4), 
                                      strides=(2,2),
                                      padding=((2,2),(2,2)),
                                      name="deconv%d" % (i + 1)) for i in range(1,self.layer_count)]

    if not self.first:
        self.lastdconv = fnn.ConvTranspose(features=self.channels, 
                                      kernel_size=(4, 4), 
                                      strides=(2,2),
                                      padding=((2,2),(2,2)),
                                      name="deconv%d" % (self.layer_count + 1))
    else:
        self.lastdconv = fnn.ConvTranspose(features=self.H, 
                                      kernel_size=(4, 4), 
                                      strides=(2,2),
                                      padding=((2,2),(2,2)),
                                      name="deconv%d" % (self.layer_count + 1))
        self.lastlayer = fnn.ConvTranspose(features=self.channels, 
                                      kernel_size=(2, 2), 
                                      name="deconv%d" % (self.layer_count + 2))
  @fnn.compact
  def __call__(self, inputs, rng, mode='all', training=True):
    if mode=='all':
      x = inputs
      mu, logvar = self.encode(x)
      z = mu if not training else reparameterize(rng, mu, logvar)
      return self.decode(z), mu, logvar
    elif mode=='decode':
      return self.decode(inputs)
    elif mode=='encode':
      mu, _ = self.encode(inputs)
      return mu

  def encode(self, x):
    if self.first: x = fnn.silu(self.firstconv(x))
    for i in range(self.layer_count):
      x = fnn.softplus(self.convlayers[i](x))
    x = jnp.reshape(x, (-1, self.d_max * self.inner * self.inner))
    h1 = self.fc1(x)
    h2 = self.fc2(x)
    return h1, h2

  def decode(self, x):
    x = self.d1(x)
    x = jnp.reshape(x,(-1,self.inner, self.inner, self.d_max))
    for i in range(self.layer_count-1):
      x = fnn.softplus(self.dconvlayers[i](x))
    if not self.first: x = fnn.sigmoid(self.lastdconv(x))
    else:
        x = fnn.silu(self.lastdconv(x))
        x = fnn.sigmoid(self.lastlayer(x))
    return x


class VAElinear(fnn.Module):
    d: int
    H: int 
    input_shape: tuple
    act: Callable
    dtype: Any = jnp.float32

    @fnn.compact
    def __call__(self, inputs, rng, mode='all', training=True):
        if mode=='all':
            x = inputs
            mu, logvar = self.encode(x)
            z = mu if not training else reparameterize(rng, mu, logvar)
            return self.decode(z), mu, logvar
        elif mode == 'encode':
            return self.encode(inputs)
        elif mode == 'decode':
            return self.decode(inputs)

    def encode(self, x):
        prefix = x.shape[:-3]
        x = jnp.reshape(x, prefix + (-1,))
        x = fnn.Dense(features=2*self.H, name='encode_fc1')(x)
        x = self.act(x)
        x = fnn.Dense(features=self.H, name='encode_fc2')(x)
        x = self.act(x)
        mean = fnn.Dense(features=self.d, name='encode_mean')(x)
        logvar = fnn.Dense(features=self.d, name='encode_var')(x)
        return mean, logvar

    def decode(self, z):
        prefix = z.shape[:-1]
        z = fnn.Dense(features=self.H, name='decode_fc1')(z)
        z = self.act(z)
        z = fnn.Dense(features=2*self.H, name='decode_fc2')(z)
        z = self.act(z)
        z = fnn.Dense(features=np.prod(self.input_shape), name='decode_out')(z)
        z = jax.nn.sigmoid(z)
        z = jnp.reshape(z, prefix + self.input_shape)
        return z


