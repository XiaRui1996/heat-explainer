import optax
import jax
from jax import numpy as jnp
from functools import partial
from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
ModuleDef = Any
Identity = lambda x: x


actFun = {'tanh': nn.tanh, 'relu': nn.relu, 'sigmoid': nn.sigmoid}


class GatedUnit(nn.Module):
    hid_features: int
    out_features: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        dense = partial(nn.Dense, dtype=self.dtype)

        x = dense(self.hid_features * 2)(x)
        gate, value = jnp.split(x, 2, axis=-1)
        x = nn.sigmoid(gate) * nn.tanh(value)
        return dense(self.out_features)(x)



class CondGatedUnit(nn.Module):
    hid_features: int
    out_features: int
    dtype: Any = jnp.float32
    out_init: Any = nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x, y):
        dense = partial(nn.Dense, dtype=self.dtype)

        x = dense(self.hid_features * 2)(x)
        x_gate, x_value = jnp.split(x, 2, axis=-1)

        y = dense(self.hid_features * 2)(y)
        y_gate, y_value = jnp.split(y, 2, axis=-1)

        x = nn.sigmoid(x_gate + y_gate) * nn.tanh(x_value + y_value)
        return dense(self.out_features, kernel_init=self.out_init)(x)


class SkipConnCondGatedUnit(nn.Module):
    hid_features: int
    norm: ModuleDef
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, y):
        r = CondGatedUnit(
            self.hid_features, x.shape[-1],
            out_init=nn.initializers.zeros,
            dtype=self.dtype
        )(x, y)
        return self.norm(dtype=self.dtype)(x + r)


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  finalact: Any = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = self.act(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    if self.finalact: x = self.finalact(x)
    x = jnp.asarray(x, self.dtype)
    return x

class CondResNet(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.silu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, z):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, num_groups=32)
        gated = partial(SkipConnCondGatedUnit, norm=norm, dtype=self.dtype)

        z = z[..., jnp.newaxis, jnp.newaxis, :]

        x = conv(self.num_filters, (7, 7),
                 strides=(2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='norm_init')(x)
        x = gated(x.shape[-1])(x, z)

        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides, conv=conv,
                                   act=self.act)(x)
                x = gated(x.shape[-1])(x, z)

        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)


def huber_loss(logits, labels):
  loss = optax.huber_loss(logits, labels, delta=25.)
  loss = jnp.sum(loss,axis=-1)
  return jnp.mean(loss)

def l2_loss(logits, labels):
  loss = optax.l2_loss(logits, labels)
  loss = jnp.sum(loss,axis=-1)
  return jnp.mean(loss)

def mae_loss(logits, labels):
  loss = jnp.abs(logits - labels)
  loss = jnp.sum(loss,axis=-1)
  return jnp.mean(loss)

def cross_entropy_loss(logits, labels):
  logits = nn.log_softmax(logits)
  one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
  return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))

def accuracy(logits, labels):
  logits = nn.log_softmax(logits)
  return jnp.mean(jnp.argmax(logits, -1) == labels)

def classification_loss(output, label):
    label = jax.nn.softmax(label)
    output = jax.nn.log_softmax(output)
    return -jnp.mean(jnp.sum(output * label, axis=-1))




class ImageRegressor(nn.Module):
    targets: int
    encoder_cls: ModuleDef = ResNet18
    dtype: Any = jnp.float32
    fourier_min: int = 5
    fourier_max: int = 8
    act: Callable = None

    def setup(self):
        self.encoder = self.encoder_cls(dtype=self.dtype)
        self.proj = nn.Dense(self.targets, dtype=self.dtype)

        self.fourier_freqs = 2.0 * jnp.pi * \
            (2.0 ** jnp.arange(self.fourier_min, self.fourier_max + 1, dtype=self.dtype))

    def __call__(self, x):
        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])

        ff = jnp.reshape(x[..., jnp.newaxis] * self.fourier_freqs, x.shape[:-1] + (-1,))
        x = jnp.concatenate([x, jnp.sin(ff), jnp.cos(ff)], axis=-1)

        h = self.encoder(x)
        h = jnp.mean(h, axis=(-2, -3))
        y = self.proj(h)

        y = jnp.reshape(y, prefix + y.shape[1:])

        if self.act is not None:
            y = self.act(y)

        return y

class CondImageRegressor(nn.Module):
    targets: int
    encoder_cls: ModuleDef = CondResNet18
    dtype: Any = jnp.float32
    fourier_min: int = 5
    fourier_max: int = 8
    act: Callable = None

    def setup(self):
        self.encoder = self.encoder_cls(dtype=self.dtype)
        self.gated = SkipConnCondGatedUnit(256, partial(nn.GroupNorm, 32), dtype=self.dtype)
        self.proj = nn.Dense(self.targets, dtype=self.dtype)

        self.x_fourier_freqs = 2.0 * jnp.pi * \
            (2.0 ** jnp.arange(self.fourier_min, self.fourier_max + 1, dtype=self.dtype))
        
        self.t_fourier_freqs = 2.0 * jnp.pi * \
            jax.random.normal(jax.random.PRNGKey(42), (255,)) * 16.0

    def __call__(self, x, t):
        '''
        Args:
          x: (..., H, W, C) array.
          t: (...) array.
        '''
        prefix = jnp.broadcast_shapes(t.shape, x.shape[:-3])
        t = jnp.broadcast_to(t, prefix)
        x = jnp.broadcast_to(x, prefix + x.shape[-3:])
        t = jnp.reshape(t, (-1, 1))
        x = jnp.reshape(x, (-1,) + x.shape[-3:])

        xff = jnp.reshape(x[..., jnp.newaxis] * self.x_fourier_freqs, x.shape[:-1] + (-1,))
        x = jnp.concatenate([x, jnp.sin(xff), jnp.cos(xff)], axis=-1)

        tff = jnp.reshape(t[..., jnp.newaxis] * self.t_fourier_freqs, t.shape[:-1] + (-1,))
        t = jnp.concatenate([t, jnp.exp(-t), jnp.sin(tff), jnp.cos(tff)], axis=-1)


        h = self.encoder(x, t)
        h = jnp.mean(h, axis=(-2, -3))
        h = self.gated(h, t)
        y = self.proj(h)

        y = jnp.reshape(y, prefix + y.shape[1:])

        if self.act is not None:
            y = self.act(y)

        return y

CondResNet18 = partial(CondResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
CondResNet34 = partial(CondResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)