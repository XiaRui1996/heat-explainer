import optax
import jax.numpy as jnp
import math
import cv2
import numpy as np


def to_heatmap(img, percentile=100):
    if img.ndim >= 3 and img.shape[-1] in (1, 3):
        img = jnp.sum(img, axis=-1)
    
    span = jnp.percentile(jnp.abs(img), percentile)
    blue, red, green = [jnp.ones_like(img) for _ in range(3)]
    pos = jnp.minimum(1, 1 - img / span)
    neg = jnp.minimum(1, 1 + img / span)
    blue = jnp.where(img > 0, pos, blue)
    red = jnp.where(img < 0, neg, red)
    green = jnp.where(img > 0, pos, neg)
    return jnp.stack([blue, green, red], axis=-1)

def postprocess(img, percentile=99):
    if len(img.shape)==3: img_2d = np.sum(img, axis=-1)
    else: img_2d = img
    span = np.percentile(abs(img_2d),percentile)
    img = img_2d[:,:,None]
    blue = img.copy()
    blue[img>0]= np.minimum(1,(1-img[img>0]/span)*1)
    blue[img<0] = 1
    blue[img==0]= 1
    red = img.copy()
    red[img<0]= np.minimum( 1,(1+img[img<0]/span)*1)
    red[img>0] = 1
    red[img==0]=1
    green = img.copy()
    green[:]=1
    green[img>0]=np.minimum(1,(1-img[img>0]/span)*1)
    green[img<0]=np.minimum(1,(1+img[img<0]/span)*1)
    return np.concatenate([blue,green,red],axis=2).astype(np.float32)



def warmup_cos_decay_lr_schedule_fn(
        base_learning_rate: float,
        num_epochs: int,
        warmup_epochs: int,
        steps_per_epoch: int):
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0):
    """
    Make a grid of images and Save it into an image file.

    Args:
      ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
        fp - A filename(string) or file object
      nrow (int, optional): Number of images displayed in each row of the grid.
        The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
      padding (int, optional): amount of padding. Default: ``2``.
      scale_each (bool, optional): If ``True``, scale each image in the batch of
        images separately rather than the (min, max) over all images. Default: ``False``.
      pad_value (float, optional): Value for the padded pixels. Default: ``0``.
  """
    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError(
            'array_like of tensors expected, got {}'.format(type(ndarray)))

    if type(fp) is not str:
        fp = str(fp)

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] +
                        padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full((height * ymaps + padding, width * xmaps +
                     padding, num_channels), pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[y * height + padding:(y + 1) * height,
                           x * width + padding:(x + 1) * width].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    cv2.imwrite(fp, np.asarray(ndarr))
