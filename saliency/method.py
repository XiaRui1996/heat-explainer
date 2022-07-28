import jax
import torch
from jax import vmap, jacrev, numpy as jnp
import saliency.core as saliency
from pathlib import Path
from absl import logging
from absl import app
from absl import flags
from ml_collections import config_flags
from tqdm import tqdm

from heat_explainer.train.predict import restore_checkpoints
from heat_explainer.trainutils.utils import to_heatmap, save_image
from heat_explainer.datautils.dataloader import get_dataset

import math

from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import numpy as np
from scipy import ndimage

TARGET_IDX_STR = 'target_idx_str'



def gaussian_blur(image, sigma):
  """Returns Gaussian blur filtered 3d (WxHxC) image.
  Args:
    image: 3 dimensional ndarray / input image (W x H x C).
    sigma: Standard deviation for Gaussian blur kernel.
  """
  if sigma == 0:
    return image
  return ndimage.gaussian_filter(
      image, sigma=[sigma, sigma, 0], mode="constant")

class BlurIG(CoreSaliency):
  """Copy from PAIR-code/saliency. Modify to return partial blur IG.
  A CoreSaliency class that implements integrated gradients along blur path.
  https://arxiv.org/abs/2004.03383
  Generates a saliency mask by computing integrated gradients for a given input
  and prediction label using a path that successively blurs the image.
  """

  expected_keys = [INPUT_OUTPUT_GRADIENTS]

  def GetMask(self,
              x_value,
              call_model_function,
              call_model_args=None,
              max_sigma=50,
              steps=100,
              grad_step=0.01,
              sqrt=False,
              batch_size=1):
    """Returns an integrated gradients mask on a blur path.
    Args:
      x_value: Input ndarray.
      call_model_function: A function that interfaces with a model to return
        specific data in a dictionary when given an input and other arguments.
        Expected function signature:
        - call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - Other arguments used to call and run the model.
          expected_keys - List of keys that are expected in the output. For this
            method (Blur IG), the expected keys are
            INPUT_OUTPUT_GRADIENTS - Gradients of the output being
              explained (the logit/softmax value) with respect to the input.
              Shape should be the same shape as x_value_batch.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      max_sigma: Maximum size of the gaussian blur kernel.
      steps: Number of successive blur applications between x and fully blurred
        image (with kernel max_sigma).
      grad_step: Gaussian gradient step size.
      sqrt: Chooses square root when deciding spacing between sigma. (Full
        mathematical implication remains to be understood).
      batch_size: Maximum number of x inputs (steps along the integration path)
        that are passed to call_model_function as a batch.
    """

    if sqrt:
      sigmas = [math.sqrt(float(i)*max_sigma/float(steps)
                          ) for i in range(0, steps+1)]
    else:
      sigmas = [float(i)*max_sigma/float(steps) for i in range(0, steps+1)]
    step_vector_diff = [sigmas[i+1] - sigmas[i] for i in range(0, steps)]

    gradients_partial = []
    total_gradients = np.zeros_like(x_value)
    x_step_batched = []
    gaussian_gradient_batched = []
    for i in range(steps):
      x_step = gaussian_blur(x_value, sigmas[i])
      gaussian_gradient = (gaussian_blur(x_value, sigmas[i] + grad_step)
                           - x_step) / grad_step
      x_step_batched.append(x_step)
      gaussian_gradient_batched.append(gaussian_gradient)
      if len(x_step_batched) == batch_size or i == steps - 1:
        x_step_batched = np.asarray(x_step_batched)
        call_model_output = call_model_function(
            x_step_batched,
            call_model_args=call_model_args,
            expected_keys=self.expected_keys)
        self.format_and_check_call_model_output(call_model_output,
                                                x_step_batched.shape,
                                                self.expected_keys)

        tmp = (
            step_vector_diff[i] *
            np.multiply(gaussian_gradient_batched,
                        call_model_output[INPUT_OUTPUT_GRADIENTS]))
        total_gradients += tmp.sum(axis=0)
        gradients_partial.append(total_gradients*(-1.0))
        x_step_batched = []
        gaussian_gradient_batched = []
        
    return np.stack(gradients_partial)

@jax.jit
def gradients_model(state, x, idx):
    def model_idx(x):
        output = state.apply_fn( {'params': state.params}, x)
        return output[..., idx]
    gradients = vmap(jacrev(model_idx))(x[:, jnp.newaxis, ...])
    return jnp.reshape(gradients, x.shape)

def call_method(predictor):
    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_idx = call_model_args[TARGET_IDX_STR]
        gradients = gradients_model(predictor, images, target_idx)
        if expected_keys == [saliency.base.INPUT_OUTPUT_GRADIENTS]:
          return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    return call_model_function

def vanilla_smooth_gradient(predictor, test_batch, test_label, config):

    call_model_function = call_method(predictor)
    gradient_saliency = saliency.GradientSaliency()
    scales = config.sg_scales
    call_model_args = {TARGET_IDX_STR: 0}

    results = []
    for j in tqdm(range(test_batch.shape[0])):
        if config.classification:
            call_model_args = {TARGET_IDX_STR: test_label[j]}

        vanilla_mask = gradient_saliency.GetMask(test_batch[j], call_model_function, 
                                                              call_model_args)
        smooth_grad_scales = []
        for i, scale in enumerate(scales):
            sg_mask = gradient_saliency.GetSmoothedMask(test_batch[j], 
                                                    call_model_function, 
                                                    call_model_args, 
                                                    stdev_spread=scale, 
                                                    nsamples=min(25,(i+1)*10))
            smooth_grad_scales.append(sg_mask)

        smooth_grad_scales = jnp.stack(smooth_grad_scales) #t, W, W, C
        smooth_grad_scales = to_heatmap(smooth_grad_scales.reshape(-1,
                                  config.image_size, config.channels)).reshape(-1, 
                                  config.image_size, config.image_size, 3)
        vanilla_mask = to_heatmap(vanilla_mask)
        vanilla_sg = jnp.concatenate([vanilla_mask[np.newaxis,...], smooth_grad_scales])
        results.append(vanilla_sg)

    return jnp.stack(results) #B, T+1, W, W, C


def integrated_gradient_blur(predictor, test_batch, test_label, config):

    call_model_function = call_method(predictor)
    integrated_gradients = saliency.IntegratedGradients()
    blur_ig = BlurIG()
    baseline = np.zeros(test_batch.shape[1:])
    call_model_args = {TARGET_IDX_STR: 0}

    steps = config.ig_steps
    partial_size = config.ig_partial

    results = []
    for j in tqdm(range(test_batch.shape[0])):
        if config.classification:
            call_model_args = {TARGET_IDX_STR: test_label[j]}
          
        vanilla_ig_mask = integrated_gradients.GetMask(test_batch[j], 
                                                     call_model_function, 
                                                     call_model_args, 
                                                     x_steps=25, 
                                                     x_baseline=baseline, 
                                                     batch_size=20)
        blur_ig_mask = blur_ig_mask = blur_ig.GetMask(test_batch[j],
                                                    call_model_function, 
                                                    call_model_args, 
                                                    batch_size=partial_size, 
                                                    steps=steps)
        vanilla_ig_mask = to_heatmap(vanilla_ig_mask) #w,w,c
        blur_ig_mask = to_heatmap(blur_ig_mask.reshape(-1,
                            config.image_size, config.channels)).reshape(-1,
                            config.image_size, config.image_size, 3) #t,w,w,c
        ig_blur = jnp.concatenate([vanilla_ig_mask[np.newaxis,...], blur_ig_mask])
        results.append(ig_blur)
    
    return jnp.stack(results) #B, T+1, W, W, C

def test(config, workdir):
    savedir = Path(workdir) / 'saliency'
    predictordir = Path(workdir) / 'predictor'

    _, testset = get_dataset(config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=True, num_workers=8, drop_last=True)
    test_batch, test_label = next(iter(testloader))
    test_batch = config.transform((test_batch.detach().numpy()).astype(jnp.float32))
    test_label = config.transform_target(test_label.detach().numpy())


    predictor = restore_checkpoints(config, predictordir)

    grad_sg = vanilla_smooth_gradient(predictor, test_batch, test_label, config)
    ig_blur = integrated_gradient_blur(predictor, test_batch, test_label, config)
    save_image(grad_sg.reshape(-1,config.image_size, config.image_size,3),
               savedir / 'gradient_smooth.png', nrow = grad_sg.shape[1])
    save_image(ig_blur.reshape(-1,config.image_size, config.image_size,3),
               savedir / 'integrated_gradient_blur.png', nrow = ig_blur.shape[1])


def main(argv):

    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    test(FLAGS.config, FLAGS.workdir)

if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('workdir', None, 'Directory to store model data.')
    config_flags.DEFINE_config_file(
      'config', None, 'File path to the training hyperparameter configuration.',
      lock_config=True)
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)  


    






