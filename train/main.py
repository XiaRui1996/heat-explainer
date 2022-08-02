from absl import app
from absl import flags
from absl import logging
import jax
import torch
import jax.numpy as jnp
from ml_collections import config_flags
from pathlib import Path

from heat_explainer.train import vae, predict, heat, decompose
from heat_explainer.saliency.method import vanilla_smooth_gradient, integrated_gradient_blur
from heat_explainer.trainutils.utils import save_image
from heat_explainer.train.predict import restore_checkpoints as predict_restore
from heat_explainer.datautils.dataloader import get_dataset

def test_compare(config, workdir):

    _, testset = get_dataset(config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=True, num_workers=8, drop_last=True)
    test_batch, test_label = next(iter(testloader))
    test_batch = config.transform((test_batch.detach().numpy()).astype(jnp.float32))
    test_label = config.transform_target(test_label.detach().numpy())

    predictor = predict_restore(config, Path(workdir) / 'predictor')
    savedir = Path(workdir) / 'results'
    
    decomp_all = decompose.laplacian_decompose(test_batch, test_label, config, workdir)
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
  
  if FLAGS.mode == 'train_predict':
    predict.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == 'train_vae': 
    vae.train_VAE_flax(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == 'train_heat':
    heat.solve_heat_kernel(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == 'compare':
    test_compare(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  FLAGS = flags.FLAGS

  flags.DEFINE_string('workdir', None, 'Directory to store model data.')
  flags.DEFINE_string('mode', None, 'Train predictor/vae, Solve heat kernel, Decomp and compare')
  config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)
  flags.mark_flags_as_required(['config', 'workdir','mode'])
  app.run(main)
