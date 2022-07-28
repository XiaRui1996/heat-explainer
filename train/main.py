from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags

from heat_explainer.train import vae, predict, heat




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
    heat.test_compare(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  FLAGS = flags.FLAGS

  flags.DEFINE_string('workdir', None, 'Directory to store model data.')
  flags.DEFINE_string('mode', None, 'Train predictor/vae, Solve heat kernel, Decomp and compare')
  config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)
  flags.mark_flags_as_required(['config', 'workdir','mode'])
  app.run(main)
