"""Default Hyperparameter configuration."""
from pathlib import Path
import ml_collections
from jax import numpy as jnp




def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.tag = 'synthetic'
    config.image_size = 128
    config.channels = 1
    config.transform_loader = None
    config.datadir = '../../data/synthetic'
    config.classification = False

    config.learning_rate = 0.0001
    config.weight_decay = 0.00001
    config.warmup_epochs = 2.0
    config.batch_size = 32
    config.num_epochs = 20
    config.log_every_steps = 100

    config.transform = lambda x: x/255.
    config.transform_target = lambda y: y.reshape(-1,1)

    config.vae_epochs = 20
    config.vae_d = 6
    config.vae_constant_epochs = 2
    config.vae_batch = 64

    config.metric_batch = 64
    config.N_train = 100000
    config.numsteps = 40000
    config.stepsize = 0.1
    config.stride = 200
    config.generate_rw_size = 10
    config.generate_rw_stride = 500
    config.heat_epochs = 5
    config.heat_learner = 'adamw'

    config.test_batch = 100
    config.num_classes = 1
    config.decomp_stride = 20
    config.decomp_accum = True
    config.compare = None

    config.sg_scales = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]
    config.ig_steps = 100
    config.ig_partial = 12
    

    return config
