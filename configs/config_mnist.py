"""Default Hyperparameter configuration."""
from pathlib import Path
import ml_collections
from jax import numpy as jnp
import torchvision.transforms as transforms




def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.tag = 'mnist'
    config.image_size = 28
    config.channels = 1
    config.transform_loader = transforms.ToTensor()
    config.datadir = '../../data/'
    config.classification = True

    config.learning_rate = 0.0001
    config.weight_decay = 0.00001
    config.warmup_epochs = 5.0
    config.batch_size = 32
    config.num_epochs = 5
    config.log_every_steps = 100

    config.transform = lambda x: jnp.transpose(x, (0,2,3,1))
    config.transform_target = lambda y: y

    config.vae_epochs = 40
    config.vae_d = 10
    config.vae_constant_epochs = 0
    config.vae_batch = 128 

    config.metric_batch = 256
    config.N_train = 200000
    config.numsteps = 2000
    config.stepsize = 0.02
    config.stride = 10
    config.generate_rw_size = 20
    config.generate_rw_stride = 100
    config.heat_epochs = 1
    config.heat_learner = 'adamw'

    config.test_batch = 16
    config.num_classes = 10
    config.decomp_stride = 20
    config.decomp_accum = False
    config.compare = [[0,6,9], [1,2,7], [2,3,7],\
                      [3,5,8], [4,7,9], [5,3,8],\
                      [6,0,8], [7,1,9], [8,3,5], [9,4,7]]

    config.sg_scales = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.4,0.45,0.5]
    config.ig_steps = 100
    config.ig_partial = 10


    return config
