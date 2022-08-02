export CUDA_VISIBLE_DEVICES=1

# train regreesion model to explain.
python main.py \
    --workdir ../results/mnist/ \
    --mode=train_predict \
    --config ../configs/config_mnist.py

# train vae.
python main.py \
    --workdir ../results/mnist/ \
    --mode=train_vae \
    --config ../configs/config_mnist.py

# solve heat equation.
python main.py \
    --workdir ../results/mnist/ \
    --mode=train_heat \
    --config ../configs/config_mnist.py

# compare with other methods
# including Grad, Smooth Grad, IG, and Blur IG
python main.py \
    --workdir ../results/mnist/ \
    --mode=compare \
    --config ../configs/config_mnist.py