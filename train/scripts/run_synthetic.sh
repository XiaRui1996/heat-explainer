export CUDA_VISIBLE_DEVICES=1

# train regreesion model to explain.
python main.py \
    --workdir ../results/synthetic/ \
    --mode=train_predict \
    --config ../configs/config_synthetic.py

# vae skipped, since true manifold is known.
# solve heat equation.
python main.py \
    --workdir ../results/synthetic/ \
    --mode=train_heat \
    --config ../configs/config_synthetic.py

# compare with other methods
# including Grad, Smooth Grad, IG, and Blur IG
python main.py \
    --workdir ../results/synthetic/ \
    --mode=compare \
    --config ../configs/config_synthetic.py